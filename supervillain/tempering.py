
import numpy as np
import h5py as h5
from scipy.special import erfc, erfcinv

from supervillain import _no_op
import supervillain
from supervillain.batch import Batch
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)


def swap_accepted(dkappa, dE, uniform):
    r'''
    The parallel-tempering swap kernel: exchanging the configurations of two rungs whose actions
    are linear in $\kappa$ changes the joint action by

    .. math::
        \Delta S = (\kappa_i - \kappa_j)(E_j - E_i)

    where $E = S/\kappa$ is each rung's energy evaluated on its own configuration,
    and the exchange is Metropolis-accepted with probability $\min(1, e^{-\Delta S})$.

    .. note::
        This is the *seam* every tempering backend shares: the arithmetic order
        (``dS = dkappa * dE``) is part of the contract, so that a distributed
        backend reproduces the serial ladder's decisions bitwise.

    Parameters
    ----------
    dkappa: float
        $\kappa_i - \kappa_j$.
    dE: float
        $E_j - E_i$.
    uniform: float
        A uniform random number in $[0, 1)$; exactly one is consumed per attempt,
        whether or not the acceptance is a foregone conclusion.

    Returns
    -------
    bool
        Should the two configurations be exchanged?
    '''
    dS = dkappa * dE
    return (dS <= 0) or (uniform < np.exp(-dS))


class EvenOddPairs(ReadWriteable):
    r'''
    The deterministic swap schedule: on even sweeps the pairs are $(0,1), (2,3), \ldots$
    and on odd sweeps $(1,2), (3,4), \ldots$, so every adjacent pair is attempted every
    other sweep and no rung appears in two pairs at once.

    Parameters
    ----------
    rungs: int
        The number of rungs in the ladder.
    '''

    def __init__(self, rungs):
        self.rungs = rungs

    def __str__(self):
        return f'EvenOddPairs({self.rungs})'

    def pairs(self, sweep):
        r'''
        Parameters
        ----------
        sweep: int
            The global sweep index; only its parity matters.

        Returns
        -------
        tuple of (int, int)
            The adjacent pairs $(i, i+1)$ attempted on this sweep.
        '''
        return tuple((i, i + 1) for i in range(sweep % 2, self.rungs - 1, 2))


class TemperedRung(ReadWriteable, Generator):
    r'''
    A generator-shaped marker stored on each rung's ensemble by :class:`ParallelTempering`.

    A tempered rung is not a Markov chain by itself --- its invariant distribution is
    only correct jointly with the rest of the ladder --- so continuing a single rung
    with :meth:`Ensemble.continue_from <supervillain.ensemble.Ensemble.continue_from>`
    would silently drop the tempering and produce wrong physics.  This marker refuses
    to ``step`` (pointing at :meth:`ParallelTempering.continue_from`) while delegating
    the bookkeeping every stored generator provides, and its attributes identify the
    ensemble as tempered in HDF5.

    Parameters
    ----------
    generator:
        The rung's local update stack.
    rung: int
        This rung's position in the ladder.
    ladder: iterable of float
        The κs of the whole ladder, in order.
    '''

    def __init__(self, generator, rung, ladder):
        self.generator = generator
        self.rung = rung
        self.ladder = np.array(ladder)

    def __str__(self):
        return f'TemperedRung({self.rung} of κ={self.ladder}, {str(self.generator)})'

    def step(self, cfg):
        r'''
        Raises
        ------
        RuntimeError
            Always: resume the whole ladder with :meth:`ParallelTempering.continue_from`.
        '''
        raise RuntimeError(
            'A tempered rung is not a Markov chain by itself; '
            'resume the whole ladder with ParallelTempering.continue_from.'
        )

    def inline_observables(self, steps):
        r'''
        The local stack's inline observables plus the ladder's ``Replica`` label.
        '''
        return self.generator.inline_observables(steps) | {
            'Replica': Batch(steps, shape=(), dtype=int),
        }

    def report(self):
        r'''
        Returns the local stack's report.
        '''
        return self.generator.report()


class ParallelTempering:
    r'''
    Parallel tempering across a ladder of actions that differ only in $\kappa$
    (and, harmlessly, in proposal machinery): every rung applies its own local update
    stack once per step, then adjacent rungs attempt to exchange whole configurations
    according to :func:`swap_accepted` and :class:`EvenOddPairs`.

    Configurations are exchanged, never κ labels, so each rung's :class:`~.Ensemble`
    stays at fixed κ and every downstream tool (HDF5, :class:`~.Bootstrap`,
    observables) applies per rung unchanged.  Because the actions are linear in κ
    and the NoIntersections constraint is κ-independent, a configuration valid at one
    rung is valid at every rung and the swap decision needs only the scalar
    $E = S/\kappa$ from each rung.

    A ``Replica`` label rides inside each configuration dictionary (initialized to the
    rung index at a cold start), so swaps transport it automatically and each rung's
    stored ``Replica`` trace records which walker occupied it at every step.  Like the
    generators' ``Ticks``, it is ladder bookkeeping with no Observable class.

    .. note::
        This class is a pure composite-kernel driver: it has no notion of
        thermalization or tuning.  Drive those as separate *legs* --- generate a
        thermalization ladder, tune each rung's proposal machinery on its last
        configuration, then start a production ``ParallelTempering`` with
        ``start=[e.configuration[-1] for e in thermalization_ensembles]``.
        Swap acceptance depends only on (κ, E), never on proposal parameters, so
        per-rung proposal settings may differ freely and change between legs.

    .. seealso::
        The design spec ``docs/superpowers/specs/2026-07-15-parallel-tempering-design.md``
        records the architecture decision and the seam contract for future
        distributed backends.

    Parameters
    ----------
    actions:
        One action per rung, strictly ascending in ``kappa``, all sharing a lattice
        and field content, each linear in κ (the Villain family).
    generators:
        One local update stack per rung; ordinary generators, ignorant of tempering.
    seed:
        Seeds a :class:`numpy.random.SeedSequence` from which one child stream per
        adjacent pair is spawned; every attempt on pair $(i, i+1)$ consumes exactly
        one uniform from child $i$.  Local generators keep their own rngs.
    '''

    def __init__(self, actions, generators, seed=None):

        self.Actions = tuple(actions)
        self.generators = tuple(generators)

        if len(self.Actions) < 2:
            raise ValueError('Parallel tempering needs at least two rungs.')
        if len(self.Actions) != len(self.generators):
            raise ValueError(f'{len(self.Actions)} actions but {len(self.generators)} generators.')

        self.kappa = np.array([S.kappa for S in self.Actions])
        r'''The κ ladder, ascending.'''
        if not (np.diff(self.kappa) >= 0).all():
            raise ValueError(f'The κ ladder must be ascending; got {self.kappa}.')

        lattices = set(id(S.Lattice) for S in self.Actions)
        if len(lattices) > 1:
            raise ValueError('All rungs must share one lattice.')

        self.schedule = EvenOddPairs(len(self.Actions))
        r'''The swap schedule.'''
        self.sweep = 0
        r'''The global sweep counter; its parity feeds the schedule, and it continues across legs.'''

        self.seed = seed
        pairs = len(self.Actions) - 1
        self._pair_rng = tuple(
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(pairs)
        )

        self.attempted = np.zeros(pairs, dtype=int)
        r'''Swap attempts per adjacent pair.'''
        self.accepted = np.zeros(pairs, dtype=int)
        r'''Accepted swaps per adjacent pair.'''

        self._crossings = np.zeros(len(self.Actions), dtype=int)
        self._last_endpoint = np.full(len(self.Actions), -1, dtype=int)

    def __str__(self):
        return f'ParallelTempering(κ={self.kappa})'

    @property
    def pair_acceptance(self):
        r'''
        Accepted/attempted per adjacent pair; ``nan`` where nothing was attempted yet.
        '''
        with np.errstate(invalid='ignore'):
            return np.where(self.attempted > 0, self.accepted / np.maximum(self.attempted, 1), np.nan)

    @property
    def round_trips(self):
        r'''
        Completed bottom↔top↔bottom round trips, summed over replicas
        (each end-to-end crossing counts half a round trip).
        '''
        return int(self._crossings.sum()) // 2

    def _energies(self, current):
        return np.array([S(**cfg) / S.kappa for S, cfg in zip(self.Actions, current)])

    def _swap_sweep(self, current):
        E = self._energies(current)

        for (i, j) in self.schedule.pairs(self.sweep):
            uniform = self._pair_rng[i].uniform()
            self.attempted[i] += 1
            if swap_accepted(self.kappa[i] - self.kappa[j], E[j] - E[i], uniform):
                current[i], current[j] = current[j], current[i]
                E[i], E[j] = E[j], E[i]
                self.accepted[i] += 1

        self.sweep += 1

        for replica, endpoint in ((int(current[0]['Replica']), 0), (int(current[-1]['Replica']), 1)):
            if self._last_endpoint[replica] != endpoint:
                if self._last_endpoint[replica] != -1:
                    self._crossings[replica] += 1
                self._last_endpoint[replica] = endpoint

        return current

    def generate(self, steps, start='cold', progress=_no_op, starting_index=0, index_stride=1):
        r'''
        Mirrors :meth:`Ensemble.generate <supervillain.ensemble.Ensemble.generate>` rung
        by rung, in lockstep, with a swap sweep between every local sweep and emission.

        Parameters
        ----------
        steps: int
            Number of configurations to generate on every rung.
        start: 'cold', or a list of configuration dictionaries, one per rung
            A cold start begins every rung with the all-zero configuration.
            A list of dictionaries seeds each rung --- this is how a production leg
            continues from a thermalization leg's last configurations.
        progress: something which wraps an iterator and provides a progress bar.
            As in :meth:`Ensemble.generate <supervillain.ensemble.Ensemble.generate>`.
        starting_index: int
            The lower value of every rung's ``.index``.
        index_stride: int
            The ``.index`` increment per step.

        Returns
        -------
        list of supervillain.Ensemble
            One standard fixed-κ ensemble per rung, each carrying its
            :class:`TemperedRung` marker as its generator.
        '''

        rungs = len(self.Actions)

        configurations = []
        for S, G in zip(self.Actions, self.generators):
            c = S.configurations(steps)
            c |= G.inline_observables(steps)
            c |= {'Replica': Batch(steps, shape=(), dtype=int)}
            configurations.append(c)

        if isinstance(start, str) and start == 'cold':
            current = [S.configurations(1)[0] for S in self.Actions]
        elif isinstance(start, (list, tuple)):
            if len(start) != rungs:
                raise ValueError(f'{rungs} rungs but {len(start)} starting configurations.')
            current = [dict(s) for s in start]
        else:
            raise ValueError(f'Not sure how to start {rungs} rungs from {type(start)}.')

        for i, cfg in enumerate(current):
            if 'Replica' not in cfg:
                cfg['Replica'] = i

        with Timer(logger.info, f'Generation of {steps} configurations on {rungs} rungs', per=steps):

            for step in progress(range(steps), desc='Tempered generation'):
                current = [G.step(cfg) for G, cfg in zip(self.generators, current)]
                current = self._swap_sweep(current)
                for i in range(rungs):
                    configurations[i][step] = current[i]

        ensembles = []
        for i, (S, G) in enumerate(zip(self.Actions, self.generators)):
            e = supervillain.Ensemble(S)
            e.configuration = configurations[i]
            e.index_stride = index_stride
            e.index = Batch(starting_index + index_stride * np.arange(steps))
            # .weight is derived from logWeight_* columns; a plain rung carries none → ones.
            e.start = start if (isinstance(start, str)) else start[i]
            e.generator = TemperedRung(G, rung=i, ladder=self.kappa)
            ensembles.append(e)

        for line in self.report().split('\n'):
            logger.info(line)

        self.ensembles = ensembles
        return ensembles

    @classmethod
    def continue_from(cls, ensembles, steps, seed=None, progress=_no_op):
        r'''
        Resume a whole tempered ladder from the per-rung ensembles a previous
        :meth:`generate` produced (or their h5 groups): every rung continues from its
        last configuration, the ``.index`` continues, and the swap schedule resumes
        with the correct parity.

        .. note::
            A fresh ladder ``seed`` is statistically sound --- chain correctness never
            depends on continuing an rng stream --- but bitwise reproduction of an
            uninterrupted run additionally requires re-seeding deterministically.

        Parameters
        ----------
        ensembles: list of supervillain.Ensemble or h5py.Group
            The rungs of the ladder, in order, each carrying a :class:`TemperedRung`.
        steps: int
            Number of new configurations per rung.
        seed:
            As in the constructor.
        progress:
            As in :meth:`generate`.

        Returns
        -------
        list of supervillain.Ensemble
            ``steps`` new configurations per rung.
        '''

        es = [supervillain.Ensemble.from_h5(e) if isinstance(e, h5.Group) else e for e in ensembles]

        for e in es:
            if not isinstance(e.generator, TemperedRung):
                raise ValueError(
                    'Every rung must carry a TemperedRung marker; '
                    'these ensembles were not generated by ParallelTempering.'
                )

        pt = cls([e.Action for e in es], [e.generator.generator for e in es], seed=seed)

        last_index = es[0].index[-1]
        stride = es[0].index_stride
        pt.sweep = int(last_index) // int(stride) + 1

        return pt.generate(
            steps,
            start=[e.configuration[-1] for e in es],
            progress=progress,
            starting_index=int(last_index) + int(stride),
            index_stride=int(stride),
        )

    def report(self):
        r'''
        Returns a string summarizing per-pair swap acceptances and replica round trips.
        '''
        lines = [f'Parallel tempering over κ = {self.kappa} after {self.sweep} sweeps:']
        for i in range(len(self.Actions) - 1):
            lines.append(
                f'    swap κ={self.kappa[i]:g} ↔ κ={self.kappa[i+1]:g}: '
                f'{self.accepted[i]} / {self.attempted[i]} accepted'
                + (f' = {self.accepted[i]/self.attempted[i]:.4f}' if self.attempted[i] > 0 else '')
            )
        lines.append(f'    {self.round_trips} completed replica round trips.')
        return '\n'.join(lines)


class ParallelTemperingTuner:
    r'''
    Recommends a κ ladder from a short pilot leg.

    This is the tempering-specific tuning problem --- *where the rungs sit* --- and it
    is action-agnostic: acceptance between neighbors depends only on $\Delta\kappa$
    and the distributions of $E = S/\kappa$.  Proposal tuning (fugacities and the
    like) is per-action and belongs to the driver with the per-action tuners.

    The recipe: estimate the thermodynamic length $\lambda(\kappa) = \int \sigma_E\, d\kappa$
    by trapezoid over the pilot rungs, calibrate the measured pair acceptances against
    the pair spacings $\Delta\lambda$ through $\mathrm{acceptance} \approx \mathrm{erfc}(c\,\Delta\lambda)$,
    and place new rungs (endpoints fixed) at equal increments of $\lambda$ sized to a
    target acceptance.

    .. note::
        Only pairs with at least one but not every swap accepted inform the
        calibration.  For Gaussian $E$ the acceptance is exactly
        $\mathrm{erfc}(\Delta\lambda/2)$, so when the pilot has no informative
        pair at all (too coarse to accept a single swap anywhere) the tuner uses
        the Gaussian value $c = 1/2$ --- a hopelessly-spaced pilot still yields a
        sensible dense ladder from its measured $\sigma_E$ alone.

    .. note::
        The tuner recommends; the driver decides.  Rebuild actions at the recommended
        κs through your own factory (e.g. ``lambda kappa: NoIntersections(L, kappa)``)
        --- the tuner never constructs or inspects actions.

    Parameters
    ----------
    tempering: ParallelTempering
        A ladder whose :meth:`~.ParallelTempering.generate` has run a pilot leg
        (so it holds ``.ensembles``, ``.pair_acceptance``, and the κs).
    cut: int
        Configurations to drop from the start of every pilot ensemble before
        measuring $\sigma_E$ --- a cold-started pilot's thermalization transient
        otherwise inflates $\sigma_E$ and over-densifies the recommendation.
    '''

    def __init__(self, tempering, cut=0):

        self.kappa = tempering.kappa
        self.acceptance = tempering.pair_acceptance

        self.E = tuple(
            np.array([S(**e.configuration[t]) for t in range(cut, len(e))]) / S.kappa
            for S, e in zip(tempering.Actions, tempering.ensembles)
        )
        self.sigma = np.array([E.std() for E in self.E])
        r'''Per-rung standard deviation of $E$ over the pilot.'''

        # λ(κ) by trapezoid, on the pilot's rungs.
        self.length = np.concatenate([
            [0.], np.cumsum(0.5 * (self.sigma[1:] + self.sigma[:-1]) * np.diff(self.kappa))
        ])
        r'''Cumulative thermodynamic length at each pilot rung.'''

        # Calibrate acceptance ≈ erfc(c Δλ) through the origin from the measured pairs.
        # A pair with zero (or every) attempt accepted carries no calibration
        # information --- only a bound --- and erfcinv of a clipped stand-in value
        # would pollute the fit, so such pairs are excluded.  When nothing
        # informative remains (a hopelessly coarse pilot), fall back to the Gaussian
        # theory value: for Gaussian E the acceptance is exactly erfc(Δλ/2)
        # (mean ΔS = Δλ², std √2 Δλ), i.e. c = 1/2.
        GAUSSIAN = 0.5
        dlambda = np.diff(self.length)
        informative = (tempering.accepted > 0) & (tempering.accepted < tempering.attempted) \
                      & (dlambda > 0)
        if not informative.any():
            logger.warning('The pilot has no informative pair acceptances; '
                           'falling back to the Gaussian calibration c = 1/2.')
            self.calibration = GAUSSIAN
        else:
            measured = self.acceptance[informative]
            self.calibration = ((erfcinv(measured) * dlambda[informative]).sum()
                                / (dlambda[informative] ** 2).sum())

    def ladder(self, target=0.25, rungs=None):
        r'''
        Parameters
        ----------
        target: float
            The desired per-pair acceptance (used to choose the rung count when
            ``rungs`` is ``None``).
        rungs: int or None
            Force the rung count; the target then only informs the caller's judgment.

        Returns
        -------
        np.ndarray
            Recommended κs, endpoints fixed, spaced at equal thermodynamic length.
        '''

        if rungs is None:
            dlambda = erfcinv(target) / self.calibration if self.calibration > 0 else np.inf
            rungs = max(2, int(np.ceil(self.length[-1] / dlambda)) + 1)

        targets = np.linspace(0., self.length[-1], rungs)
        return np.interp(targets, self.length, self.kappa)

    def predicted_acceptance(self, kappas):
        r'''
        Parameters
        ----------
        kappas: iterable of float
            A candidate ladder.

        Returns
        -------
        np.ndarray
            The calibrated $\mathrm{erfc}(c\,\Delta\lambda)$ prediction per adjacent pair.
        '''
        lam = np.interp(kappas, self.kappa, self.length)
        return erfc(self.calibration * np.diff(lam))

    def report(self):
        r'''
        Returns a string summarizing the pilot and the default recommendation.
        '''
        recommended = self.ladder()
        return '\n'.join([
            f'Pilot ladder κ = {self.kappa}',
            f'    pair acceptances {self.acceptance}',
            f'    σ_E per rung {self.sigma}',
            f'    thermodynamic length {self.length[-1]:g}',
            f'Recommended ladder (target 0.25): κ = {recommended}',
            f'    predicted acceptances {self.predicted_acceptance(recommended)}',
        ])
