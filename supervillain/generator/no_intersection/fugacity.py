#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np

from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import Form, d
from supervillain.generator.villain.site import SiteUpdate
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge
import supervillain.action


class FugacityWorm(ReadWriteable, Generator):
    r"""
    Grand-canonical defect sampler for the $q = dn \wedge dn = 0$ constraint in 4D,
    with the inline estimator of the Lagrange-multiplier correlator
    $\left\langle e^{+i\theta_{x}} e^{-i\theta_{y}} \right\rangle$.

    Where the clean-set worms (:class:`~.AdaptiveIntersectionWorm` and friends) walk a
    defect pair through the *valid* configurations and demand every step be an exact
    unit-dipole transport --- steps that often do not exist at all on the dense sheets of
    small $\kappa$ --- this sampler stops demanding cleanliness and instead *prices the
    mess*.  The underlying chain samples the enlarged ensemble

    .. math ::
        \pi(\phi, n) = e^{-S_{V}(\phi, n)}\, \zeta^{D(n)},
        \qquad D(n) = \sum_{x} \left|q_{x}(n)\right|,
        \quad q = dn \wedge dn,

    with single-link Metropolis: a link and $c = \pm 1$ are drawn uniformly (a
    symmetric proposal) and accepted with $\min\!\left(1, e^{-\Delta S}\,
    \zeta^{\Delta D}\right)$.  In a :meth:`step` the $\phi$ field is **frozen**: the
    chain then targets the conditional $\pi_{\zeta}(n \mid \phi)$, whose vacuum trace
    preserves the constrained conditional --- so the step composes Gibbs-style with a
    $\phi$-update in a :class:`~supervillain.generator.combining.Sequentially`, exactly
    like the other $n$-only worms.  (The standalone :meth:`run` interleaves its own
    :class:`~supervillain.generator.villain.SiteUpdate` sweeps so it is self-contained.)  Every link is always
    proposable --- a messy $\Delta q$ is a *legal* state with more defects,
    exponentially discounted by the fugacity $\zeta < 1$, and defect-annihilating moves
    are correspondingly rewarded.  The higher-defect sectors are the corridors through
    jammed backgrounds that clean worms lack, so the sampler cannot jam; single-link
    moves connect every $n$ with nonzero acceptance, making ergodicity on the enlarged
    space manifest.

    **As a** :class:`~supervillain.generator.Generator`, :meth:`step` advances the
    enlarged chain until its ``emit_every``-th visit to the **vacuum sector** and emits
    that configuration --- the trace of the chain on the constraint surface.  Restricted
    to the vacuum sector the enlarged weight is $e^{-S_{V}} \zeta^{0} = e^{-S_{V}}$, so
    the emitted ensemble is the **constrained theory, exactly**: like the walking worms,
    invalid states live only *inside* a step, and every emitted configuration satisfies
    $q \equiv 0$.  Because the step consumes and produces valid configurations while
    preserving the constrained measure, it composes with the other constrained
    generators in a :class:`~supervillain.generator.combining.Sequentially` --- and its
    defect excursions tunnel between valid configurations that the constrained updates
    may connect only slowly (or, for the frozen configurations, not at all).

    The correlator is read off by bookkeeping rather than steering.  Since inserting
    $e^{+i\theta_{x}} e^{-i\theta_{y}}$ shifts the constraint to
    $q = \delta_{x} - \delta_{y}$,

    .. math ::
        \left\langle e^{+i\theta_{x}} e^{-i\theta_{y}} \right\rangle
        = \frac{E\!\left[\mathbf{1}_{q = \delta_{x} - \delta_{y}}\right]}
               {\zeta^{2}\, E\!\left[\mathbf{1}_{q \equiv 0}\right]},

    tallied after every proposal.  Per step the pair-sector dwell histogram (scaled by
    the known price, $H_{\text{pair}} / V \zeta^{2}$) and the vacuum dwell are emitted
    as the inline observables ``Theta_Theta`` and ``Vacuum_Ticks``, so on an ensemble
    ``e``

    .. math ::
        G(r) = \frac{\overline{\texttt{Theta\_Theta}}(r)}{\overline{\texttt{Vacuum\_Ticks}}}

    (ratio of ensemble means; use :class:`~supervillain.analysis.Bootstrap` for
    errors).  Two properties are worth internalizing:

    * $G(0) = 1$ **identically** --- a coincident pair *is* the vacuum --- so $G$ is
      **absolutely normalized**; the $r = 0$ bin of ``Theta_Theta`` is empty by
      construction.
    * $G$ is **independent of** $\zeta$ --- the fugacity price the sampler charged the
      pair sector is divided back out --- so $\zeta$ tunes only the variance.  Running
      at two values of $\zeta$ and comparing is a sharp end-to-end exactness test.

    Tuning: entropy pushes $D$ upward (each defect may live anywhere, and the denser
    the sheet the larger a single link's $\left|\Delta D\right|$), so the right $\zeta$
    shrinks with volume and with $1/\kappa$.  Symptoms of a bad choice are loud: too
    large and $D$ pins at ``D_max`` with the vacuum never revisited (:meth:`step` then
    raises rather than hang), too small and pair excursions become needlessly rare.
    :meth:`tune` automates the choice.  ``D_max`` truncates the state space (proposals
    beyond it are ordinary zero-weight rejections); it exists to keep a badly-tuned
    chain out of the defect condensate, not for correctness.  Note that a *physical*
    defect condensate --- $\theta$ long-range order, where pairs cost $O(1)$ at any
    separation --- shows up as the tuned $\zeta$ acquiring a strong volume dependence
    and the pair dwell spreading flat in $r$; that is signal, not failure.

    $\left\langle e^{i\theta} \right\rangle$ itself vanishes identically on the torus
    (the total charge $Q = \sum_{x} q_{x}$ vanishes for every $n$, since
    $q = d(n \wedge dn)$ is exact), so the large-$r$ plateau of $G$ is the only
    order-parameter diagnostic for the $\theta$ shift symmetry.

    Besides the :class:`~supervillain.Ensemble` route, :meth:`run` +
    :meth:`correlator` drive the same chain standalone (block-jackknife errors) ---
    convenient for $\zeta$ scans and quick studies; ``example/no-intersection/
    fugacity_worm.py`` is the command-line driver.

    Parameters
    ----------
    S: a NoIntersections action
        Supplies $S_{V}$, $\kappa$, and the lattice.
    zeta: float
        The per-defect fugacity $\zeta \in (0, 1]$.
    D_max: int or None
        Hard cap on $D$; ``None`` uncaps.
    emit_every: int, optional
        Vacuum ticks per :meth:`step` emission; defaults to $4 N^{4}$ (about one
        sweep's worth of vacuum time).
    max_step_sweeps: int
        Safety horizon: :meth:`step` raises after this many sweeps without reaching
        ``emit_every`` vacuum ticks (the defect-condensation symptom).
    rng: numpy Generator, optional

    .. warning ::

        Restricted to $D = 4$.  As a :class:`~supervillain.generator.Generator` this
        updates $n$ only, so it is not ergodic on its own; at least combine it with a
        $\phi$-update such as :class:`~supervillain.generator.villain.SiteUpdate`.
    """

    def __init__(self, S, zeta, D_max=None, emit_every=None, max_step_sweeps=500,
                 rng=None):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('FugacityWorm requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('FugacityWorm is only implemented for D = 4.')
        if not (0 < zeta <= 1):
            raise ValueError(f'zeta must be in (0, 1]; got {zeta}.')

        self.S = S
        self.L = S.Lattice
        self.N = self.L.N
        self.kappa = S.kappa
        self.zeta = float(zeta)
        self.D_max = D_max
        self.emit_every = emit_every if emit_every is not None else 4 * self.N**4
        self.max_step_sweeps = max_step_sweeps
        self.rng = rng if rng is not None else np.random.default_rng()
        # Tallies for the standalone run()/correlator() path (blocked for jackknife).
        self.blocks = []            # (H_pair array, H_Z scalar) per block
        self._new_block()
        self.D_trace = []           # per-sweep defect count (diagnostic)
        self.accepted = 0
        self.proposed = 0
        # The persistent chain state used by step(); rebuilt whenever the incoming
        # configuration does not match (first call, or another generator moved it).
        self._state = None

    def __str__(self):
        return f'FugacityWorm(zeta={self.zeta}, D_max={self.D_max})'

    # ---------------------------------------------------------------- chain internals

    def _init_state(self, phi, n, update_phi):
        # The whole state the moves need, maintained incrementally so a proposal costs
        # O(1): the field strength F = dn (the per-link charge stencils read it), the
        # charge density q = dn∧dn only through its NONZERO cells (the sparse `defects`
        # dict -- on a well-tuned chain almost all of q is zero), and the scalar defect
        # count D = Σ|q| that the fugacity prices.  (The TOTAL charge Q = Σ q is
        # identically zero for every n -- q is exact -- so every sector is neutral.)
        L = self.L
        st = SimpleNamespace()
        st.n = np.asarray(n).astype(np.int64).copy()
        st.phi = np.asarray(phi).astype(float).copy()
        st.F = np.asarray(d(Form(st.n, degree=1, lattice=L))).astype(np.int64)
        q_arr = np.asarray(charge(Form(st.n, degree=1, lattice=L))).astype(np.int64)
        # defects: nonzero hypercubes, keyed by the 4-tuple cell (component axis stripped).
        st.defects = {tuple(int(x) for x in z[1:]): int(q_arr[tuple(z)])
                      for z in np.argwhere(q_arr != 0)}
        st.D = int(np.abs(q_arr).sum())
        st.site_update = SiteUpdate(self.S)
        st.site_update.rng = self.rng
        st.dphi = np.asarray(d(Form(st.phi, degree=0, lattice=L)))
        st.n_links = 4 * self.N**4
        st.update_phi = update_phi
        self._draw_batch(st)
        return st

    def _draw_batch(self, st):
        # Draw a sweep's worth of proposals up front (numpy batching); the proposal
        # distribution -- uniform link, uniform c = ±1 -- is SYMMETRIC, so plain
        # Metropolis needs no Hastings factor.  This is the whole move set: no
        # templates, no clean sets, no directions.
        rng, N = self.rng, self.N
        st.mus = rng.integers(0, 4, size=st.n_links)
        st.sites = rng.integers(0, N, size=(st.n_links, 4))
        st.cs = rng.choice((-1, 1), size=st.n_links)
        st.us = rng.uniform(0, 1, size=st.n_links)
        st.i = 0

    def _phi_sweep(self, st):
        # One φ sweep at fixed n, then refresh dphi.  φ must fluctuate or the Villain
        # weights are sampled at frozen dφ; SiteUpdate is exact for any n (it never
        # touches the constraint), and dφ is fixed during a link sweep so the cached
        # dphi array stays valid between calls.
        L = self.L
        cfg = st.site_update.step({'phi': Form(st.phi, degree=0, lattice=L),
                                   'n': Form(st.n, degree=1, lattice=L)})
        st.phi = np.asarray(cfg['phi']).astype(float)
        st.dphi = np.asarray(d(cfg['phi']))

    def _tick(self, st):
        r"""One clock tick of the enlarged chain: a single-link Metropolis proposal
        (with the $\phi$ sweep interleaved every $4 N^{4}$ ticks).  Returns
        ``(vacuum, pair_displacement)`` classifying the sector the chain sits in at
        this tick --- the raw material of every estimate this class makes."""
        if st.i == st.n_links:
            # Sweep boundary.  In the standalone run() path we refresh φ here; in the
            # step() path φ is FROZEN for the whole step -- the chain then preserves
            # the conditional π_ζ(n | φ), whose vacuum trace preserves
            # π(n | φ, q ≡ 0), so the step composes Gibbs-style with a φ-update
            # (SiteUpdate) in a Sequentially, exactly like the other n-only worms.
            self.D_trace.append(st.D)
            if st.update_phi:
                self._phi_sweep(st)
            self._draw_batch(st)
        i = st.i
        st.i += 1
        mu = int(st.mus[i])
        site = (int(st.sites[i, 0]), int(st.sites[i, 1]),
                int(st.sites[i, 2]), int(st.sites[i, 3]))
        c = int(st.cs[i])
        # The link's charge response on the CURRENT background, from the same local
        # stencils the clean worms use -- but here a messy Δq is not a rejection, it is
        # a price: ΔD counts how many units of |q| the move creates (+) or
        # annihilates (-), summed over the touched hypercubes.
        dq = local_charge.charge_change_from_link(st.F, mu, site, c, self.N)
        dD = 0
        for cell, dv in dq.items():
            q0 = st.defects.get(cell, 0)
            dD += abs(q0 + dv) - abs(q0)
        self.proposed += 1
        if self.D_max is None or st.D + dD <= self.D_max:
            # Metropolis on the ENLARGED weight e^{-S_V} ζ^D: the Villain ΔS is local
            # to this one link, and the constraint enters only through ζ^ΔD --
            # defect-annihilating moves (ΔD < 0) are REWARDED, which is what lets the
            # mess clean itself up.  (The D_max cap above is just a truncated state
            # space: proposals past it are ordinary zero-weight rejections.)
            link = (mu,) + site
            A = st.dphi[link] - 2 * np.pi * st.n[link]
            dS = (self.kappa / 2) * ((A - 2 * np.pi * c)**2 - A**2)
            if st.us[i] < np.exp(-dS) * self.zeta**dD:
                st.n[link] += c
                local_charge.apply_link_to_F(st.F, mu, site, c, self.N)
                for cell, dv in dq.items():
                    q1 = st.defects.get(cell, 0) + dv
                    if q1:
                        st.defects[cell] = q1
                    else:
                        st.defects.pop(cell, None)
                st.D += dD
                self.accepted += 1
        # ---- classify the sector at this tick (accepted or not).  This is where the
        # physics is read off: the estimator is pure bookkeeping of WHERE the chain
        # happens to sit.  A rejection is a genuine self-loop and must be counted, or
        # the dwell-time ratio is biased.
        if st.D == 0:
            # Vacuum sector: a valid q ≡ 0 configuration -- one tick of Z.
            return True, None
        if st.D == 2 and len(st.defects) == 2:
            # Exactly the worm's G-sector: a single ±1 pair.  (D == 2 alone is not
            # enough -- one cell with |q| = 2 also has D = 2.)
            (c1, v1), (c2, v2) = st.defects.items()
            if v1 == -v2 and abs(v1) == 1:
                plus, minus = (c1, c2) if v1 == 1 else (c2, c1)
                return False, tuple((plus[k] - minus[k]) % self.N for k in range(4))
        # Every other sector (4 defects, charge-2 cells, ...) is scaffolding: legal
        # states that carry the chain THROUGH jammed backgrounds but never enter the
        # estimator.
        return False, None

    # ---------------------------------------------------------------- Generator API

    def inline_observables(self, steps):
        r"""Storage for the inline ``Theta_Theta`` histogram and ``Vacuum_Ticks``."""
        return {
            'Theta_Theta': Batch(steps, shape=self.L.dims),
            'Vacuum_Ticks': Batch(steps, shape=(), dtype=float),
        }

    def step(self, configuration):
        r"""
        Advance the enlarged chain until its ``emit_every``-th vacuum tick and emit
        that configuration --- the trace of the chain on the constraint surface, so
        every emitted configuration satisfies $q \equiv 0$ exactly and the emitted
        ensemble is the constrained theory.  The pair-sector dwell accumulated along
        the way rides along as the inline ``Theta_Theta`` (already scaled by
        $1/V\zeta^{2}$) and ``Vacuum_Ticks``.
        """
        L, V = self.L, self.N**4
        n_in = np.asarray(configuration['n']).astype(np.int64)
        phi_in = np.asarray(configuration['phi']).astype(float)
        # Rebuild the persistent state if the incoming configuration is not the one we
        # left behind (first call, or another generator in a Sequentially moved it).
        st = self._state
        if st is None or not (np.array_equal(st.n, n_in)
                              and np.array_equal(st.phi, phi_in)):
            st = self._state = self._init_state(phi_in, n_in, update_phi=False)
        pair = np.zeros(L.dims)
        vacuum = 0
        cap = self.max_step_sweeps * st.n_links
        for ticks in range(1, cap + 1):
            vac, disp = self._tick(st)
            if vac:
                vacuum += 1
                if vacuum == self.emit_every:
                    break
            elif disp is not None:
                pair[disp] += 1
        else:
            raise RuntimeError(
                f'no {self.emit_every} vacuum ticks in {self.max_step_sweeps} sweeps: '
                f'zeta={self.zeta} is likely too large for this volume/kappa (defect '
                f'condensation).  Retune (FugacityWorm.tune), lower zeta, or note that '
                f'a genuinely condensed theta phase requires zeta ~ 1/V.')
        # Emit AT a vacuum tick: st.n is exactly valid here.  Copies, so the chain's
        # working arrays stay private.
        # φ was frozen for the whole step, so it passes through unchanged; only n is
        # re-emitted (a copy, so the chain's working array stays private).
        return configuration | {
            'n': Form(st.n.copy(), degree=1, lattice=L),
            'Theta_Theta': pair / (V * self.zeta**2),
            'Vacuum_Ticks': float(vacuum),
        }

    # ---------------------------------------------------------------- standalone API

    @classmethod
    def tune(cls, S, D_max=None, rng=None, phi=None, n=None,
             ladder=(0.1, 0.05, 0.02, 0.01, 0.005, 0.002),
             sweeps=60, target=0.15):
        r"""
        Pick $\zeta$ by short probes down a ladder, keeping the first value whose
        vacuum dwell exceeds ``target`` (the pair sector then follows, being the
        vacuum's nearest excursion).  Because the estimator is $\zeta$-independent,
        tuning affects only the variance, never the answer.  Physics of the knob:
        entropy pushes $D$ up, so the right $\zeta$ shrinks with volume and sheet
        density; a $\zeta$ that must shrink like $1/V$ is itself a defect-condensation
        (i.e.\ $\theta$ long-range order) diagnostic.

        Parameters
        ----------
        S: a NoIntersections action
        phi, n: arrays, optional
            A thermalized starting configuration (cold if omitted).

        Returns
        -------
        float
            The chosen $\zeta$.
        """
        L = S.Lattice
        if phi is None:
            phi = np.zeros(L.dims)
        if n is None:
            n = np.zeros((L.D,) + L.dims, dtype=np.int64)
        zeta = ladder[-1]
        for z in ladder:
            probe = cls(S, zeta=z, D_max=D_max,
                        rng=rng if rng is not None else np.random.default_rng())
            probe.run(phi, n, sweeps, tally=False)
            probe.blocks = []
            probe._new_block()
            probe.run(phi, n, sweeps, tally=True)
            probe.close_block()
            vac = probe.blocks[0][1] / max(1, probe.proposed / 2)
            zeta = z
            if vac > target:
                break
        return zeta

    def _new_block(self):
        self._H_pair = np.zeros(self.L.dims)
        self._H_Z = 0

    def close_block(self):
        r"""End the current jackknife block and start a new one (:meth:`run` path)."""
        self.blocks.append((self._H_pair, self._H_Z))
        self._new_block()

    def run(self, phi, n, sweeps, tally=True, progress=None):
        r"""
        Standalone driver: evolve ``sweeps`` sweeps --- each $4 N^{4}$ single-link
        proposals plus one $\phi$ sweep --- from ``(phi, n)``, tallying the sector
        dwell into the current jackknife block after every proposal unless ``tally``
        is ``False`` (thermalization).

        Parameters
        ----------
        phi: array
            Site angles (any real values).
        n: integer array
            Link field; need **not** satisfy the constraint.
        sweeps: int
        tally: bool
        progress: callable, optional
            e.g. ``tqdm``.

        Returns
        -------
        (phi, n)
            The evolved configuration, suitable for chaining calls.
        """
        st = self._init_state(phi, n, update_phi=True)
        iterator = range(sweeps)
        if progress is not None:
            iterator = progress(iterator)
        for _ in iterator:
            for _ in range(st.n_links):
                vac, disp = self._tick(st)
                if tally:
                    if vac:
                        self._H_Z += 1
                    elif disp is not None:
                        self._H_pair[disp] += 1
        self._phi_sweep(st)
        return st.phi, st.n

    def correlator(self):
        r"""
        The block-jackknife mean and error of the absolutely-normalized correlator
        $G(r) = H_{\text{pair}}(r) / (V \zeta^{2} H_{Z})$ over the blocks accumulated
        by :meth:`run` (skipping leave-one-out terms whose vacuum dwell vanishes ---
        only possible when $\zeta$ is badly tuned).

        Returns
        -------
        (G, dG)
            Arrays of spatial shape ``L.dims``; recall $G(0) = 1$ by definition and
            the $r = 0$ bin of $H_{\text{pair}}$ is empty by construction.
        """
        # The estimator: E[1_{pair at displacement r}] / E[1_{vacuum}] equals
        # ζ² V G(r) in the enlarged ensemble -- the pair sector carries the known
        # fugacity price ζ² (divided back out here, which is why the answer cannot
        # depend on ζ) and V translated copies contribute to each displacement bin.
        # G(0) = 1 identically: a coincident pair IS the vacuum, so no origin
        # normalization is needed -- G comes out ABSOLUTE.
        V = self.N**4
        H_pair = np.stack([b[0] for b in self.blocks])
        H_Z = np.array([b[1] for b in self.blocks], dtype=float)
        if H_Z.sum() == 0:
            nan = np.full(self.L.dims, np.nan)
            return nan, nan
        total = H_pair.sum(axis=0) / (V * self.zeta**2 * H_Z.sum())
        rows = [j for j in range(len(self.blocks)) if H_Z.sum() - H_Z[j] > 0]
        jack = np.stack([
            (H_pair.sum(axis=0) - H_pair[j]) / (V * self.zeta**2 * (H_Z.sum() - H_Z[j]))
            for j in rows])
        err = np.sqrt((len(rows) - 1) * jack.var(axis=0)) if len(rows) > 1 \
            else np.full(self.L.dims, np.nan)
        return total, err

    def report(self):
        r"""A short summary: acceptance, the defect-count trace, and (run-path) sector dwell."""
        Dt = np.array(self.D_trace)
        H_Z = sum(b[1] for b in self.blocks)
        H_pair = sum(b[0].sum() for b in self.blocks)
        lines = [f'proposals {self.proposed}  acceptance {self.accepted/max(1,self.proposed):.4f}',
                 f'defect count D: mean {Dt.mean():.2f}  max {int(Dt.max())}' if len(Dt) else 'no sweeps',
                 f'run-path sector dwell: vacuum {H_Z}  single-pair {int(H_pair)}  '
                 f'other {max(0, self.proposed - int(H_Z) - int(H_pair))}']
        return '\n'.join(lines)
