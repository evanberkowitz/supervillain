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
from supervillain.generator.no_intersection import defect_gas_kernel
import supervillain.action


class DefectGas(ReadWriteable, Generator):
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
        \Pi = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]}\, \zeta^{D(n)},
        \qquad D(n) = \sum_{x} \left|q_{x}(n)\right|,
        \quad q = dn \wedge dn,

    with $\zeta$ conjugate to each *insertion* of the charge operator $e^{\pm i\theta}$.
    $D$ is always even ($\left|q\right| \equiv q \bmod 2$ sitewise and the total charge
    vanishes identically), so $D/2$ counts the $\pm$ *pairs in flight* --- the number of
    worms the grand-canonical ensemble runs at once.

    Standard single-link updates $n\to n \pm 1$ are Metropolis tested with the 
    fugacity included in the weight and are accepted with probability $\min\!\left(1, e^{-\Delta S}\,
    \zeta^{\Delta D}\right)$.  Every link can always receive a proposal but a proposal with
    a messy $\Delta q$ is exponentially discounted by the fugacity $\zeta < 1$, and defect-annihilating moves
    are correspondingly rewarded.  The higher-defect sectors are the corridors through
    jammed backgrounds that clean worms lack, so the sampler cannot jam; single-link
    moves connect every $n$ with nonzero acceptance, making ergodicity on the enlarged
    space manifest.

    While generating, the :meth:`step` advances the the enlarged chain until its
    ``emit_every``th visit to the vacuum sector and emits that configuration.
    Restricted to the vacuum sector the enlarged weight is $e^{-S_{V}} \zeta^{0} = e^{-S_{V}}$,
    so the emitted ensemble is exactly the constrained theory.  Like the worms,
    invalid states live only *inside* a step, and every emitted configuration satisfies
    $q \equiv 0$ and therefore this generator can be combined with other constrained generators.

    The correlator is read off by bookkeeping rather than steering.  Since inserting
    $e^{+i\theta_{x}} e^{-i\theta_{y}}$ shifts the constraint to
    $q = \delta_{x} - \delta_{y}$, tallying after every proposal which sector the
    chain sits in gives

    .. math ::
        \Theta_{x,y}
        = \frac{\left\langle \prod_{p} [q_{p} = \delta_{px} - \delta_{py}] \right\rangle_{\Pi}}
               {\zeta^{2} \left\langle \prod_{p} [q_{p} = 0] \right\rangle_{\Pi}},

    with $[\cdots]$ the Iverson bracket: the ratio of the time spent in the exact
    single-pair sector to the time spent in the vacuum, with the known price
    $\zeta^{2}$ divided back out.  Per step the pair-sector dwell histogram (scaled by
    that price, $H_{\text{pair}} / V \zeta^{2}$) and the vacuum dwell are emitted as
    the inline observables ``Theta_Theta`` and ``Vacuum_Ticks``, so on an ensemble
    ``e``

    .. math ::
        \Theta_{\Delta x} = \frac{\overline{\texttt{Theta\_Theta}}_{\Delta x}}{\overline{\texttt{Vacuum\_Ticks}}}

    (ratio of ensemble means; use :class:`~supervillain.analysis.Bootstrap` for
    errors).  Two properties are worth internalizing:

    * $\Theta_{0} = 1$ **identically** --- a coincident pair *is* the vacuum --- so
      $\Theta$ is **absolutely normalized**; the $\Delta x = 0$ bin of ``Theta_Theta``
      is empty by construction.
    * $\Theta$ is **independent of** $\zeta$ --- the fugacity price the sampler charged
      the pair sector is divided back out --- so $\zeta$ tunes only the variance.
      Running at two values of $\zeta$ and comparing is a sharp end-to-end exactness
      test.

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
    and the pair dwell spreading flat in $\Delta x$; that is signal, not failure.

    $\left\langle e^{i\theta} \right\rangle$ itself vanishes identically on the torus
    (the total charge $Q = \sum_{x} q_{x}$ vanishes for every $n$, since
    $q = d(n \wedge dn)$ is exact), so the large-$\Delta x$ plateau of $\Theta$ is the
    only order-parameter diagnostic for the $\theta$ shift symmetry.

    Besides the :class:`~supervillain.Ensemble` route, :meth:`run` +
    :meth:`correlator` drive the same chain standalone (block-jackknife errors) ---
    convenient for $\zeta$ scans and quick studies; ``defect_gas.py`` in the
    companion supervillain-no-intersections repository is the command-line driver.

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

    # D == 4 sector classes for the fourth moment, keyed by the sorted charge values:
    # index 0: {+1,+1,-1,-1}, 1: {+2,-1,-1}, 2: {+1,+1,-2}, 3: {+2,-2}.
    _FOUR = {(-1, -1, 1, 1): 0, (-1, -1, 2): 1, (-2, 1, 1): 2, (-2, 2): 3}

    def __init__(self, S, zeta, D_max=None, emit_every=None, max_step_sweeps=500,
                 rng=None):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('DefectGas requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('DefectGas is only implemented for D = 4.')
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
        return f'DefectGas(zeta={self.zeta}, D_max={self.D_max})'

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
        # 0-forms carry a leading singleton component axis, (1,) + dims; accept bare
        # (N, ..., N) arrays too (the cold default) by restoring it.
        if st.phi.shape == tuple(L.dims):
            st.phi = st.phi.reshape((1,) + tuple(L.dims))
        st.F = np.asarray(d(Form(st.n, degree=1, lattice=L))).astype(np.int64)
        q_arr = np.asarray(charge(Form(st.n, degree=1, lattice=L))).astype(np.int64)
        # defects: nonzero hypercubes, keyed by the 4-tuple cell (component axis stripped).
        st.defects = {tuple(int(x) for x in z[1:]): int(q_arr[tuple(z)])
                      for z in np.argwhere(q_arr != 0)}
        st.D = int(np.abs(q_arr).sum())
        # Flat views and the dense charge state for the compiled tick loop (step); the
        # sparse dict above serves the pure-python step_reference.  n2/F2 are VIEWS, so
        # kernel writes keep st.n/st.F current; q/nzc mirror the dict and are kept in
        # sync by BOTH paths so step and step_reference may be freely interleaved.
        st.F2 = st.F.reshape(st.F.shape[0], -1)
        st.n2 = st.n.reshape(4, -1)
        st.q = np.ascontiguousarray(q_arr.ravel())
        st.nzc = np.zeros(self.N**4, dtype=np.int64)
        nz = np.flatnonzero(st.q)
        st.nzc[:len(nz)] = nz
        st.nnz = int(len(nz))
        # Transport instrumentation, shared by both paths: [current excursion length,
        # completed excursions, max single-pair min-image separation squared] and the
        # power-of-two excursion-length histogram.  These turn empty far bins into
        # honest transport-censoring statements instead of silent zeros.
        st.tstate = np.zeros(3, dtype=np.int64)
        st.exc_hist = np.zeros(32, dtype=np.int64)
        st.site_update = SiteUpdate(self.S)
        st.site_update.rng = self.rng
        st.dphi = np.ascontiguousarray(d(Form(st.phi, degree=0, lattice=L)))
        st.dphi2 = st.dphi.reshape(4, -1)
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
        st.dphi = np.ascontiguousarray(d(cfg['phi']))
        st.dphi2 = st.dphi.reshape(4, -1)

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
                N = self.N
                for cell, dv in dq.items():
                    q1 = st.defects.get(cell, 0) + dv
                    if q1:
                        st.defects[cell] = q1
                    else:
                        st.defects.pop(cell, None)
                    # Mirror into the dense q and compact nonzero list the compiled
                    # step uses, so the two paths can be freely interleaved.
                    rav = ((cell[0] * N + cell[1]) * N + cell[2]) * N + cell[3]
                    if st.q[rav] == 0:
                        st.nzc[st.nnz] = rav
                        st.nnz += 1
                    elif q1 == 0:
                        for b in range(st.nnz):
                            if st.nzc[b] == rav:
                                st.nzc[b] = st.nzc[st.nnz - 1]
                                st.nnz -= 1
                                break
                    st.q[rav] = q1
                st.D += dD
                self.accepted += 1
        # ---- classify the sector at this tick (accepted or not).  This is where the
        # physics is read off: the estimator is pure bookkeeping of WHERE the chain
        # happens to sit.  A rejection is a genuine self-loop and must be counted, or
        # the dwell-time ratio is biased.
        if st.D == 0:
            # Vacuum sector: a valid q ≡ 0 configuration -- one tick of Z.  Close any
            # excursion: power-of-two length bin (bit_length), top bin saturating.
            if st.tstate[0] > 0:
                st.exc_hist[min(31, int(st.tstate[0]).bit_length())] += 1
                st.tstate[1] += 1
                st.tstate[0] = 0
            return True, None, None
        st.tstate[0] += 1
        if st.D == 2 and len(st.defects) == 2:
            # Exactly the worm's G-sector: a single ±1 pair.  (D == 2 alone is not
            # enough -- one cell with |q| = 2 also has D = 2.)
            (c1, v1), (c2, v2) = st.defects.items()
            if v1 == -v2 and abs(v1) == 1:
                plus, minus = (c1, c2) if v1 == 1 else (c2, c1)
                disp = tuple((plus[k] - minus[k]) % self.N for k in range(4))
                # Min-image separation squared: the transport ceiling.
                rsq = sum(min(d, self.N - d)**2 for d in disp)
                if rsq > st.tstate[2]:
                    st.tstate[2] = rsq
                return (False, disp, None)
        if st.D == 4:
            # The two-pair sectors that feed <|M|^4> for the Binder cumulant.  Net-zero
            # D = 4 patterns come in exactly four geometric classes, keyed by the sorted
            # charge multiset; their ordered-insertion multiplicities (4, 2, 2, 1) enter
            # the Binder formula, not the tally.
            cls = self._FOUR.get(tuple(sorted(st.defects.values())))
            if cls is not None:
                return False, None, cls
        # Every other sector is scaffolding: legal states that carry the chain THROUGH
        # jammed backgrounds but never enter the estimators.
        return False, None, None

    # ---------------------------------------------------------------- Generator API

    def inline_observables(self, steps):
        r"""Storage for the inline ``Theta_Theta`` histogram, ``Vacuum_Ticks``, the
        four-defect classes, and the transport diagnostics."""
        return {
            'Theta_Theta': Batch(steps, shape=self.L.dims),
            'Vacuum_Ticks': Batch(steps, shape=(), dtype=float),
            'Four_Defect': Batch(steps, shape=(4,), dtype=float),
            'Pair_Excursions': Batch(steps, shape=(), dtype=float),
            'Max_Pair_RSq': Batch(steps, shape=(), dtype=float),
            'Excursion_Lengths': Batch(steps, shape=(32,), dtype=float),
        }

    def _kernel_ticks(self, st, tally, vac_stop, H_pair, H_four):
        # One compiled pass over the remainder of the current proposal batch (stopping
        # early at vac_stop vacuum ticks when positive); mutates the dense chain state
        # in place and returns the number of vacuum ticks seen.
        i0 = st.i
        i, D, nnz, acc, vac = defect_gas_kernel.tick_batch(
            st.F2, st.n2, st.dphi2, st.q, st.nzc, st.D, st.nnz,
            st.mus, st.sites, st.cs, st.us, i0,
            self.kappa, self.zeta,
            -1 if self.D_max is None else int(self.D_max), self.N,
            *defect_gas_kernel.stencil_pack(),
            H_pair, H_four, tally, vac_stop, st.tstate, st.exc_hist)
        st.i = int(i)
        st.D = int(D)
        st.nnz = int(nnz)
        self.proposed += st.i - i0
        self.accepted += int(acc)
        return int(vac)

    def _step_body(self, configuration, ticker):
        # Shared frame of step and step_reference: state (re)build, the advance-until-
        # emit loop via `ticker`, and the emission.  `ticker` advances the chain and
        # returns (vacuum ticks seen, ticks consumed), tallying into its arguments.
        L, V = self.L, self.N**4
        n_in = np.asarray(configuration['n']).astype(np.int64)
        phi_in = np.asarray(configuration['phi']).astype(float)
        # Rebuild the persistent state if the incoming configuration is not the one we
        # left behind (first call, or another generator in a Sequentially moved it).
        st = self._state
        if st is None or not (np.array_equal(st.n, n_in)
                              and np.array_equal(st.phi, phi_in)):
            st = self._state = self._init_state(phi_in, n_in, update_phi=False)
        H_pair = np.zeros(V, dtype=np.int64)
        H_four = np.zeros(4, dtype=np.int64)
        # Per-step transport bookkeeping: the max separation resets each step; the
        # excursion count and length histogram are emitted as this step's increments.
        st.tstate[2] = 0
        exc0 = int(st.tstate[1])
        hist0 = st.exc_hist.copy()
        vacuum = 0
        ticks = 0
        cap = self.max_step_sweeps * st.n_links
        while vacuum < self.emit_every:
            if ticks >= cap:
                raise RuntimeError(
                    f'no {self.emit_every} vacuum ticks in {self.max_step_sweeps} sweeps: '
                    f'zeta={self.zeta} is likely too large for this volume/kappa (defect '
                    f'condensation).  Retune (DefectGas.tune), lower zeta, or note that '
                    f'a genuinely condensed theta phase requires zeta ~ 1/V.')
            v, t = ticker(st, self.emit_every - vacuum, H_pair, H_four)
            vacuum += v
            ticks += t
        # Emit AT a vacuum tick: st.n is exactly valid here.
        # φ was frozen for the whole step, so it passes through unchanged; only n is
        # re-emitted (a copy, so the chain's working array stays private).
        return configuration | {
            'n': Form(st.n.copy(), degree=1, lattice=L),
            'Theta_Theta': H_pair.reshape(tuple(L.dims)) / (V * self.zeta**2),
            'Vacuum_Ticks': float(vacuum),
            'Four_Defect': H_four / self.zeta**4,
            'Pair_Excursions': float(st.tstate[1] - exc0),
            'Max_Pair_RSq': float(st.tstate[2]),
            'Excursion_Lengths': (st.exc_hist - hist0).astype(float),
        }

    def step(self, configuration):
        r"""
        Advance the enlarged chain until its ``emit_every``-th vacuum tick and emit
        that configuration --- the trace of the chain on the constraint surface, so
        every emitted configuration satisfies $q \equiv 0$ exactly and the emitted
        ensemble is the constrained theory.  The pair-sector dwell accumulated along
        the way rides along as the inline ``Theta_Theta`` (already scaled by
        $1/V\zeta^{2}$) and ``Vacuum_Ticks``.

        The tick loop runs in a compiled kernel
        (:func:`~supervillain.generator.no_intersection.defect_gas_kernel.tick_batch`)
        consuming the same pre-drawn proposals as the pure-python
        :meth:`step_reference`, which it reproduces bit-for-bit.
        """
        def ticker(st, vac_stop, H_pair, H_four):
            if st.i == st.n_links:
                self.D_trace.append(st.D)
                self._draw_batch(st)
            i0 = st.i
            vac = self._kernel_ticks(st, True, vac_stop, H_pair, H_four)
            return vac, st.i - i0
        out = self._step_body(configuration, ticker)
        # Mirror the dense charge state back into the sparse dict so step_reference
        # can pick up where the kernel left off.
        st, dims = self._state, tuple(self.L.dims)
        st.defects = {
            tuple(int(x) for x in np.unravel_index(int(cell), dims)): int(st.q[cell])
            for cell in st.nzc[:st.nnz]}
        return out

    def step_reference(self, configuration):
        r"""
        The plain, obviously-correct :meth:`step`: the same advance-until-emit loop
        driven tick-by-tick through the pure-python :meth:`_tick` (sparse defect dict,
        per-proposal stencil dicts).  Kept as the correctness oracle the compiled
        :meth:`step` is validated against --- same proposals, same accept test, same
        tallies, bit-for-bit.
        """
        def ticker(st, vac_stop, H_pair, H_four):
            vac, disp, cls4 = self._tick(st)
            if vac:
                return 1, 1
            if disp is not None:
                N = self.N
                H_pair[((disp[0] * N + disp[1]) * N + disp[2]) * N + disp[3]] += 1
            elif cls4 is not None:
                H_four[cls4] += 1
            return 0, 1
        return self._step_body(configuration, ticker)

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
            phi = np.zeros((1,) + tuple(L.dims))
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

    @classmethod
    def tune_edge(cls, S, D_max=32, rng=None, phi=None, n=None,
                  ladder=(0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05,
                          0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3),
                  min_vacuum_ticks=500, probe_sweeps=200, max_probe_sweeps=4000,
                  step_sweeps=25, floor=None):
        r"""
        Ride the edge: pick the **largest** $\zeta$ at which the chain still
        *demonstrably* returns to the vacuum, ascending the ladder until it does not.

        Where :meth:`tune` optimizes the vacuum clock (dwell $> 15\%$: cheap steps,
        best denominator statistics), this optimizes the *numerator's reach*: pair
        creation and --- through the paid corridors of a dense sheet --- pair
        *transport* both scale like $\zeta^{2}$, so far-separation dwell responds
        $\sim \zeta^{4}$, and a conservatively small $\zeta$ silently censors exactly
        the large-$\Delta x$ bins that diagnose $\theta$ order.

        There is deliberately **no dwell percentage** here.  The vacuum sector is not
        optional --- :meth:`step` can only emit at a vacuum tick --- but per tick the
        chain costs the same at any dwell and the step path's denominator is exact
        (``Vacuum_Ticks`` $\equiv$ ``emit_every``), so the only *hard* floor is
        measurability: a rung is accepted iff its probe collects
        ``min_vacuum_ticks`` vacuum ticks within ``max_probe_sweeps`` sweeps
        (implicitly dwell $\gtrsim$ ``min_vacuum_ticks / (max_probe_sweeps
        \cdot 4V)`` --- $\sim 10^{-5}$ at $N = 8$ with the defaults) **and** the
        dwell is stationary across the probe (second half at least a quarter of the
        first: a collapsing dwell is condensation in progress, not an equilibrium
        rate).  An explicit ``floor`` may still be imposed on top.

        Because a step still needs ``emit_every`` vacuum ticks, the emission cadence
        shrinks with the dwell: the returned ``emit_every`` is sized so one
        :meth:`step` costs about ``step_sweeps`` sweeps at the measured dwell.

        If a chain run at the returned $\zeta$ later stops returning to the vacuum
        (the :meth:`step` RuntimeError), that is *data* --- the documented
        defect-condensation signature --- and the honest response is to record it and
        start a fresh chain at a smaller $\zeta$, never to silently retry.

        Parameters
        ----------
        S: a NoIntersections action
        phi, n: arrays, optional
            A thermalized **valid** starting configuration (cold if omitted).
        min_vacuum_ticks: int
            Vacuum ticks a probe must collect for the rung to count as measurable.
        probe_sweeps: int
            Equilibration sweeps at each rung, and the tallied chunk size.
        max_probe_sweeps: int
            Tallied-sweep budget per rung; exhausting it rejects the rung.
        step_sweeps: int
            Target sweeps per :meth:`step` at the chosen $\zeta$.
        floor: float, optional
            An explicit minimum dwell imposed on top of measurability.

        Returns
        -------
        (float, int)
            The chosen $\zeta$ and the matched ``emit_every``.
        """
        L = S.Lattice
        if phi is None:
            phi = np.zeros((1,) + tuple(L.dims))
        if n is None:
            n = np.zeros((L.D,) + L.dims, dtype=np.int64)
        n_links = 4 * L.N**4
        best = None
        for z in ladder:
            probe = cls(S, zeta=z, D_max=D_max,
                        rng=rng if rng is not None else np.random.default_rng())
            # Equilibrate at THIS rung before believing anything: from a valid start
            # the condensate takes time to build, and a probe that tallies the
            # transient accepts rungs that later never come home.
            probe.run(phi, n, 2 * probe_sweeps, tally=False)
            probe.blocks = []
            probe._new_block()
            # Tally at least 4 chunks (so stationarity has something to compare) and
            # keep going until the vacuum-tick quota is met or the budget dies.
            chunks = []
            tallied = 0
            while tallied < max_probe_sweeps and (len(chunks) < 4
                                                  or sum(chunks) < min_vacuum_ticks):
                probe.run(phi, n, probe_sweeps, tally=True)
                probe.close_block()
                chunks.append(probe.blocks[-1][1])
                tallied += probe_sweeps
            H_Z = sum(chunks)
            dwell = H_Z / (tallied * n_links)
            half = len(chunks) // 2
            stationary = sum(chunks[half:]) >= sum(chunks[:half]) / 4
            if (H_Z >= min_vacuum_ticks and stationary
                    and (floor is None or dwell > floor)):
                best = (z, dwell)
            else:
                # Past the edge (unmeasurable, collapsing, or below the explicit
                # floor); dwell falls monotonically with zeta, so stop probing.
                break
        if best is None:
            raise RuntimeError(
                f'tune_edge: even the smallest ladder rung zeta={ladder[0]} never '
                f'demonstrated {min_vacuum_ticks} vacuum ticks in '
                f'{max_probe_sweeps} sweeps; the chain cannot emit here (defect '
                f'condensation?).  Treat as signal and investigate D_trace.')
        zeta, dwell = best
        emit_every = max(1, int(round(dwell * n_links * step_sweeps)))
        return zeta, emit_every

    def _new_block(self):
        self._H_pair = np.zeros(self.L.dims)
        self._H_Z = 0
        self._H_four = np.zeros(4)

    def close_block(self):
        r"""End the current jackknife block and start a new one (:meth:`run` path)."""
        self.blocks.append((self._H_pair, self._H_Z, self._H_four))
        self._new_block()

    def run(self, phi, n, sweeps, tally=True, progress=None, compiled=True):
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
        compiled: bool
            Use the compiled tick kernel (default); ``False`` runs the pure-python
            :meth:`_tick` loop, which the kernel reproduces bit-for-bit.

        Returns
        -------
        (phi, n)
            The evolved configuration, suitable for chaining calls.
        """
        st = self._init_state(phi, n, update_phi=True)
        iterator = range(sweeps)
        if progress is not None:
            iterator = progress(iterator)
        if compiled:
            V = self.N**4
            for _ in iterator:
                # Sweep boundary, in the same order the pure _tick performs it.
                if st.i == st.n_links:
                    self.D_trace.append(st.D)
                    self._phi_sweep(st)
                    self._draw_batch(st)
                H_pair = np.zeros(V, dtype=np.int64)
                H_four = np.zeros(4, dtype=np.int64)
                vac = self._kernel_ticks(st, tally, -1, H_pair, H_four)
                if tally:
                    self._H_Z += vac
                    self._H_pair += H_pair.reshape(tuple(self.L.dims))
                    self._H_four += H_four
        else:
            for _ in iterator:
                for _ in range(st.n_links):
                    vac, disp, cls4 = self._tick(st)
                    if tally:
                        if vac:
                            self._H_Z += 1
                        elif disp is not None:
                            self._H_pair[disp] += 1
                        elif cls4 is not None:
                            self._H_four[cls4] += 1
        self._phi_sweep(st)
        return st.phi, st.n

    def correlator(self):
        r"""
        The block-jackknife mean and error of the absolutely-normalized correlator
        $\Theta_{\Delta x} = H_{\text{pair}}(\Delta x) / (V \zeta^{2} H_{Z})$ over the
        blocks accumulated by :meth:`run` (skipping leave-one-out terms whose vacuum
        dwell vanishes --- only possible when $\zeta$ is badly tuned).

        Returns
        -------
        (Theta, dTheta)
            Arrays of spatial shape ``L.dims``; recall $\Theta_{0} = 1$ by definition
            and the $\Delta x = 0$ bin of $H_{\text{pair}}$ is empty by construction.
        """
        # The estimator: the sector-dwell ratio ⟨Π_p [q_p = δ_{px} - δ_{py}]⟩ over
        # ⟨Π_p [q_p = 0]⟩ equals ζ² V Θ_Δx in the enlarged ensemble -- the pair sector
        # carries the known fugacity price ζ² (divided back out here, which is why the
        # answer cannot depend on ζ) and V translated copies contribute to each
        # displacement bin.  Θ_0 = 1 identically: a coincident pair IS the vacuum, so
        # no origin normalization is needed -- Θ comes out ABSOLUTE.
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

    def binder(self):
        r"""
        The block-jackknife mean and error of the Binder ratio
        $U = \left\langle\left|M\right|^{4}\right\rangle /
        \left\langle\left|M\right|^{2}\right\rangle^{2}$ for the $\theta$-shift
        order parameter $M = \sum_{x} e^{i\theta_{x}}$, over the blocks accumulated by
        :meth:`run`.  See :class:`~supervillain.observable.ThetaBinderCumulant` for the
        sector decomposition; $U \to 2$ (complex Gaussian) deep in the symmetric phase
        and $U \to 1$ in a broken phase.
        """
        V = self.N**4
        Hp = np.stack([b[0] for b in self.blocks])
        HZ = np.array([b[1] for b in self.blocks], dtype=float)
        H4 = np.stack([b[2] for b in self.blocks])
        C = np.array([4., 2., 2., 1.])

        def U(hp, hz, h4):
            # <|M|^2> = V (1 + S1); <|M|^4> = (2V^2 - V) + 4(V-1) V S1 + sector term.
            S1 = hp.sum() / (V * self.zeta**2 * hz)
            M2 = V * (1 + S1)
            M4 = (2 * V**2 - V) + 4 * (V - 1) * V * S1 \
                + (C * h4).sum() / (self.zeta**4 * hz)
            return M4 / M2**2

        if HZ.sum() == 0:
            return np.nan, np.nan
        total = U(Hp.sum(axis=0), HZ.sum(), H4.sum(axis=0))
        rows = [j for j in range(len(self.blocks)) if HZ.sum() - HZ[j] > 0]
        jack = np.array([U(Hp.sum(axis=0) - Hp[j], HZ.sum() - HZ[j],
                           H4.sum(axis=0) - H4[j]) for j in rows])
        err = np.sqrt((len(rows) - 1) * jack.var()) if len(rows) > 1 else np.nan
        return total, err

    def report(self):
        r"""A short summary: acceptance, the pairs-in-flight trace, and (run-path) sector dwell."""
        # D = Σ|q| is always EVEN (|q| ≡ q mod 2 per site and the total charge Q = Σq
        # vanishes identically), so D/2 -- the number of ±pairs in flight, i.e. the
        # number of worms the grand-canonical ensemble is running at once -- is the
        # natural human-facing count.  The MEASURE stays in per-endpoint (per-insertion)
        # convention: ζ per unit of |q|.
        Dt = np.array(self.D_trace)
        H_Z = sum(b[1] for b in self.blocks)
        H_pair = sum(b[0].sum() for b in self.blocks)
        lines = [f'proposals {self.proposed}  acceptance {self.accepted/max(1,self.proposed):.4f}',
                 (f'pairs in flight D/2: mean {Dt.mean()/2:.2f}  max {int(Dt.max())//2}'
                  if len(Dt) else 'no sweeps'),
                 f'run-path sector dwell: vacuum {H_Z}  single-pair {int(H_pair)}  '
                 f'other {max(0, self.proposed - int(H_Z) - int(H_pair))}']
        return '\n'.join(lines)
