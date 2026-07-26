#!/usr/bin/env python

import logging
import math
from types import SimpleNamespace

import numpy as np

from supervillain.generator import Generator
from supervillain.generator.combining import Sequentially
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import Form, d
from supervillain.generator.villain.site_fourier_heatbath import FourierSiteHeatbath
from supervillain.generator.villain.exact import ExactUpdate
from supervillain.generator.villain.cohomology import CohomologyUpdate
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge
from supervillain.generator.no_intersection import defect_gas_kernel
import supervillain.action

logger = logging.getLogger(__name__)


def pair_shells(N):
    r"""
    The distinct min-image separation-squared shells a defect pair can occupy
    on the $N^4$ torus: returns ``(lookup, values)`` with ``values`` the sorted
    realized $r^2 > 0$ and ``lookup[r2]`` the shell index (-1 if unrealized).
    """
    d = np.minimum(np.arange(N), N - np.arange(N))**2
    rsq = (d[:, None, None, None] + d[None, :, None, None]
           + d[None, None, :, None] + d[None, None, None, :]).ravel()
    values = np.unique(rsq[rsq > 0])
    lookup = np.full(N**2 + 1, -1, dtype=np.int64)
    lookup[values] = np.arange(len(values))
    return lookup, values


def shell_multiplicity(N):
    r"""
    The per-shell bin multiplicity for :func:`pair_shells`\ 's realized shells on
    the $N^4$ torus: how many of the $N^4$ displacement bins with $r^2 > 0$ land
    in each shell.  Shared by :meth:`DefectGasWeightTuner._probe_umbrella` (the
    per-bin dwell) and :meth:`DefectGasWeightTuner.tune_umbrella` (the spec's
    conserved sector-total quantity).
    """
    lookup, values = pair_shells(N)
    d = np.minimum(np.arange(N), N - np.arange(N))**2
    rsq = (d[:, None, None, None] + d[None, :, None, None]
           + d[None, None, :, None] + d[None, None, None, :]).ravel()
    nz = rsq > 0
    mult = np.bincount(lookup[rsq[nz]], minlength=len(values))
    return lookup, values, rsq, nz, mult


def geo_weight_dict(defects, w2, lookup, N):
    r"""Python mirror of the kernel's ``_geo_weight`` on the sparse dict."""
    if w2.size == 0:
        return 1.0
    items = list(defects.items())
    if len(items) == 2:
        (c1, v1), (c2, v2) = items
        if v1 * v2 == -1:
            rsq = sum(min(d % N, N - d % N)**2 for d in
                      (c1[k] - c2[k] for k in range(4)))
            return w2[lookup[rsq]]
        return 1.0
    if len(items) == 4:
        pos = [c for c, v in items if v == 1]
        neg = [c for c, v in items if v == -1]
        if len(pos) != 2 or len(neg) != 2:
            return 1.0

        def _w(a, b):
            rsq = sum(min(d % N, N - d % N)**2 for d in
                      (a[k] - b[k] for k in range(4)))
            return w2[lookup[rsq]]

        return 0.5 * (_w(pos[0], neg[0]) * _w(pos[1], neg[1])
                     + _w(pos[0], neg[1]) * _w(pos[1], neg[0]))
    return 1.0


def cell_edge(cell, e, N):
    r"""
    Edge ``e`` $\in [0, 32)$ of the hypercube at raveled corner ``cell``:
    direction $\mu = \lfloor e / 8 \rfloor$; the bits of $e \bmod 8$ offset the three
    non-$\mu$ coordinates by $+1$.  Returns ``(mu, site)``.  (Pure-python
    mirror of the kernel's ``_cell_edge`` for the reference tick path.)
    """
    c = [0, 0, 0, 0]
    r = cell
    for k in (3, 2, 1, 0):
        c[k] = r % N
        r //= N
    mu = e // 8
    b = e % 8
    j = 0
    for nu in range(4):
        if nu == mu:
            continue
        if (b >> j) & 1:
            c[nu] = (c[nu] + 1) % N
        j += 1
    return mu, tuple(c)


def link_charge_sum(mu, site, q, N, dq=None):
    r"""
    $s(\ell, q) = \sum |q_c|$ over the 8 hypercubes containing link
    ``(mu, site)`` --- the corners ``site - b`` over the three non-$\mu$
    directions --- against the dense raveled charge array ``q``, optionally
    with the reference path's $\Delta q$ dict (4-tuple keys) applied.
    (Pure-python mirror of the kernel's ``_link_charge_sum`` /
    ``_link_charge_sum_delta``.)
    """
    s = 0
    for b in range(8):
        c = list(site)
        j = 0
        for nu in range(4):
            if nu == mu:
                continue
            if (b >> j) & 1:
                c[nu] = (c[nu] - 1) % N
            j += 1
        qq = int(q[((c[0] * N + c[1]) * N + c[2]) * N + c[3]])
        if dq is not None:
            qq += dq.get(tuple(c), 0)
        s += abs(qq)
    return s


class DefectGas(ReadWriteable, Generator):
    r"""
    Grand-canonical defect sampler for the $q = dn \wedge dn = 0$ constraint in 4D,
    with the inline estimator of the Lagrange-multiplier correlator
    $\left\langle e^{+i\theta_{x}} e^{-i\theta_{y}} \right\rangle$.

    The update starts from a constraint-satisfying defect-free configuration and
    makes updates by sampling in the enlarged ensemble of configurations with any arbitrary number of defects, with each defect weighted by the fugacity $\zeta$,

    .. math ::
        \Pi = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S[\phi, n]}\, \zeta^{D(n)},
        \qquad D(n) = \sum_{h} \left|q_{h}(n)\right|,
        \quad q = dn \wedge dn,

    The number of defects $D$ is always even since $Q=\sum_h q_h = \sum_h dj = 0$
    on every configuration, so $D/2$ counts the $\pm$ pairs in flight --- the number of
    worms the grand-canonical ensemble runs at once.
    Though, to be perfectly clear: there are no dedicated 'worms' which need to be advanced using any special worm moves.
    We just update links at random and they do what they want: idling, building up flux, creating and annihilating defects, and transporting charge.

    Standard single-link updates $n\to n \pm 1$ are Metropolis tested with the 
    fugacity included in the weight and are accepted with probability $\min\!\left(1, e^{-\Delta S}\,
    \zeta^{\Delta D}\right)$.  Every link can always receive a proposal but a proposal with
    a messy $\Delta q$ is exponentially discounted by the fugacity $\zeta < 1$, and defect-annihilating moves
    are correspondingly incentivized.  The higher-defect sectors are the corridors through
    jammed backgrounds that clean worms lack, so the sampler cannot jam; single-link
    moves connect every $n$ with nonzero acceptance, making ergodicity on the enlarged
    space manifest.

    The :meth:`step` advances the enlarged chain until its
    ``emit_every``-th visit to the vacuum sector and emits that configuration.
    Restricted to the vacuum sector the enlarged weight is $e^{-S_{V}} \zeta^{0} = e^{-S_{V}}$,
    so the emitted ensemble is exactly the constrained theory.  Like the worms,
    invalid states live only *inside* a step, and every emitted configuration satisfies
    $q \equiv 0$ and therefore this generator can be combined with other constrained generators.

    In a spirit similar to the worms, we can read off defect correlation functions by
    tallying the state of the chain after every proposal.
    Each positive defect amounts to an insertion of $e^{+i\theta}$ and each negative defect amounts to an insertion of $e^{-i\theta}$; for more details see :meth:`~supervillain.generator.no_intersection.DefectGas.inline_observables`.

    .. danger ::

        However, the fugacity $0 < \zeta < 1$ must be handled with care.
        Too large and the defects will proliferate and never return to the vacuum sector.
        Too small and the defects will become exceedingly rare and the samples will not explore the full grand-canonical ensemble.

        In other words, the limit $\zeta\to 1$ is the unconstrained theory while $\zeta\to 0$ restricts to defect-free configurations.
        
    So, we need to think about how to tune the fugacity $\zeta$.
    Entropy pushes $D$ upward (each defect may live anywhere, and the denser
    the sheet the larger a single link's $\left|\Delta D\right|$), so the right $\zeta$
    shrinks with volume and with $1/\kappa$.  Symptoms of a bad choice are loud: too
    large and $D$ pins at ``max_defects`` with the vacuum never revisited (:meth:`step` then
    raises rather than hang), too small and pair excursions become needlessly rare.
    The :class:`DefectGasFugacityTuner` automates the choice.

    .. warning ::

        Restricted to $D = 4$.  As a :class:`~supervillain.generator.Generator` this
        updates $n$ only, so it is not ergodic on its own; at least combine it with a
        $\phi$-update such as :class:`~supervillain.generator.villain.SiteUpdate`.

    Parameters
    ----------
    S: a NoIntersections action
        Supplies $S_{V}$, $\kappa$, and the lattice.
    fugacity: float, optional
        The per-defect fugacity $\zeta \in (0, 1]$.  Exactly one of ``fugacity``
        and ``sectorWeights`` is required.
    max_defects: int or None
        Hard cap on $D$; ``None`` uncaps (geometric path only --- ``sectorWeights`` pins it).
    emit_every: int, optional
        Vacuum ticks per :meth:`step` emission; defaults to $4 N^{4}$ (about one
        sweep's worth of vacuum time).
    max_step_sweeps: int
        Safety horizon: :meth:`step` raises after this many sweeps without reaching
        ``emit_every`` vacuum ticks (the defect-condensation symptom).
    rng: numpy Generator, optional
    sectorWeights: sequence of floats, optional
        The sector table $w_k$ for $k = D/2 = 0, \ldots, K$, an alternative to
        the geometric ``fugacity`` ($w_k = \zeta^{2k}$).  Normalized so $w_0 = 1$;
        the length pins ``max_defects`` $= 2K$.  Chosen well (see
        :class:`DefectGasWeightTuner`) the sector occupancies flatten, so the chain
        shuttles freely between the vacuum and the multi-pair sectors instead of
        paying $e^{-\langle D/2 \rangle}$ for them.
    pairSeparationUmbrella: numpy array, optional
        The umbrella table $w_2$: one positive entry per distinct minimal-image
        $r^{2}$ shell (see :func:`pair_shells`), multiplying the enlarged-ensemble
        weight by $w_{2}(\left|x - y\right|)$ in the single-pair sector and by the
        averaged Wick product $\frac{1}{2}[w_{2}(\left|x_{1}-y_{1}\right|)
        w_{2}(\left|x_{2}-y_{2}\right|) + w_{2}(\left|x_{1}-y_{2}\right|)
        w_{2}(\left|x_{2}-y_{1}\right|)]$ in the unit-charge $D = 4$ class.
        ``Theta_Theta`` is divided binwise by $w_{2}$ at emission and the
        four-defect tally accumulates $1/W$ per tick, so every published estimator
        is $w_{2}$-independent.  Omitted means $W = 1$ identically (bit-for-bit the
        unumbrella'd sampler).  Composes with either pricing; learned by
        :meth:`DefectGasWeightTuner.tune_umbrella`.
    uniformProposalFraction: float or sequence of floats, optional
        The mixture table $\gamma$ of the defect-adjacent proposal targeting:
        in sector $k = D/2$, a proposal is
        drawn uniformly with probability $\gamma_k$ and otherwise targeted at a
        live defect (a cell with probability $\propto \left|q_c\right|$, then one
        of its 32 edges uniformly), with the full Metropolis--Hastings ratio in
        the accept test so every estimator is exactly unchanged.  A scalar
        broadcasts to all $K + 1$ sectors; entries lie in $(0, 1]$; requires a
        capped gas.  Omitted means the legacy uniform proposal, bit-for-bit.
        Net creation/annihilation fluxes are Hastings-invariant by construction;
        the win is in-sector transport (the pair's $r$-space diffusion rate),
        which is what starves far $\Theta$ bins and umbrella round trips at
        large volume.

    """

    # D == 4 sector classes for the fourth moment, keyed by the sorted charge values:
    # index 0: {+1,+1,-1,-1}, 1: {+2,-1,-1}, 2: {+1,+1,-2}, 3: {+2,-2}.
    _FOUR = {(-1, -1, 1, 1): 0, (-1, -1, 2): 1, (-2, 1, 1): 2, (-2, 2): 3}

    def __init__(self, S, fugacity=None, max_defects=None, emit_every=None, max_step_sweeps=500,
                 rng=None, sectorWeights=None, pairSeparationUmbrella=None,
                 uniformProposalFraction=None):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('DefectGas requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('DefectGas is only implemented for D = 4.')
        if (fugacity is None) == (sectorWeights is None):
            raise ValueError('exactly one of fugacity and sectorWeights is required.')
        if sectorWeights is not None:
            w = np.asarray(sectorWeights, dtype=np.float64)
            if w.ndim != 1 or len(w) < 2:
                raise ValueError(f'sectorWeights must be a 1D sequence of at least two sectors; got shape {w.shape}.')
            if not np.all(w > 0):
                raise ValueError('sectorWeights must be positive.')
            w = w / w[0]
            if max_defects is None:
                max_defects = 2 * (len(w) - 1)
            elif max_defects != 2 * (len(w) - 1):
                raise ValueError(f'sectorWeights of length {len(w)} pin max_defects = {2 * (len(w) - 1)}; got max_defects={max_defects}.')
        else:
            if not (0 < fugacity <= 1):
                raise ValueError(f'fugacity must be in (0, 1]; got {fugacity}.')
            w = np.zeros(0, dtype=np.float64)   # empty table: the kernel's geometric path

        self.S = S
        self.L = S.Lattice
        self.N = self.L.N
        self.kappa = S.kappa
        self.fugacity = None if fugacity is None else float(fugacity)
        self.sectorWeights = w
        # The tally prices unify the two paths: the D = 2 histogram is divided by _w1
        # and the D = 4 classes by _w4 at emission.
        if self.fugacity is not None:
            self._w1, self._w4 = self.fugacity**2, self.fugacity**4
        else:
            self._w1 = float(w[1])
            self._w4 = float(w[2]) if len(w) > 2 else 1.0
        # The pair-separation umbrella: a per-shell table multiplying the
        # enlarged weight in the D = 2 and D = 4 unit-charge sectors (an empty
        # table means W = 1 identically and every prior path is bit-for-bit).
        if pairSeparationUmbrella is None:
            self.pairSeparationUmbrella = np.zeros(0, dtype=np.float64)
            self._rsq_shell = np.zeros(0, dtype=np.int64)
            self._w2_field = None
        else:
            lookup, values = pair_shells(self.N)
            w2 = np.asarray(pairSeparationUmbrella, dtype=np.float64)
            if w2.shape != values.shape:
                raise ValueError(f'pairSeparationUmbrella must have one entry per realized shell '
                                 f'({len(values)} for N={self.N}); got {w2.shape}.')
            if not np.all(w2 > 0):
                raise ValueError('pairSeparationUmbrella must be positive.')
            self.pairSeparationUmbrella = w2
            self._rsq_shell = lookup
            d = np.minimum(np.arange(self.N), self.N - np.arange(self.N))**2
            rsq = (d[:, None, None, None] + d[None, :, None, None]
                   + d[None, None, :, None] + d[None, None, None, :])
            field = np.ones(tuple(self.L.dims))
            nz = rsq > 0
            field[nz] = w2[lookup[rsq[nz]]]
            self._w2_field = field
        self.max_defects = max_defects
        if uniformProposalFraction is None:
            self.uniformProposalFraction = np.zeros(0, dtype=np.float64)
        else:
            if self.max_defects is None:
                raise ValueError('uniformProposalFraction requires a capped gas (sectorWeights or max_defects).')
            K1 = self.max_defects // 2 + 1
            g = np.asarray(uniformProposalFraction, dtype=np.float64)
            if g.ndim == 0:
                g = np.full(K1, float(g))
            if g.shape != (K1,):
                raise ValueError(f'uniformProposalFraction must be a scalar or a vector of length '
                                 f'{K1} (one entry per sector); got shape {g.shape}.')
            if not np.all((g > 0) & (g <= 1)):
                raise ValueError('uniformProposalFraction entries must be in (0, 1].')
            self.uniformProposalFraction = g
        self.emit_every = emit_every if emit_every is not None else 4 * self.N**4
        self.max_step_sweeps = max_step_sweeps
        self.rng = rng if rng is not None else np.random.default_rng()
        # The mixture/target/edge draws ride on an INDEPENDENT stream, spawned from
        # the seed sequence (not the consumable output) of self.rng, so drawing them
        # never perturbs the legacy mus/sites/cs/us sequence -- a gamma=1 chain (the
        # mixture never fires, Hastings factor exactly 1) then tracks a gamma-less
        # chain sweep after sweep, not just through the first batch.
        self._aux_rng = (np.random.default_rng(self.rng.bit_generator.seed_seq.spawn(1)[0])
                          if self.uniformProposalFraction.size > 0 else None)
        self.D_trace = []           # per-sweep defect count (diagnostic)
        self.accepted = 0
        self.proposed = 0
        # The persistent chain state used by step(); rebuilt whenever the incoming
        # configuration does not match (first call, or another generator moved it).
        self._state = None

    def __str__(self):
        tag = ('' if self.uniformProposalFraction.size == 0 else
       f', uniformProposalFraction={np.array2string(self.uniformProposalFraction, precision=3)}')
        if self.fugacity is not None:
            return f'DefectGas(fugacity={self.fugacity}, max_defects={self.max_defects}{tag})'
        return f'DefectGas(sectorWeights={np.array2string(self.sectorWeights, precision=4)}, max_defects={self.max_defects}{tag})'

    # ---------------------------------------------------------------- chain internals

    def _init_state(self, phi, n):
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
        # completed excursions, max single-pair min-image separation squared,
        # touched-top-sector flag, round trips] and the power-of-two excursion-length
        # histogram.  These turn empty far bins into honest transport-censoring
        # statements instead of silent zeros.
        st.tstate = np.zeros(5, dtype=np.int64)
        st.exc_hist = np.zeros(32, dtype=np.int64)
        st.dphi = np.ascontiguousarray(d(Form(st.phi, degree=0, lattice=L)))
        st.dphi2 = st.dphi.reshape(4, -1)
        st.n_links = 4 * self.N**4
        self._draw_batch(st)
        return st

    def _draw_batch(self, st):
        # Draw a sweep's worth of proposals up front (numpy batching).  The legacy
        # proposal -- uniform link, uniform c = ±1 -- is symmetric and needs no
        # Hastings factor; with a gamma table the mixture component, target cell,
        # and edge draws ride along and _tick applies the full MH ratio.  Legacy
        # arrays are drawn first, in the historical order, from self.rng, so the
        # gamma-less stream is bit-for-bit unchanged; the extra arrays are drawn
        # from the independent _aux_rng (see __init__) so a gamma chain's legacy
        # prefix never shifts relative to a gamma-less chain sweep after sweep.
        rng, N = self.rng, self.N
        st.mus = rng.integers(0, 4, size=st.n_links)
        st.sites = rng.integers(0, N, size=(st.n_links, 4))
        st.cs = rng.choice((-1, 1), size=st.n_links)
        st.us = rng.uniform(0, 1, size=st.n_links)
        if self.uniformProposalFraction.size > 0:
            aux = self._aux_rng
            st.comps = aux.uniform(0, 1, size=st.n_links)
            st.ucells = aux.uniform(0, 1, size=st.n_links)
            st.uedges = aux.uniform(0, 1, size=st.n_links)
        else:
            st.comps = np.zeros(0)
            st.ucells = np.zeros(0)
            st.uedges = np.zeros(0)
        st.i = 0

    def _tick(self, st):
        r"""One clock tick of the enlarged chain: a single-link Metropolis proposal
        ($\phi$ frozen for the whole step).  Returns
        ``(vacuum, pair_displacement)`` classifying the sector the chain sits in at
        this tick --- the raw material of every estimate this class makes."""
        if st.i == st.n_links:
            # Sweep boundary.  φ is FROZEN for the whole step -- the chain then
            # preserves the conditional π_ζ(n | φ), whose vacuum trace preserves
            # π(n | φ, q ≡ 0), so the step composes Gibbs-style with a φ-update
            # (SiteUpdate) in a Sequentially, exactly like the other n-only updates.
            self.D_trace.append(st.D)
            self._draw_batch(st)
        i = st.i
        st.i += 1
        mu = int(st.mus[i])
        site = (int(st.sites[i, 0]), int(st.sites[i, 1]),
                int(st.sites[i, 2]), int(st.sites[i, 3]))
        c = int(st.cs[i])
        targeted = self.uniformProposalFraction.size > 0
        if targeted and st.D > 0 and st.comps[i] >= self.uniformProposalFraction[st.D // 2]:
            # Adjacent draw: a defect cell with probability |q_c|/D, then one of
            # its 32 edges uniformly.  (In vacuum there is nothing to target and
            # the density below degenerates to pure uniform.)
            target = st.ucells[i] * st.D
            cum = 0.0
            cell = int(st.nzc[0])
            for b in range(st.nnz):
                cum += abs(int(st.q[st.nzc[b]]))
                if target < cum:
                    cell = int(st.nzc[b])
                    break
            mu, site = cell_edge(cell, int(st.uedges[i] * 32), self.N)
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
        if self.max_defects is None or st.D + dD <= self.max_defects:
            # Metropolis on the ENLARGED weight e^{-S_V} ζ^D: the Villain ΔS is local
            # to this one link, and the constraint enters only through ζ^ΔD --
            # defect-annihilating moves (ΔD < 0) are REWARDED, which is what lets the
            # mess clean itself up.  (The max_defects cap above is just a truncated state
            # space: proposals past it are ordinary zero-weight rejections.)
            link = (mu,) + site
            A = st.dphi[link] - 2 * np.pi * st.n[link]
            dS = (self.kappa / 2) * ((A - 2 * np.pi * c)**2 - A**2)
            if self.sectorWeights.size > 0:
                ratio = self.sectorWeights[(st.D + dD) // 2] / self.sectorWeights[st.D // 2]
            else:
                ratio = self.fugacity**dD
            if self.pairSeparationUmbrella.size > 0:
                newD = st.D + dD
                if st.D in (2, 4) or newD in (2, 4):
                    trial = dict(st.defects)
                    for cell, dv in dq.items():
                        q1 = trial.get(cell, 0) + dv
                        if q1:
                            trial[cell] = q1
                        else:
                            trial.pop(cell, None)
                    Wcur = geo_weight_dict(st.defects, self.pairSeparationUmbrella,
                                           self._rsq_shell, self.N)
                    Wnew = geo_weight_dict(trial, self.pairSeparationUmbrella,
                                           self._rsq_shell, self.N)
                    ratio = ratio * (Wnew / Wcur)
            if targeted:
                nl = np.float64(st.n_links)
                pu = 1.0 / nl
                if st.D > 0:
                    g = self.uniformProposalFraction[st.D // 2]
                    s_fwd = link_charge_sum(mu, site, st.q, self.N)
                    p_fwd = g * pu + (1.0 - g) * s_fwd / (32.0 * st.D)
                else:
                    p_fwd = pu
                newD = st.D + dD
                if newD > 0:
                    g2 = self.uniformProposalFraction[newD // 2]
                    s_rev = link_charge_sum(mu, site, st.q, self.N, dq=dq)
                    p_rev = g2 * pu + (1.0 - g2) * s_rev / (32.0 * newD)
                else:
                    p_rev = pu
                ratio = ratio * (p_rev / p_fwd)
            if st.us[i] < np.exp(-dS) * ratio:
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
            if st.tstate[3] == 1:
                # A vacuum return after touching the top sector: one round trip.
                st.tstate[4] += 1
                st.tstate[3] = 0
            return True, None, None
        st.tstate[0] += 1
        if self.max_defects is not None and st.D == self.max_defects:
            st.tstate[3] = 1
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
        r"""
        We tally ``VacuumTicks``, how often the chain visits the vacuum sector,

        .. math ::

            \texttt{VacuumTicks} = \sum_{\text{ticks}} \prod [q = 0],

        and ``Theta_Theta``, the pair-sector dwell histogram (translation averaged,
        scaled by its fugacity price $\zeta^2$),

        .. math ::

            \texttt{Theta\_Theta}_{\Delta h} = \frac{1}{V \zeta^2} \sum_{h}
                \prod [q = \delta_{h+\Delta h} - \delta_{h}]

        whose ratio give the :class:`~.Intersection_Intersection` correlator $\Theta$.

        ``Ticks`` counts the total number of proposals the step consumed, so the step's vacuum
        dwell is ``VacuumTicks / Ticks``.

        The ``FourDefectDistribution`` (see :class:`~.FourDefects` and its derivation) is the :class:`~supervillain.generator.no_intersection.DefectGas`'s per-step dwell in
        the four $D = 4$ sector classes, scaled by the known fugacity price $1/\zeta^{4}$.
        Index 0 counts all the charges being on four distinct hypercubes,
        index 1 counts there being a $+2$ charge and two distinct $-1$ charges,
        index 2 counts there being two distinct $+1$ charges and a $-2$ charge, and
        index 3 counts there being both a $+2$ and a $-2$ charge.

        Physical observables probably should consume the
        multiplicity-weighted combination :class:`~.FourDefects`.

        .. note ::

            The $\Delta x = 0$ bin of the ``Theta_Theta`` histogram is exactly zero in every
            configuration --- not because $\Theta_0$ vanishes, but because the chain
            *cannot dwell* there: a coincident $\pm$ pair has $q \equiv 0$, so that
            "sector" is the vacuum itself and its ticks are tallied by
            :class:`~.VacuumTicks` instead.  Physically $\Theta_{0} = 1$ identically
            (coincident insertions are the identity), which is exactly what makes the
            estimator absolutely normalized.  :class:`~.Intersection_Intersection`
            writes that origin value outright, so read $\Theta$ (or sum it, as
            :class:`~.IntersectionSusceptibility` does) from there rather than patching
            the raw histogram.

        Per step these ride along with ``PairExcursions``,
        ``MaxPairSeparationSquared``, and ``ExcursionLengths``.

        Returns initialized :class:`~supervillain.batch.Batch` storage for each; counters are integers but ``Theta_Theta`` and ``FourDefectDistribution`` are floats because they are scaled by powers of the fugacity.
        """
        obs = {
            'Theta_Theta': Batch(steps, shape=self.L.dims),
            'VacuumTicks': Batch(steps, shape=(), dtype=np.int64),
            'Ticks': Batch(steps, shape=(), dtype=np.int64),
            'FourDefectDistribution': Batch(steps, shape=(4,), dtype=float),
            'PairExcursions': Batch(steps, shape=(), dtype=np.int64),
            'MaxPairSeparationSquared': Batch(steps, shape=(), dtype=np.int64),
            'ExcursionLengths': Batch(steps, shape=(32,), dtype=np.int64),
        }
        if self.max_defects is not None:
            # Generator bookkeeping like Ticks: SectorTicks is the per-sector tick
            # histogram (the weight tuner's input and the flat-histogram health
            # check) and RoundTrips counts vacuum returns that touched the top
            # sector since the previous vacuum tick.  No Observable class for either.
            obs['SectorTicks'] = Batch(steps, shape=(self.max_defects // 2 + 1,), dtype=np.int64)
            obs['RoundTrips'] = Batch(steps, shape=(), dtype=np.int64)
        return obs

    def _kernel_ticks(self, st, tally, vac_stop, H_pair, H_four, t_sector):
        # One compiled pass over the remainder of the current proposal batch (stopping
        # early at vac_stop vacuum ticks when positive); mutates the dense chain state
        # in place and returns the number of vacuum ticks seen.
        i0 = st.i
        i, D, nnz, acc, vac = defect_gas_kernel.tick_batch(
            st.F2, st.n2, st.dphi2, st.q, st.nzc, st.D, st.nnz,
            st.mus, st.sites, st.cs, st.us, st.comps, st.ucells, st.uedges, i0,
            self.kappa, 0.0 if self.fugacity is None else self.fugacity, self.sectorWeights,
            self.pairSeparationUmbrella, self._rsq_shell, self.uniformProposalFraction,
            -1 if self.max_defects is None else int(self.max_defects), self.N,
            *defect_gas_kernel.stencil_pack(),
            H_pair, H_four, tally, vac_stop, st.tstate, st.exc_hist, t_sector)
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
            st = self._state = self._init_state(phi_in, n_in)
        H_pair = np.zeros(V, dtype=np.int64)
        H_four = np.zeros(4, dtype=np.float64)
        # Per-tick sector histogram and round trips, emitted whenever the chain is
        # capped (an empty t_sector switches the tally off in the tick loops).
        K1 = 0 if self.max_defects is None else self.max_defects // 2 + 1
        t_sector = np.zeros(K1, dtype=np.int64)
        rt0 = int(st.tstate[4])
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
                    f'{self} is likely too heavy for this volume/kappa (defect '
                    f'condensation).  Retune (DefectGasFugacityTuner or DefectGasWeightTuner), '
                    f'lighten the pricing, or note that a genuinely condensed theta phase '
                    f'requires per-pair weight ~ 1/V.')
            v, t = ticker(st, self.emit_every - vacuum, H_pair, H_four, t_sector)
            vacuum += v
            ticks += t
        # Emit AT a vacuum tick: st.n is exactly valid here.
        # φ was frozen for the whole step, so it passes through unchanged; only n is
        # re-emitted (a copy, so the chain's working array stays private).
        out = configuration | {
            'n': Form(st.n.copy(), degree=1, lattice=L),
            'Theta_Theta': H_pair.reshape(tuple(L.dims))
                           / (V * self._w1
                              * (self._w2_field if self._w2_field is not None else 1.0)),
            'VacuumTicks': int(vacuum),
            'Ticks': int(ticks),
            'FourDefectDistribution': H_four / self._w4,
            'PairExcursions': int(st.tstate[1] - exc0),
            'MaxPairSeparationSquared': int(st.tstate[2]),
            'ExcursionLengths': st.exc_hist - hist0,
        }
        if K1:
            out['SectorTicks'] = t_sector
            out['RoundTrips'] = int(st.tstate[4]) - rt0
        return out

    def step(self, configuration):
        r"""
        Advance the enlarged chain until its ``emit_every``-th vacuum tick and emit
        that configuration --- the trace of the chain on the constraint surface, so
        every emitted configuration satisfies $q \equiv 0$ exactly and the emitted
        ensemble is the constrained theory.  The pair-sector dwell accumulated along
        the way rides along as the inline ``Theta_Theta`` (already scaled by
        $1/V\zeta^{2}$) and ``VacuumTicks``.

        The tick loop runs in a compiled kernel
        (:func:`~supervillain.generator.no_intersection.defect_gas_kernel.tick_batch`)
        consuming the same pre-drawn proposals as the pure-python
        :meth:`step_reference`, which it reproduces bit-for-bit.
        """
        def ticker(st, vac_stop, H_pair, H_four, t_sector):
            if st.i == st.n_links:
                self.D_trace.append(st.D)
                self._draw_batch(st)
            i0 = st.i
            vac = self._kernel_ticks(st, True, vac_stop, H_pair, H_four, t_sector)
            return vac, st.i - i0
        out = self._step_body(configuration, ticker)
        self._sync_defects()
        return out

    def _sync_defects(self):
        # Mirror the dense charge state back into the sparse dict so step_reference
        # can pick up where the kernel left off.
        st, dims = self._state, tuple(self.L.dims)
        st.defects = {
            tuple(int(x) for x in np.unravel_index(int(cell), dims)): int(st.q[cell])
            for cell in st.nzc[:st.nnz]}

    def _probe_sweeps(self, configuration, sweeps, tally=False):
        r"""
        Advance the enlarged chain a fixed number of sweeps --- never waiting for the
        vacuum, never emitting --- and return ``(configuration, SectorTicks,
        RoundTrips, H_pair)`` for the block.

        Tuner plumbing, not Generator API: the returned ``n`` is the raw chain state,
        generally invalid (mid-excursion), and must not enter an
        :class:`~supervillain.Ensemble`.  Passing the returned configuration back in
        (possibly with a companion-updated ``phi``) continues the chain exactly; the
        tick budget is exact whenever the chain sits at a sweep boundary, which the
        tuner's usage guarantees.
        """
        if self.max_defects is None:
            raise ValueError('_probe_sweeps needs a capped chain (finite max_defects).')
        n_in = np.asarray(configuration['n']).astype(np.int64)
        phi_in = np.asarray(configuration['phi']).astype(float)
        st = self._state
        if st is None or not (np.array_equal(st.n, n_in)
                              and np.array_equal(st.phi, phi_in)):
            st = self._state = self._init_state(phi_in, n_in)
        t_sector = np.zeros(self.max_defects // 2 + 1, dtype=np.int64)
        H_pair = np.zeros(self.N**4, dtype=np.int64)
        H_four = np.zeros(4, dtype=np.float64)
        rt0 = int(st.tstate[4])
        remaining = sweeps * st.n_links
        while remaining > 0:
            if st.i == st.n_links:
                self.D_trace.append(st.D)
                self._draw_batch(st)
            i0 = st.i
            # vac_stop=0: run to the end of the proposal batch; tally parameter skips the
            # pair/four histograms (the sector histogram is always on).
            self._kernel_ticks(st, tally, 0, H_pair, H_four, t_sector)
            remaining -= st.i - i0
        self._sync_defects()
        return ({'phi': st.phi, 'n': st.n.copy()},
                t_sector, int(st.tstate[4]) - rt0, H_pair)

    def step_reference(self, configuration):
        r"""
        The plain, obviously-correct :meth:`step`: the same advance-until-emit loop
        driven tick-by-tick through the pure-python :meth:`_tick` (sparse defect dict,
        per-proposal stencil dicts).  Kept as the correctness oracle the compiled
        :meth:`step` is validated against --- same proposals, same accept test, same
        tallies, bit-for-bit.
        """
        def ticker(st, vac_stop, H_pair, H_four, t_sector):
            vac, disp, cls4 = self._tick(st)
            if t_sector.size > 0:
                t_sector[st.D // 2] += 1
            if vac:
                return 1, 1
            if disp is not None:
                N = self.N
                H_pair[((disp[0] * N + disp[1]) * N + disp[2]) * N + disp[3]] += 1
            elif cls4 is not None:
                H_four[cls4] += 1.0 / geo_weight_dict(st.defects, self.pairSeparationUmbrella,
                                                      self._rsq_shell, self.N)
            return 0, 1
        return self._step_body(configuration, ticker)

    def report(self):
        r"""A short summary: acceptance and the pairs-in-flight trace."""
        # D = Σ|q| is always EVEN (|q| ≡ q mod 2 per site and the total charge Q = Σq
        # vanishes identically), so D/2 -- the number of ±pairs in flight -- is the
        # natural human-facing count.
        Dt = np.array(self.D_trace)
        return '\n'.join([
            f'proposals {self.proposed}  acceptance {self.accepted/max(1,self.proposed):.4f}',
            (f'pairs in flight D/2: mean {Dt.mean()/2:.2f}  max {int(Dt.max())//2}'
             if len(Dt) else 'no sweeps')])


class DefectGasFugacityTuner:
    r"""
    Picks the :class:`DefectGas` fugacity $\zeta$ by short Monte-Carlo probes down (or
    up) a ladder, driving each probe through the ordinary
    :class:`~supervillain.Ensemble` route: every rung runs a throwaway
    ``Sequentially((*companions, DefectGas(S, fugacity)))`` chain and reads the vacuum
    dwell from the emitted inline quantities as ``VacuumTicks / Ticks``.  A rung whose
    chain stops returning to the vacuum (the :meth:`DefectGas.step` ``RuntimeError``)
    is rejected --- that is the defect-condensation signature, not an error.

    A tuner is not a generator: it has no ``step`` and never rides into an
    ensemble.  It runs experiments to decide *which* generator to build;
    :meth:`generator` returns the production-ready chain with fugacity $\zeta$ and
    ``emit_every`` matched by construction.

    Parameters
    ----------
    S: a NoIntersections action
    companions: iterable of generators, optional
        Interleaved with the probe gas (and in the
        :meth:`generator`).  Defaults to
        :class:`~supervillain.generator.villain.SiteUpdate` ($\phi$ must
        fluctuate or the Villain weights are sampled at frozen $d\phi$),
        :class:`~supervillain.generator.villain.ExactUpdate`, and
        :class:`~supervillain.generator.villain.CohomologyUpdate` (the two
        $D$-neutral $n$ moves at fixed $dn$).  For honest dwell,
        pass the companions production will run.
    max_defects: int or None
        Handed to every probe (and production) :class:`DefectGas`.
    rng: numpy Generator, optional
    """

    def __init__(self, S, companions=None, max_defects=8, rng=None):
        self.S = S
        self.max_defects = max_defects
        self.rng = rng if rng is not None else np.random.default_rng()
        if companions is not None:
            self.companions = tuple(companions)
        else:
            # FourierSiteHeatbath draws phi (exactly, and jointly -- it reads n only
            # through delta(dphi - 2 pi n), so it neither inspects nor disturbs it);
            # ExactUpdate stirs n at fixed dn (q exactly preserved); CohomologyUpdate
            # shifts the winding holonomy no local move reaches (and with it the
            # theta-current windings J_mu).  All three are D-neutral and tolerate the
            # mid-excursion (invalid-n) states the probes hand them.
            defaults = (FourierSiteHeatbath(S), ExactUpdate(S), CohomologyUpdate(S))
            for generator in defaults:
                generator.rng = self.rng
            self.companions = defaults

    def _probe(self, fugacity, start, steps, emit_every, max_step_sweeps):
        # One rung: a throwaway Generator-route chain.  Returns the per-step
        # (VacuumTicks, Ticks) arrays, or None if the chain condensed (the step
        # RuntimeError) -- an unhealthy rung, not an error.
        import supervillain.ensemble
        gas = DefectGas(self.S, fugacity, max_defects=self.max_defects, emit_every=emit_every,
                        max_step_sweeps=max_step_sweeps, rng=self.rng)
        chain = Sequentially((*self.companions, gas))
        try:
            e = supervillain.ensemble.Ensemble(self.S).generate(steps, chain,
                                                                start=start)
        except RuntimeError:
            return None
        return np.asarray(e.VacuumTicks), np.asarray(e.Ticks)

    def tune(self, start='cold', ladder=(0.1, 0.05, 0.02, 0.01, 0.005, 0.002),
             steps=120, target=0.15):
        r"""
        Descend the ladder and keep the first fugacity $\zeta$ whose vacuum dwell exceeds
        ``target`` (falling back to the smallest rung if none does).  Because the
        estimator is $\zeta$-independent, tuning affects only the variance, never the
        answer.  Entropy pushes the defect count up, so the right $\zeta$ shrinks with
        volume and with $1/\kappa$.

        Each rung probes ``steps`` configurations from ``start`` with a small
        ``emit_every`` (a healthy step then costs about a sweep); the first half is
        per-rung equilibration and the dwell is read off the second half.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where each rung's probe chain starts; anything
            :meth:`~supervillain.Ensemble.generate` accepts.  Probes from a
            thermalized valid configuration measure the equilibrium dwell; a fresh
            start's early dwell is optimistic.
        ladder: iterable of floats
            Candidate fugacities in descending order; the first rung whose dwell
            exceeds ``target`` wins.
        steps: int
            Configurations per probe; the first half is per-rung equilibration and
            the dwell is read off the second half.
        target: float
            The vacuum-dwell fraction a rung must exceed to be kept.

        Returns
        -------
        float
            The chosen $\zeta$.
        """
        V = self.S.Lattice.N ** 4
        fugacity = ladder[-1]
        condensed = True
        for z in ladder:
            fugacity = z
            probe = self._probe(z, start, steps,
                                emit_every=max(1, round(target * 4 * V)),
                                max_step_sweeps=25)
            if probe is None:
                condensed = True
                continue                        # condensed: descend
            vac, ticks = (a[steps // 2:] for a in probe)
            condensed = False
            if vac.sum() / max(1.0, ticks.sum()) > target:
                break
        if condensed:
            logger.warning(
                'tune: every ladder rung condensed; returning the smallest rung %g, '
                'which itself condensed -- production will likely raise.  Investigate '
                'D_trace / lower the ladder.', fugacity)
        return fugacity

    def tune_edge(self, start='cold',
                  ladder=(0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05,
                          0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3),
                  steps=200, min_vacuum_ticks=500, max_probe_sweeps=4000,
                  step_sweeps=25, floor=None):
        r"""
        Ride the edge: ascend the ladder and keep the *largest* fugacity $\zeta$ at which
        the chain still demonstrably returns to the vacuum --- pair transport scales
        like $\zeta^{2}$, so a conservatively small $\zeta$ silently censors exactly the
        large-separation bins that diagnose $\theta$ order.

        A rung is accepted iff its probe completes (each completed step *is*
        ``emit_every`` vacuum ticks, so completing the tallied half collects
        ``min_vacuum_ticks``) **and** the dwell is stationary across the tallied half
        (its second half at least a quarter of its first: a collapsing dwell is
        condensation in progress).  An explicit ``floor`` may be imposed on top.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where each rung's probe chain starts; anything
            :meth:`~supervillain.Ensemble.generate` accepts.
        ladder: iterable of floats
            Candidate fugacities in ascending order; the largest measurable,
            stationary rung wins and the first failing rung stops the ascent.
        steps: int
            Configurations per probe; the first half is per-rung equilibration and
            the tallied second half must collect ``min_vacuum_ticks``.
        min_vacuum_ticks: int
            Vacuum ticks the tallied half must collect for the rung to count as
            measurable; sets each probe step's ``emit_every``.
        max_probe_sweeps: int
            Sweep budget per rung, divided across its ``steps`` as each probe step's
            horizon; a rung that exhausts a step's share is rejected as condensed.
        step_sweeps: int
            Target sweeps per production step; sizes the returned ``emit_every``
            from the measured dwell.
        floor: float, optional
            An explicit minimum dwell imposed on top of measurability.

        Returns
        -------
        (float, int)
            The chosen $\zeta$ and a matched ``emit_every`` sized so one production
            step costs about ``step_sweeps`` sweeps at the measured dwell.
        """
        if not ladder:
            raise RuntimeError(
                'tune_edge: an empty ladder has no rung to probe (all candidate '
                'fugacities already condensed?).  Treat as signal and investigate D_trace.')
        n_links = 4 * self.S.Lattice.N ** 4
        best = None
        for z in ladder:
            probe = self._probe(
                z, start, steps,
                emit_every=max(1, -(-min_vacuum_ticks // max(1, steps // 2))),
                max_step_sweeps=max(1, max_probe_sweeps // steps))
            if probe is None:
                break                           # past the edge; dwell falls monotonically
            vac, ticks = (a[steps // 2:] for a in probe)
            half = len(vac) // 2
            dwell = vac.sum() / max(1.0, ticks.sum())
            d1 = vac[:half].sum() / max(1.0, ticks[:half].sum())
            d2 = vac[half:].sum() / max(1.0, ticks[half:].sum())
            if (vac.sum() >= min_vacuum_ticks and d2 >= d1 / 4
                    and (floor is None or dwell > floor)):
                best = (z, dwell)
            else:
                break
        if best is None:
            raise RuntimeError(
                f'tune_edge: even the smallest ladder rung fugacity={ladder[0]} never '
                f'completed its probe; the chain cannot emit here (defect '
                f'condensation?).  Treat as signal and investigate D_trace.')
        fugacity, dwell = best
        return fugacity, max(1, int(round(dwell * n_links * step_sweeps)))

    def generator(self, start='cold', edge=False, **kwargs):
        r"""
        The normal way to consume a tune: probe from ``start`` (with :meth:`tune`, or
        :meth:`tune_edge` when ``edge``; ``kwargs`` forward), then return the
        production-ready ``Sequentially((*companions, DefectGas(...)))`` with $\zeta$ and
        ``emit_every`` matched by construction.  The chain carries ``fugacity``
        and ``emit_every`` as plain metadata for introspection.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where the tuning probes start; anything
            :meth:`~supervillain.Ensemble.generate` accepts.
        edge: bool
            Tune with :meth:`tune_edge` (largest measurable $\zeta$, matched
            ``emit_every``) instead of :meth:`tune` (dwell target, default
            ``emit_every``).
        kwargs:
            Forwarded to :meth:`tune` or :meth:`tune_edge`.

        Returns
        -------
        :class:`~supervillain.generator.combining.Sequentially`
            The production-ready chain, its tuned :class:`DefectGas` last.
        """
        if edge:
            fugacity, emit_every = self.tune_edge(start=start, **kwargs)
        else:
            fugacity = self.tune(start=start, **kwargs)
            emit_every = None
        gas = DefectGas(self.S, fugacity, max_defects=self.max_defects, emit_every=emit_every,
                        rng=self.rng)
        chain = Sequentially((*self.companions, gas))
        chain.fugacity = gas.fugacity
        chain.emit_every = gas.emit_every
        return chain


class DefectGasWeightTuner:
    r"""
    Learns the :class:`DefectGas` sector table $w_k$ (its ``sectorWeights``) by
    damped histogram recursion: probe the chain, cut every visited sector's
    weight toward the mean occupancy, repeat until the sector histogram is
    flat.  Flat occupancies pay only $1/(K+1)$ vacuum dwell for full multi-pair
    traffic --- against the $e^{-\langle D/2\rangle}$ any geometric fugacity
    pays --- and cannot condense: a sector that hogs time gets its weight cut
    on the next iteration.  A second, optional stage (:meth:`tune_umbrella`)
    then learns the pair-separation table $w_2$ (the
    ``pairSeparationUmbrella``) with the sector table frozen.

    A tuner is not a generator: it runs experiments to decide which
    :class:`DefectGas` to build, and :meth:`generator` returns the
    production-ready chain with the frozen tables and a matched
    ``emit_every``.

    Probes are *sweep-budgeted* (:meth:`DefectGas._probe_sweeps`), never
    waiting for vacuum returns, so a condensing candidate produces a lopsided
    histogram --- a measurement the recursion corrects --- rather than a
    ``RuntimeError``.  Because nucleation out of the metastable vacuum can be
    slow, convergence always demands round trips and half-vs-half
    stationarity on top of flatness --- a histogram that has not equilibrated
    can look beautifully flat and be wrong.

    Parameters
    ----------
    S: a NoIntersections action
    companions: iterable of generators, optional
        Interleaved with the probe gas every ``companion_every`` sweeps (and ride in
        the :meth:`generator` chain).  Defaults to
        :class:`~supervillain.generator.villain.SiteUpdate`,
        :class:`~supervillain.generator.villain.ExactUpdate`, and
        :class:`~supervillain.generator.villain.CohomologyUpdate` --- the
        $\phi$ move plus the two $D$-neutral $n$ moves at fixed $dn$, the
        latter reaching the winding holonomy (and with it the $\theta$-current
        windings $J_\mu$) that no local move touches.  Companions must
        tolerate mid-excursion (invalid) ``n``; all three do.
    max_defects: even int
        The cap; the table has $K + 1 = $ ``max_defects/2 + 1`` sectors.
    rng: numpy Generator, optional
    uniformProposalFraction: float, sequence of floats, or None
        The mixture table $\gamma$ of the defect-adjacent proposal targeting
        (see :class:`DefectGas`), used in
        every probe AND in the production gas :meth:`generator` builds, so the
        measured dwell, ``mixing_sweeps``, and ``emit_every`` are all
        self-consistent.  ``None`` restores the legacy uniform proposal.
    """

    def __init__(self, S, companions=None, max_defects=16, rng=None,
                 uniformProposalFraction=0.5):
        self.S = S
        if max_defects is None or max_defects < 2 or max_defects % 2:
            raise ValueError(f'max_defects must be a positive even integer; got {max_defects}.')
        self.max_defects = int(max_defects)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.uniformProposalFraction = uniformProposalFraction
        if companions is not None:
            self.companions = tuple(companions)
        else:
            # FourierSiteHeatbath draws phi (exactly, and jointly -- it reads n only
            # through delta(dphi - 2 pi n), so it neither inspects nor disturbs it);
            # ExactUpdate stirs n at fixed dn (q exactly preserved); CohomologyUpdate
            # shifts the winding holonomy no local move reaches (and with it the
            # theta-current windings J_mu).  All three are D-neutral and tolerate the
            # mid-excursion (invalid-n) states the probes hand them.
            defaults = (FourierSiteHeatbath(S), ExactUpdate(S), CohomologyUpdate(S))
            for generator in defaults:
                generator.rng = self.rng
            self.companions = defaults

    def _start(self, start):
        if isinstance(start, str) and start == 'cold':
            L = self.S.Lattice
            return {'phi': np.zeros((1,) + tuple(L.dims)),
                    'n': np.zeros((4,) + tuple(L.dims), dtype=np.int64)}
        return {'phi': start['phi'], 'n': start['n']}

    def _probe(self, w, cfg, probe_sweeps, companion_every):
        # One iteration's experiment: a fresh frozen-w gas driven sweep-budgeted with
        # companions interleaved.  Returns the (total, first-half, second-half)
        # sector histograms, the round trips, and the end configuration (raw,
        # possibly invalid --- fine, the next probe continues it).
        gas = DefectGas(self.S, sectorWeights=w, rng=self.rng,
                        uniformProposalFraction=self.uniformProposalFraction)
        K1 = self.max_defects // 2 + 1
        halves = [np.zeros(K1, dtype=np.int64), np.zeros(K1, dtype=np.int64)]
        trips = 0
        done = 0
        while done < probe_sweeps:
            block = min(companion_every, probe_sweeps - done)
            cfg, t, r, _ = gas._probe_sweeps(cfg, block)
            halves[(2 * done) // probe_sweeps] += t
            trips += r
            done += block
            L = self.S.Lattice
            for companion in self.companions:
                # Companions speak Forms; the probe chain speaks raw arrays.  BOTH
                # fields are taken back: the default companions include n-movers
                # (ExactUpdate, CohomologyUpdate) whose moves are q-neutral even on
                # the mid-excursion (invalid) n; discarding them would tune the
                # dwell under different dynamics than production runs.
                updated = companion.step({'phi': Form(cfg['phi'], degree=0, lattice=L),
                                          'n': Form(cfg['n'], degree=1, lattice=L)})
                cfg = {'phi': np.asarray(updated['phi']),
                       'n': np.asarray(updated['n'])}
        return halves[0] + halves[1], halves[0], halves[1], trips, cfg

    def tune(self, start='cold', probe_sweeps=2000, companion_every=25,
             max_iterations=12, damping=0.5, flat=3.0, min_round_trips=5,
             step_sweeps=25, u=None, max_update=10.0, max_probe_growth=8,
             lighten=1.5, warmStart=None):
        r"""
        First stage: learn the sector table.  Run the recursion from the
        Poisson-envelope warm start $w_k = k!\, u^k$ (default $u = 1/V$; the
        dilute-gas entropy is roughly $\lambda^k/k!$ for $k$ pairs, so the
        factorial undoes the identical-pair suppression) --- or, when
        ``warmStart`` is given, from that table (e.g. the frozen table of a
        nearby $\kappa$: adjacent rungs of a tempering ladder have nearly
        identical sector profiles, so the recursion starts close to its fixed
        point) --- and return
        ``(sectorWeights, emit_every)`` --- the frozen, light-by-policy table
        and an ``emit_every`` sized so one production step costs about
        ``step_sweeps`` sweeps at the measured vacuum dwell.

        Each iteration probes and multiplies every *visited* sector's weight by
        $(\bar t / t_k)^\alpha$ ($\alpha = 1$ first, ``damping`` after), clipped to
        $[1/\texttt{max\_update}, \texttt{max\_update}]$ --- a sector visited by a
        handful of ticks must not receive an enormous noisy boost.  Unvisited
        sectors are left alone: extrapolating into unmeasured territory is how
        multicanonical recursions blow up.  A probe with fewer than
        ``min_round_trips`` round trips signals that the sectors mix slower than
        the probe measures, so the next probe doubles in length (up to
        ``max_probe_growth`` $\times$ ``probe_sweeps``).  Converged when every
        sector was visited, $\max_k t_k \le$ ``flat`` $\times \min_k t_k$, the
        probe completed ``min_round_trips`` round trips, and the histogram is
        half-vs-half stationary.  On iteration-cap expiry: warn and freeze the
        best *probed* table --- most round trips, then flattest, and never the
        unmeasured post-update one.  (Trips lead the score here because for a
        SECTOR table more round trips means better vacuum--top shuttling;
        contrast :meth:`tune_umbrella`, where trips-first would be perverse.)
        Production ``SectorTicks`` is the check that catches a bad freeze.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where the first probe starts; later probes continue the enlarged chain.
        probe_sweeps: int
            Sweeps per iteration.
        companion_every: int
            Sweeps between companion interleavings.
        max_iterations: int
        damping: float
            The update exponent $\alpha$ after the first iteration.
        flat: float
            Acceptable max/min sector-occupancy ratio.
        min_round_trips: int
            Vacuum--top--vacuum round trips the final probe must complete.
        step_sweeps: int
            Target sweeps per production step; sizes the returned ``emit_every``.
        u: float, optional
            The warm start's per-pair weight scale; defaults to $1/V$.
            Ignored when ``warmStart`` is given.
        warmStart: numpy array, optional
            Initial sector table, one positive entry per sector
            (``max_defects // 2 + 1``); overrides the Poisson envelope.
            Pass a neighboring κ's frozen table (a tempering ladder's rung
            above, or a smaller volume's table rescaled by
            $(V_{\rm old}/V_{\rm new})^k$ per sector $k$, matching the
            envelope's $u \propto 1/V$) to start the recursion near its
            fixed point.
        max_update: float
            Per-iteration clip on each sector's update factor.
        max_probe_growth: int
            Cap on the adaptive probe lengthening, in units of ``probe_sweeps``.
        lighten: float
            Light-by-policy factor applied to the frozen table after the
            freeze: $w_k \to w_k / \texttt{lighten}^k$, so production sits
            below sector coexistence rather than at it (the umbrella, when
            used, carries the tail statistics that coexistence traffic used
            to).  ``lighten=1.0`` disables it.

        Returns
        -------
        (numpy array, int)
            The frozen sector table --- ready to be handed to
            :class:`DefectGas` as ``sectorWeights`` --- and the matched
            ``emit_every``.
        """
        K = self.max_defects // 2
        V = self.S.Lattice.N ** 4
        if warmStart is not None:
            w = np.asarray(warmStart, dtype=np.float64).copy()
            if w.shape != (K + 1,):
                raise ValueError(f'warmStart must have one entry per sector '
                                 f'({K + 1} for max_defects={self.max_defects}); got {w.shape}.')
            if not np.all(w > 0):
                raise ValueError('warmStart must be positive.')
        else:
            u0 = (1.0 / V) if u is None else float(u)
            w = np.array([math.factorial(k) * u0**k for k in range(K + 1)])
        w /= w[0]
        cfg = self._start(start)
        sweeps = probe_sweeps
        best = None       # (trips, -flatness, w, t, sweeps) of the best PROBED table
        for iteration in range(max_iterations):
            t, t1, t2, trips, cfg = self._probe(w, cfg, sweeps, companion_every)
            flatness = t.max() / max(1, t.min())
            if best is None or (trips, -flatness) > best[:2]:
                best = (trips, -flatness, w.copy(), t.copy(), sweeps)
            visited = t > 0
            big = (t1 + t2) >= 10     # stationarity is meaningless on a handful of ticks
            stationary = bool(np.all((t2[big] <= 2 * t1[big]) & (t1[big] <= 2 * t2[big])))
            if (visited.all() and flatness <= flat
                    and trips >= min_round_trips and stationary):
                break
            if trips < min_round_trips:
                # Too few round trips to trust the histogram's shape: the sectors mix
                # slower than the probe measures -- lengthen before re-measuring.
                sweeps = min(2 * sweeps, max_probe_growth * probe_sweeps)
            alpha = 1.0 if iteration == 0 else damping
            factor = np.ones_like(w)
            factor[visited] = (t[visited].mean() / t[visited]) ** alpha
            np.clip(factor, 1.0 / max_update, max_update, out=factor)
            w = w * factor
            w /= w[0]
        else:
            trips, _, w, t, sweeps = best
            logger.warning(
                'tune: no convergence in %d iterations; freezing the best probed '
                'table (sector ticks %s, %d round trips) -- watch SectorTicks in '
                'production.', max_iterations, t.tolist(), trips)
        # Light-by-policy: sit below sector coexistence rather than at it; the
        # umbrella carries the tail statistics that coexistence traffic used to.
        w = w / lighten ** np.arange(K + 1)
        w /= w[0]
        # The measured round-trip timescale: vacuum ticks arrive in bursts spaced by
        # roughly this many sweeps, so the production step horizon must accommodate
        # it (consumed by generator()).
        self.mixing_sweeps = sweeps / max(1, trips)
        dwell = t[0] / max(1, t.sum())
        emit_every = max(1, int(round(dwell * 4 * V * step_sweeps)))
        return w, emit_every

    def _probe_umbrella(self, w, w2, cfg, probe_sweeps, companion_every):
        r"""
        One umbrella-stage experiment: a frozen-``(w, w2)`` gas, sweep-budgeted,
        companions interleaved, returning the per-bin SHELL dwell histogram
        (total, first half, second half), the round trips, and the end
        configuration.
        """
        gas = DefectGas(self.S, sectorWeights=w, pairSeparationUmbrella=w2, rng=self.rng,
                        uniformProposalFraction=self.uniformProposalFraction)
        lookup, values, rsq, nz, mult = shell_multiplicity(self.S.Lattice.N)
        halves = [np.zeros(len(values)), np.zeros(len(values))]
        trips = 0
        done = 0
        while done < probe_sweeps:
            block = min(companion_every, probe_sweeps - done)
            cfg, _, r, H_pair = gas._probe_sweeps(cfg, block, tally=True)
            shell = np.bincount(lookup[rsq[nz]],
                                weights=H_pair[nz], minlength=len(values))
            halves[(2 * done) // probe_sweeps] += shell / mult
            trips += r
            done += block
            L = self.S.Lattice
            for companion in self.companions:
                updated = companion.step({'phi': Form(cfg['phi'], degree=0, lattice=L),
                                          'n': Form(cfg['n'], degree=1, lattice=L)})
                cfg = {'phi': np.asarray(updated['phi']),
                       'n': np.asarray(updated['n'])}
        return halves[0] + halves[1], halves[0], halves[1], trips, cfg

    def tune_umbrella(self, sectorWeights, start='cold', probe_sweeps=4000,
                      companion_every=25, max_iterations=10, damping=0.5,
                      flat=3.0, min_round_trips=5, max_update=10.0,
                      max_probe_growth=4, warmStart=None):
        r"""
        Second stage: with ``sectorWeights`` frozen, learn the
        pair-separation table $w_2$ (the ``pairSeparationUmbrella``) by the
        same damped, clipped histogram recursion, driven by the per-shell
        dwell of the $D = 2$ sector.

        Each iteration's update is renormalized so the total pair-sector
        weight is unchanged: $\sum_{\text{shell}} \text{mult} \times
        \hat\Theta \times w_2$ is held fixed, with $\hat\Theta \propto h /
        w_2^{\text{current}}$ the per-shell physics estimate implied by the
        current table's dwell $h$.  This decouples the knobs: the sector
        table owns sector traffic, the umbrella owns the within-sector
        profile.

        Converged when every shell was visited, the occupancy is flat, the
        histogram is half-vs-half stationary, AND the probe completed
        ``min_round_trips`` round trips --- the umbrella must not silently
        trade transport for flatness.  A probe with too few round trips, an
        unvisited shell, or a non-stationary histogram lengthens the next
        probe.  On iteration-cap expiry: warn and freeze the best *probed*
        table, scored by round-trip health as a boolean tier (healthy or
        not), then shells visited, then flatness.  Health is a gate, not a
        magnitude, because a heavier umbrella always completes FEWER round
        trips --- score by trips (as :meth:`tune` correctly does for the
        sector table) and the recursion would freeze the least-umbrella'd
        probe every time.

        Parameters
        ----------
        sectorWeights: numpy array
            The frozen sector table from :meth:`tune`.
        start: 'cold', or a configuration as a dictionary
            Where the first probe starts; later probes continue the enlarged chain.
        probe_sweeps: int
            Sweeps per iteration.
        companion_every: int
            Sweeps between companion interleavings.
        max_iterations: int
        damping: float
            The update exponent applied after the first iteration.
        flat: float
            Acceptable max/min shell-occupancy ratio.
        min_round_trips: int
            Vacuum--top--vacuum round trips the final probe must complete.
        max_update: float
            Per-iteration clip on each shell's update factor.
        max_probe_growth: int
            Cap on the adaptive probe lengthening, in units of ``probe_sweeps``.
        warmStart: numpy array, optional
            Initial umbrella table, one positive entry per realized shell;
            defaults to all-ones.  Because unvisited shells never receive a
            boost (extrapolating into unmeasured territory is how
            multicanonical recursions blow up), an all-ones start can only
            grow the umbrella outward shell by shell --- hopeless within the
            iteration cap when the decay is steep.  At such couplings pass
            $w_2 \sim 1/\hat\Theta$ per shell from a prior run or a stored
            correlator, so the recursion starts with the far shells already
            lifted.

        Returns
        -------
        numpy array
            The frozen umbrella table --- ready to be handed to
            :class:`DefectGas` as ``pairSeparationUmbrella``.
        """
        w = sectorWeights
        _, values, _, _, mult = shell_multiplicity(self.S.Lattice.N)
        if warmStart is None:
            w2 = np.ones(len(values))
        else:
            warmStart = np.asarray(warmStart, dtype=np.float64)
            if warmStart.shape != values.shape:
                raise ValueError(f'warmStart must have one entry per realized shell '
                                 f'({len(values)} for N={self.S.Lattice.N}); got {warmStart.shape}.')
            if not np.all(warmStart > 0):
                raise ValueError('warmStart must be positive.')
            w2 = warmStart.copy()
        cfg = self._start(start)
        sweeps = probe_sweeps
        best = None       # (healthy, visited, -flatness, trips, w2, h) of the best PROBED table
        for iteration in range(max_iterations):
            h, h1, h2, trips, cfg = self._probe_umbrella(
                w, w2, cfg, sweeps, companion_every)
            visited = h > 0
            flatness = (h[visited].max() / h[visited].min()) if visited.any() else np.inf
            score = (int(trips >= min_round_trips), int(visited.sum()), -flatness)
            if best is None or score > best[:3]:
                best = (*score, trips, w2.copy(), h.copy())
            big = (h1 + h2) >= 10
            stationary = bool(np.all((h2[big] <= 2 * h1[big])
                                     & (h1[big] <= 2 * h2[big])))
            if (visited.all() and flatness <= flat and stationary
                    and trips >= min_round_trips):
                break
            if not visited.all() or not stationary or trips < min_round_trips:
                sweeps = min(2 * sweeps, max_probe_growth * probe_sweeps)
            if visited.any():
                alpha = 1.0 if iteration == 0 else damping
                factor = np.ones_like(w2)
                factor[visited] = (h[visited].mean() / h[visited]) ** alpha
                np.clip(factor, 1.0 / max_update, max_update, out=factor)
                # Renormalize so the total pair-sector weight is unchanged: the
                # per-bin physics estimate implied by the CURRENT table is
                # Theta_hat = h / w2, so the conserved sector-total carrier is
                # mult * Theta_hat -- NOT h itself (which already carries a
                # factor of w2).
                theta_hat = np.where(w2 > 0, h / w2, 0.0)
                sector = mult * theta_hat
                w2new = w2 * factor
                w2 = w2new * max(1e-300, sector @ w2) / max(1e-300, sector @ w2new)
        else:
            _, _, _, trips, w2, h = best
            logger.warning(
                'tune_umbrella: no convergence in %d iterations (shell dwell '
                '%s, %d round trips); freezing the best probed table.',
                max_iterations, np.array2string(h, precision=2), trips)
        return w2

    def generator(self, start='cold', umbrella=False, umbrella_kwargs=None, **kwargs):
        r"""
        The normal way to consume a tune: run :meth:`tune` from ``start``
        (``kwargs`` forward), optionally follow with :meth:`tune_umbrella`
        (``umbrella_kwargs`` forward) at frozen sector table, and return the
        production-ready chain --- the companions followed by a
        :class:`DefectGas` built with the frozen tables, the matched
        ``emit_every``, and a step horizon sized to the measured sector-mixing
        time (vacuum ticks arrive in bursts, and a horizon blind to that
        kills healthy chains by timeout).  The chain carries
        ``sectorWeights``, ``pairSeparationUmbrella``, and ``emit_every`` as
        plain metadata for introspection.

        Parameters
        ----------
        start: 'cold', or a configuration as a dictionary
            Where the tuning probes start; anything
            :meth:`~supervillain.Ensemble.generate` accepts.
        umbrella: bool
            Run the second stage (:meth:`tune_umbrella`) after :meth:`tune` and
            hand the learned pair-separation shell table to production.
        umbrella_kwargs: dict, optional
            Forwarded to :meth:`tune_umbrella`.
        kwargs:
            Forwarded to :meth:`tune`.

        Returns
        -------
        :class:`~supervillain.generator.combining.Sequentially`
            The production-ready chain, its tuned :class:`DefectGas` last.
        """
        w, emit_every = self.tune(start=start, **kwargs)
        w2 = None
        if umbrella:
            w2 = self.tune_umbrella(w, start=start, **(umbrella_kwargs or {}))
        # Vacuum ticks arrive in bursts spaced by the sector-mixing time; the step
        # horizon must be generous relative to it or healthy chains die by timeout.
        horizon = max(500, int(round(20 * self.mixing_sweeps)))
        gas = DefectGas(self.S, sectorWeights=w, pairSeparationUmbrella=w2,
                        emit_every=emit_every, max_step_sweeps=horizon,
                        rng=self.rng,
                        uniformProposalFraction=self.uniformProposalFraction)
        chain = Sequentially((*self.companions, gas))
        chain.sectorWeights = gas.sectorWeights
        chain.pairSeparationUmbrella = gas.pairSeparationUmbrella
        chain.emit_every = gas.emit_every
        return chain
