#!/usr/bin/env python
r"""
Compiled tables and Green machinery for the :class:`~.SurfaceWormGas`.

The surface worm's per-proposal work (a single-plaquette toggle of the integer
2-form $F$) is $O(1)$ because every quantity the acceptance needs --- the change
in the coexact norm, the change in the closure defect $dF$, and whether $F\wedge F$
would leave zero --- is read off a **local stencil** rather than recomputed from
scratch.  This module builds those stencils once (cached per lattice size $N$) and
the scalar lattice Green's function that turns a single-plaquette toggle into an
$O(V)$ (not $O(V^2)$) update of the incremental potential.

Layout: tables and builders come first; the batch ``njit`` kernel (``gas_batch``
and its helpers) is appended below them, after everything it depends on.

.. note ::
    All of this is a straight port of the audited reference implementation
    (``worm.py``, ``stencils.py``, ``worm_numba.py``, ``gas.py`` in the
    no-intersections lab notebook's ``swg-audit-2026-07-31`` snapshot), gated
    against the library's own :func:`~.lattice.d` and :func:`~.lattice.wedge`
    operators rather than re-derived here.  See ``test_surface_worm_kernel_tables.py``
    for the gates.
"""

import numpy as np
from numba import njit

from supervillain.lattice import Lattice, d, wedge


# ============================================================ scalar Green's function
def scalar_green(N):
    r"""The scalar lattice Green's function $g_0 = \Delta^{-1}$ on $T^4$, zero-mode removed.

    $\Delta$ is the standard finite-difference Laplacian on the $N^4$ torus, diagonal
    in Fourier space with symbol $\hat k^2 = 4\sum_\mu \sin^2(k_\mu/2)$.  Its zero
    mode ($\hat k = 0$) is dropped (set to $0$ rather than inverted), which is exactly
    what is needed to solve $\Delta \phi = \rho$ for a source $\rho$ that integrates to
    zero over the torus (the physical case here: $F\wedge F = 0$ never sources a net
    charge).

    Parameters
    ----------
    N: int
        Linear lattice extent (the same $N$ in all four directions).

    Returns
    -------
    g0: numpy.ndarray
        The real-space Green's function, shape ``(N,)*4``, translation-invariant so a
        toggle at any site is read off by ``numpy.roll``.
    self_energy: float
        $g_0(0)$, the coincident-point value every diagonal (self-interaction) term
        in the worm's acceptance ratio needs.
    """
    k = 2 * np.pi * np.fft.fftfreq(N)
    K = np.meshgrid(k, k, k, k, indexing='ij')
    k2 = 4 * sum(np.sin(Ki / 2) ** 2 for Ki in K)
    with np.errstate(divide='ignore'):
        inv = np.where(k2 == 0, 0.0, 1.0 / k2)
    g0 = np.real(np.fft.ifftn(inv))
    return g0, float(g0.reshape(-1)[0])


def pot(Fc, N):
    r"""Solve $\Delta\phi = F_c$ on $T^4$ by FFT, zero mode dropped (see :func:`scalar_green`).

    Parameters
    ----------
    Fc: numpy.ndarray
        A single plaquette component's source, shape ``(N,)*4``.
    N: int
        Linear lattice extent.

    Returns
    -------
    numpy.ndarray
        $\phi = \Delta^{-1} F_c$, real, shape ``(N,)*4``.
    """
    k = 2 * np.pi * np.fft.fftfreq(N)
    K = np.meshgrid(k, k, k, k, indexing='ij')
    k2 = 4 * sum(np.sin(Ki / 2) ** 2 for Ki in K)
    with np.errstate(divide='ignore'):
        inv = np.where(k2 == 0, 0.0, 1.0 / k2)
    return np.real(np.fft.ifftn(np.fft.fftn(Fc) * inv))


# ============================================================ local stencils
# d-stencil:   toggling F[c, x] += s changes the 3-form dF by
#              s*sign on cube (cube_comp, x + off) for each entry.
# wedge-stencil: F∧F is a top-form (scalar per site). Toggling F[c, x] += s
#              changes (F∧F) at site x + off by
#              s * sign * F[partner_comp, x + poff], summed over entries
#              (bilinear cross terms; a plaquette never self-intersects).
# Both are translation-invariant, so the origin stencil + np.roll suffices.

_CACHE = {}


def _build(N):
    r"""Derive the $d$- and wedge-cross stencils at the origin plaquette by probing the
    library's own :func:`~.lattice.d` and :func:`~.lattice.wedge` on unit forms.

    Not called directly; :func:`stencils` caches the result per $N$.
    """
    L = Lattice(4, N)

    def unit(c, x=(0, 0, 0, 0)):
        F = L.form(2); a = np.asarray(F); a[:] = 0; a[(c,) + x] = 1
        return F

    # d-stencil: (cube_comp, offset(4), sign) per plaquette comp c
    dsten = []
    for c in range(6):
        dF = np.asarray(d(unit(c)))
        entries = []
        for z in np.argwhere(dF != 0):
            entries.append((int(z[0]), tuple(int(v) % N for v in z[1:]), int(dF[tuple(z)])))
        dsten.append(entries)

    # wedge cross-stencil: for toggling F[c,origin], the change in (F∧F) is
    #   Δ(F∧F)[h] = s · Σ_entries sign · F[pc, h + poff_rel]
    # Extract by: wedge is bilinear & symmetric; put unit at (c,origin) and a
    # unit partner at (pc, y); the cross term appears in wedge(F,F) with
    # coefficient 2·(structure). Recover the per-(c) list of (pc, rel, sign)
    # such that Δ(F∧F)[h] += sign·F[pc, h+rel] when F[c,origin] toggles by +1.
    # Do it cleanly via finite difference on a random background.
    rng = np.random.default_rng(0)
    wsten = []
    for c in range(6):
        bg = L.form(2); bga = np.asarray(bg)
        bga[:] = rng.integers(-2, 3, size=bga.shape)
        w0 = np.asarray(wedge(bg, bg))
        bga[(c,) + (0, 0, 0, 0)] += 1
        w1 = np.asarray(wedge(bg, bg))
        dW = w1 - w0                      # = Δ(F∧F) for this +1 toggle, exact (linear+self; self=0)
        # dW[h] must equal Σ sign·F[pc, h+rel]; identify (pc,rel,sign) by
        # correlating dW against each partner field shifted. Robust route:
        # dW is linear in bg, so its dependence on bg[pc, z] gives the stencil.
        # Recompute analytically: probe each (pc, z) unit partner on zero bg.
        bga[:] = 0
        entries = {}
        for pc in range(6):
            for z in np.ndindex((N, N, N, N)):
                bga[:] = 0
                bga[(c,) + (0, 0, 0, 0)] = 1
                bga[(pc,) + z] += 1
                wcross = np.asarray(wedge(bg, bg))
                # subtract the pure-c self term (0) — wcross is entirely the cross term
                for h in np.argwhere(wcross[0] != 0):
                    hh = (0,) + tuple(int(v) for v in h)
                    val = int(wcross[hh])
                    # this contributes: Δ(F∧F)[hh] gets val from partner (pc, z).
                    # relative offsets: site of toggled plaq = origin; partner at z; affected h.
                    rel = tuple(int(v) % N for v in z)          # partner offset rel to toggle site
                    hrel = tuple(int(v) % N for v in h)         # affected-cell offset (4d) rel to toggle site
                    entries.setdefault((pc, rel, hrel), 0)
                    entries[(pc, rel, hrel)] += val
        # keep nonzero
        wsten.append([(pc, rel, hrel, v) for (pc, rel, hrel), v in entries.items() if v != 0])
    return dsten, wsten


def stencils(N):
    r"""The $(d, F\wedge F)$ local-update stencils at lattice size $N$, cached.

    Parameters
    ----------
    N: int
        Linear lattice extent.

    Returns
    -------
    dsten: list
        Six entries (one per plaquette component $c$), each a list of
        ``(cube_comp, offset, sign)`` triples: toggling ``F[c, x] += s`` changes the
        3-form $dF$ by ``s*sign`` on cube ``(cube_comp, x + offset)``, for every entry.
    wsten: list
        Six entries (one per plaquette component $c$), each a list of
        ``(partner_comp, rel, hrel, v)`` quadruples: toggling ``F[c, x] += s`` changes
        $F\wedge F$ at site ``x + hrel`` by ``s * v * F[partner_comp, x + rel]``,
        summed over entries (the bilinear cross term; a plaquette never
        self-intersects).
    """
    if N not in _CACHE:
        _CACHE[N] = _build(N)
    return _CACHE[N]


# ============================================================ flat numba tables
def build_arrays(N):
    r"""Flatten :func:`stencils`\ 's per-$N$ python lists into fixed-shape integer
    arrays the ``njit`` kernels can index without Python overhead.

    Parameters
    ----------
    N: int
        Linear lattice extent.

    Returns
    -------
    dsten_cc: numpy.ndarray
        Shape ``(6, 4)``; ``dsten_cc[c, k]`` is the cube component of the $k$-th
        $d$-stencil entry for plaquette component $c$ (always exactly 4 entries).
    dsten_off: numpy.ndarray
        Shape ``(6, 4, 4)``; the corresponding 4-vector offsets.
    dsten_sign: numpy.ndarray
        Shape ``(6, 4)``; the corresponding signs.
    faces_fc: numpy.ndarray
        Shape ``(4, 6)``; ``faces_fc[cc, f]`` is the plaquette component of the
        $f$-th of the 6 plaquettes bordering cube component ``cc``.
    faces_rel: numpy.ndarray
        Shape ``(4, 6, 4)``; the corresponding relative offsets (from the cube to the
        plaquette).
    w_pc, w_rel, w_hrel, w_v: numpy.ndarray
        Shape ``(6, T)`` / ``(6, T, 4)`` / ``(6, T, 4)`` / ``(6, T)`` where $T$ is the
        largest per-component wedge-stencil length: the flattened wedge cross-terms
        (partner component, partner relative offset, affected-cell relative offset,
        coefficient), padded with zeros past each component's ``w_nterm[c]``.
    w_gid: numpy.ndarray
        Shape ``(6, T)``; groups wedge terms that land on the same affected cell
        ``hrel`` under a shared group id, so ``_intersects``-style kernels can test
        one running sum per group instead of per term.
    w_nterm: numpy.ndarray
        Shape ``(6,)``; the number of valid wedge-stencil terms per component.
    w_ngroup: numpy.ndarray
        Shape ``(6,)``; the number of distinct groups (affected cells) per component.
    """
    dsten, wsten = stencils(N)
    # d-stencil: 4 cubes per plaquette comp c
    dsten_cc = np.zeros((6, 4), np.int64)
    dsten_off = np.zeros((6, 4, 4), np.int64)
    dsten_sign = np.zeros((6, 4), np.int64)
    for c in range(6):
        for k, (cc, off, sign) in enumerate(dsten[c]):
            dsten_cc[c, k] = cc
            dsten_off[c, k] = off
            dsten_sign[c, k] = sign
    # faces(cc): 6 plaquettes (fc, rel) bordering cube cc
    faces = {cc: [] for cc in range(4)}
    for c in range(6):
        for (cc, off, sign) in dsten[c]:
            faces[cc].append((c, tuple((-o) % N for o in off)))
    faces_fc = np.zeros((4, 6), np.int64)
    faces_rel = np.zeros((4, 6, 4), np.int64)
    for cc in range(4):
        for f, (fc, rel) in enumerate(faces[cc]):
            faces_fc[cc, f] = fc
            faces_rel[cc, f] = rel
    # wedge groups: per c, terms (pc, rel, v) with a group id (by hrel)
    T = max(len(wsten[c]) for c in range(6))
    w_pc = np.zeros((6, T), np.int64)
    w_rel = np.zeros((6, T, 4), np.int64)
    w_hrel = np.zeros((6, T, 4), np.int64)     # affected-cell offset (for lc accumulation)
    w_v = np.zeros((6, T), np.int64)
    w_gid = np.zeros((6, T), np.int64)
    w_nterm = np.zeros(6, np.int64)
    w_ngroup = np.zeros(6, np.int64)
    for c in range(6):
        groups = {}
        for (pc, rel, hrel, v) in wsten[c]:
            groups.setdefault(hrel, len(groups))
        for t, (pc, rel, hrel, v) in enumerate(wsten[c]):
            w_pc[c, t] = pc
            w_rel[c, t] = rel
            w_hrel[c, t] = hrel
            w_v[c, t] = v
            w_gid[c, t] = groups[hrel]
        w_nterm[c] = len(wsten[c])
        w_ngroup[c] = len(groups)
    return (dsten_cc, dsten_off, dsten_sign, faces_fc, faces_rel,
            w_pc, w_rel, w_hrel, w_v, w_gid, w_nterm, w_ngroup)


def group_hrel(N):
    r"""Per plaquette comp $c$ and wedge-group id, the affected-cell offset ``hrel``.

    An index built on top of :func:`build_arrays`: the wedge-stencil terms for a
    plaquette component are already grouped by ``w_gid`` (:func:`build_arrays`'s
    docstring), and this collapses each group down to the single ``hrel`` its terms
    share, for kernels that only need "which cell did group $g$ land on" rather than
    the per-term detail.

    Parameters
    ----------
    N: int
        Linear lattice extent.

    Returns
    -------
    tuple
        ``(dsten_cc, dsten_off, dsten_sign, w_pc, w_rel, w_hrel, w_v, w_gid,
        w_nterm, w_ngroup, ghrel)`` --- the same tables :func:`build_arrays` returns
        for the $d$- and wedge-stencils (``faces_fc``/``faces_rel`` dropped, not
        needed by the kernels that consume this), plus ``ghrel``, shape
        ``(6, max(w_ngroup), 4)``: ``ghrel[c, g]`` is the ``hrel`` shared by every
        term in group $g$ of component $c$.
    """
    (dsten_cc, dsten_off, dsten_sign, faces_fc, faces_rel,
     w_pc, w_rel, w_hrel, w_v, w_gid, w_nterm, w_ngroup) = build_arrays(N)
    maxg = int(w_ngroup.max())
    ghrel = np.zeros((6, maxg, 4), np.int64)
    for c in range(6):
        for t in range(int(w_nterm[c])):
            ghrel[c, int(w_gid[c, t])] = w_hrel[c, t]
    return (dsten_cc, dsten_off, dsten_sign, w_pc, w_rel, w_hrel, w_v, w_gid,
            w_nterm, w_ngroup, ghrel)


# ============================================================ njit helpers
@njit(cache=True)
def nb_seed(s):
    r"""Seed numba's OWN RNG stream (Python's ``np.random.seed`` does NOT --- separate
    state). Must be called from ``njit`` for the jitted inner loops to be controlled."""
    np.random.seed(s)


@njit(cache=True)
def green_add(Gc, g0, x0, x1, x2, x3, s, N):
    r"""``Gc += s * roll(g0, (x0..x3))`` --- the incremental potential update, $O(V)$.

    The per-proposal cost of the worm's local move: toggling plaquette component $c$
    at site $x$ by $s$ shifts that component's potential $\phi_c = \Delta^{-1}F_c$ by
    $s$ times the Green's function recentred at $x$, so the whole $V$-site potential
    is patched in $O(V)$ rather than refetched via a fresh $O(V\log V)$ FFT.
    """
    for y0 in range(N):
        a0 = (y0 - x0) % N
        for y1 in range(N):
            a1 = (y1 - x1) % N
            for y2 in range(N):
                a2 = (y2 - x2) % N
                for y3 in range(N):
                    Gc[y0, y1, y2, y3] += s * g0[a0, a1, a2, (y3 - x3) % N]


# ============================================================ coboundary tables
def build_cob(S):
    r"""Per direction $\mu$: the coboundary $d(e_\mu)$ of the unit 1-form, flattened,
    plus its coexact-norm cost $K_\text{cob}[\mu]$.

    The gas's coboundary move shifts the whole 1-form winding sector by adding
    $d(e_\mu)$ to $F$ --- a global, exact 2-form with exactly 6 nonzero plaquette
    cells (one per pair of directions involving $\mu$) --- which changes the
    coexact norm $C(F) = (1/V)\sum_{k\neq0} |\tilde F(k)|^2/\hat k^2$ by a
    $\mu$-dependent constant $K_\text{cob}[\mu]$ (plus the usual Green cross term,
    computed elsewhere): $d(e_\mu)$ is a fixed pattern independent of $F$, so its own
    coexact norm is a one-time FFT.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action, used only for its lattice (``S.Lattice.N``).

    Returns
    -------
    cob_pc: numpy.ndarray
        Shape ``(4, 6)``; ``cob_pc[mu, j]`` is the plaquette component of the $j$-th
        nonzero cell of $d(e_\mu)$.
    cob_off: numpy.ndarray
        Shape ``(4, 6, 4)``; the corresponding site offsets (mod $N$).
    cob_sign: numpy.ndarray
        Shape ``(4, 6)``; the corresponding signs (values of $d(e_\mu)$ there).
    Kcob: numpy.ndarray
        Shape ``(4,)``; $K_\text{cob}[\mu] = (1/V)\sum_{k\neq0}|\widetilde{d(e_\mu)}(k)|^2/\hat k^2$,
        the coexact norm of $d(e_\mu)$ alone.
    """
    N = S.Lattice.N; V = N ** 4
    cob_pc = np.zeros((4, 6), np.int64)
    cob_off = np.zeros((4, 6, 4), np.int64)
    cob_sign = np.zeros((4, 6), np.int64)
    Kcob = np.zeros(4, np.float64)
    for mu in range(4):
        a = S.Lattice.form(1); aa = np.asarray(a); aa[:] = 0; aa[(mu,) + (0, 0, 0, 0)] = 1
        F = np.asarray(d(a))
        cells = np.argwhere(F != 0)
        for j, z in enumerate(cells):
            cob_pc[mu, j] = z[0]
            cob_off[mu, j] = [int(v) % N for v in z[1:]]
            cob_sign[mu, j] = int(F[tuple(z)])
        Ff = np.zeros((6,) + (N,) * 4)
        for j in range(6):
            Ff[(cob_pc[mu, j],) + tuple(cob_off[mu, j])] = cob_sign[mu, j]
        Ft = np.fft.fftn(Ff, axes=(1, 2, 3, 4))
        k = 2 * np.pi * np.fft.fftfreq(N); K = np.meshgrid(k, k, k, k, indexing='ij')
        k2 = 4 * sum(np.sin(Ki / 2) ** 2 for Ki in K); k2s = np.where(k2 == 0, 1.0, k2)
        Kcob[mu] = float((np.abs(Ft) ** 2 / k2s * (k2 != 0)).sum()) / V
    return cob_pc, cob_off, cob_sign, Kcob


# ============================================================ batch njit kernel
@njit(cache=True)
def log_winding_1d(w, quantum, a):
    r"""One direction's contribution to $\log Z_\text{wind}$ — the same nine-term,
    max-shifted logsumexp the python reference sums in ``_log_winding_weight``.

    Kept arithmetically identical to the reference on purpose: the kernel and the
    reference are cross-gated against each other, so any divergence here would read
    as a physics disagreement rather than as a code difference."""
    centre = int(round(-w / quantum))
    best = -1.0e300
    for k in range(centre - 4, centre + 5):
        m = w + k * quantum
        lw = -a * m * m
        if lw > best:
            best = lw
    total = 0.0
    for k in range(centre - 4, centre + 5):
        m = w + k * quantum
        total += np.exp(-a * m * m - best)
    return best + np.log(total)


@njit(cache=True)
def log_proposal_density_nb(openCells, D, targetFraction, sites):
    r"""Compiled twin of ``SurfaceWormGas._log_proposal_density``.  D == 0 means the
    targeted branch has no cell to draw, so the proposal is purely uniform."""
    if D == 0 or targetFraction == 0.0:
        return -np.log(sites)
    return np.log((1.0 - targetFraction) / sites + targetFraction * openCells / (6.0 * D))


@njit(cache=True)
def log_sector_weight(D, table, tailSlope, hardWall):
    r"""log w(D) -- the compiled twin of ``SectorWeights.__call__``.

    With ``hardWall`` the window [0, cap] is CLOSED: log w = -inf beyond it, so the
    Metropolis exponent goes to -inf and any move proposing D > cap is rejected. Detailed
    balance is preserved because w is still a genuine function of D (zero outside), and the
    reverse of a forbidden move is equally forbidden.

    Without it, the linear tail applies. See the PROVISIONAL note in ``SectorWeights``: the
    tail is an escape hatch that let the walk leave the flattening window and never return
    (N=6, cap=12: all 3000 ticks above cap by iteration 49), which is why the wall is the
    current default and why it may need revisiting."""
    cap = table.shape[0] - 1
    if D <= cap:
        return table[D]
    if hardWall:
        return -np.inf
    return table[cap] + (D - cap) * tailSlope



@njit(cache=True)
def sep2_flat(i0, i1, N):
    r"""Squared minimal-image separation between two flat 4D cell indices.

    Flat indices, not tuples: numba handles integer arithmetic far better than tuple
    juggling, and the kernel already speaks flat indices everywhere else."""
    a0 = i0 // (N * N * N); r0 = i0 % (N * N * N)
    a1 = r0 // (N * N);     r0 = r0 % (N * N)
    a2 = r0 // N;           a3 = r0 % N
    b0 = i1 // (N * N * N); r1 = i1 % (N * N * N)
    b1 = r1 // (N * N);     r1 = r1 % (N * N)
    b2 = r1 // N;           b3 = r1 % N
    s = 0
    d = (a0 - b0) % N
    if N - d < d:
        d = N - d
    s += d * d
    d = (a1 - b1) % N
    if N - d < d:
        d = N - d
    s += d * d
    d = (a2 - b2) % N
    if N - d < d:
        d = N - d
    s += d * d
    d = (a3 - b3) % N
    if N - d < d:
        d = N - d
    s += d * d
    return s


@njit(cache=True)
def pair_log_umbrella(nCharge, chargeList, qflat, N, table):
    r"""log w_2 of the CURRENT state: 0 unless exactly two cells carry +1 and -1.

    The same condition CorrelatorAccumulator.tick bins on.  If the weight and the
    accumulator disagreed about what counts as a pair, the division would not undo the
    bias."""
    if nCharge != 2:
        return 0.0
    i0 = chargeList[0]
    i1 = chargeList[1]
    v0 = qflat[i0]
    v1 = qflat[i1]
    if not ((v0 == 1 and v1 == -1) or (v0 == -1 and v1 == 1)):
        return 0.0
    return table[sep2_flat(i0, i1, N)]


@njit(cache=True)
def cob_log_umbrella(D, nCharge, chargeList, qflat, hhFlat, vv, nu, N, table, postList):
    r"""$\log w_2$ of the state the coboundary heatbath would reach for shift ``D``.

    The heatbath draws $D$ from a normalized categorical over the local conditional, so
    every candidate's log weight must carry the umbrella --- including $D=0$.  Omitting it
    is not a small bias: the coboundary move is the one that changes $q = F\wedge F$ at
    fixed $dF$, i.e. it is *the* move that transports the pair, so leaving $w_2$ out of it
    while the plaquette move carries it gives the two updates different stationary
    distributions and the chain converges to neither.
    """
    nPost = 0
    for jj in range(nCharge):
        cell = chargeList[jj]
        touched = False
        for u in range(nu):
            if hhFlat[u] == cell:
                touched = True
                break
        if not touched:
            postList[nPost] = cell
            nPost += 1
    for u in range(nu):
        if qflat[hhFlat[u]] + D * vv[u] != 0:
            postList[nPost] = hhFlat[u]
            nPost += 1
    if nPost != 2:
        return 0.0
    v0 = qflat[postList[0]]
    v1 = qflat[postList[1]]
    for u in range(nu):
        if hhFlat[u] == postList[0]:
            v0 = qflat[postList[0]] + D * vv[u]
        if hhFlat[u] == postList[1]:
            v1 = qflat[postList[1]] + D * vv[u]
    if not ((v0 == 1 and v1 == -1) or (v0 == -1 and v1 == 1)):
        return 0.0
    return table[sep2_flat(postList[0], postList[1], N)]


@njit(cache=True)
def gas_batch(nmoves, p_cob, F, dF, q, G, g0, counts, ctr,
                  N, V, kappa, chargeLogWeight, chargeTailSlope, chargeHardWall, self_energy,
                  dsten_cc, dsten_off, dsten_sign,
                  w_pc, w_rel, w_v, w_gid, w_nterm, w_ngroup, ghrel,
                  cob_pc, cob_off, cob_sign, Kcob, Harr,
                  windingSensitivity, winding, periods, quantum, windingCoefficient,
                  edgeTolerance, windowCap, sectorLogWeight, sectorTailSlope, sectorHardWall,
                  targetFraction, revPc, revOff, openList, openPos,
                  pairUmbrellaLog, chargeList, chargePos, affIdx, affNew, postList,
                  affIdxApplied):
    r"""counts=[D,Q]; ctr=[plaq_prop,plaq_acc,cob_prop,cob_acc,targeted] (5 entries:
    ``targeted`` counts how often the defect-adjacent proposal actually fired,
    i.e. ``targetFraction`` branch taken AND ``D>0``); winding is the
    4-vector maintained in place.  ``sectorLogWeight``/``sectorTailSlope`` carry the
    open-surface weight table w(D); with the default fugacity table log w is linear in D
    and this reproduces the old ``dD * log(eta_dF)`` pricing exactly.  There is no separate
    ``lg_dF`` argument: the table is the sole open-surface price, and passing both was the
    ambiguity ``SurfaceWormGas.__init__`` now rejects.

    ``chargeLogWeight``/``chargeTailSlope``/``chargeHardWall`` are the same three for the
    intersection count Q, and arrived the same way and for the same reason: a bare scalar
    price is linear in the exponent, so it can shift the Q distribution but never broaden
    it, and Q is the axis a torus-wrapping sheet's saddle actually lives on.  A
    ``Fugacity`` reproduces the old scalar pricing exactly, so the migration is testable
    rather than a leap."""
    # Occupancy list of open cells, so the targeted draw is O(1) instead of an O(V) scan.
    # Rebuilt once per batch (O(4V), negligible against thousands of moves) and maintained
    # incrementally on every accepted toggle by swapping with the last entry.
    nOpen = 0
    for cc in range(4):
        for a0 in range(N):
            for a1 in range(N):
                for a2 in range(N):
                    for a3 in range(N):
                        flat = cc * V + ((a0 * N + a1) * N + a2) * N + a3
                        if dF[cc, a0, a1, a2, a3] != 0:
                            openList[nOpen] = flat
                            openPos[flat] = nOpen
                            nOpen += 1
                        else:
                            openPos[flat] = -1
    # Charged-cell occupancy list, same pattern as openList: the umbrella needs the pair
    # separation on EVERY plaquette proposal, and an O(V) scan per move is impossible.
    # Rebuilt once per batch (O(V), negligible against thousands of moves) and maintained
    # incrementally on every accepted q change.
    qflat = q.reshape(-1)
    nCharge = 0
    for cell in range(V):
        if qflat[cell] != 0:
            chargeList[nCharge] = cell
            chargePos[cell] = nCharge
            nCharge += 1
        else:
            chargePos[cell] = -1
    sites = 6.0 * V
    twopi2k = 2.0 * np.pi ** 2 * kappa
    maxg = ghrel.shape[1]
    cellFlat = np.empty(4, np.int64)
    cellNew = np.empty(4, np.int64)
    gdq = np.empty(maxg, np.int64)
    maxK = 6 * w_pc.shape[1]
    hh = np.empty((maxK, 4), np.int64)
    vv = np.empty(maxK, np.int64)
    used = np.empty(maxK, np.uint8)
    hhFlat = np.empty(maxK, np.int64)   # flat form of hh, for the umbrella's candidates
    for _ in range(nmoves):
        if np.random.random() < p_cob:
            # ---------- coboundary heatbath ----------
            ctr[2] += 1
            mu = np.random.randint(4)
            y0 = np.random.randint(N); y1 = np.random.randint(N)
            y2 = np.random.randint(N); y3 = np.random.randint(N)
            Kc = Kcob[mu]
            L = 0.0
            for j in range(6):
                pc = cob_pc[mu, j]
                xp0 = (y0 + cob_off[mu, j, 0]) % N; xp1 = (y1 + cob_off[mu, j, 1]) % N
                xp2 = (y2 + cob_off[mu, j, 2]) % N; xp3 = (y3 + cob_off[mu, j, 3]) % N
                L += cob_sign[mu, j] * G[pc, xp0, xp1, xp2, xp3]
            # gather per-unit q change dq1 over the 6 plaquettes' wedge terms
            K = 0
            for j in range(6):
                pc = cob_pc[mu, j]; sgn = cob_sign[mu, j]
                xp0 = (y0 + cob_off[mu, j, 0]) % N; xp1 = (y1 + cob_off[mu, j, 1]) % N
                xp2 = (y2 + cob_off[mu, j, 2]) % N; xp3 = (y3 + cob_off[mu, j, 3]) % N
                for t in range(w_nterm[pc]):
                    p0 = (xp0 + w_rel[pc, t, 0]) % N; p1 = (xp1 + w_rel[pc, t, 1]) % N
                    p2 = (xp2 + w_rel[pc, t, 2]) % N; p3 = (xp3 + w_rel[pc, t, 3]) % N
                    vv[K] = sgn * w_v[pc, t] * F[w_pc[pc, t], p0, p1, p2, p3]
                    gg = w_gid[pc, t]     # terms in a group share hrel (= ghrel)
                    hh[K, 0] = (xp0 + ghrel[pc, gg, 0]) % N
                    hh[K, 1] = (xp1 + ghrel[pc, gg, 1]) % N
                    hh[K, 2] = (xp2 + ghrel[pc, gg, 2]) % N
                    hh[K, 3] = (xp3 + ghrel[pc, gg, 3]) % N
                    K += 1
            # dedupe hh->unique summed
            for a in range(K):
                used[a] = 0
            nu = 0
            for a in range(K):
                if used[a] == 1:
                    continue
                tot = vv[a]
                for b in range(a + 1, K):
                    if used[b] == 0 and hh[b, 0] == hh[a, 0] and hh[b, 1] == hh[a, 1] and hh[b, 2] == hh[a, 2] and hh[b, 3] == hh[a, 3]:
                        tot += vv[b]; used[b] = 1
                used[a] = 1
                if tot != 0:
                    hh[nu, 0] = hh[a, 0]; hh[nu, 1] = hh[a, 1]
                    hh[nu, 2] = hh[a, 2]; hh[nu, 3] = hh[a, 3]
                    vv[nu] = tot; nu += 1
            # Flat indices of the affected q-cells, AFTER the in-place dedupe compaction --
            # the umbrella's candidate weights need them once per heatbath, not per D.
            for u in range(nu):
                hhFlat[u] = (((hh[u, 0] * N + hh[u, 1]) * N + hh[u, 2]) * N + hh[u, 3])
            # Winding shift per unit Delta, resolved AT THIS POSITION.  It is not a
            # per-direction constant: the integer primitive cones from the origin, so it
            # is linear but not translation-covariant and the shift depends on y.  The
            # difference is an exact multiple of the quantum, which the winding weight
            # cannot see, so only the stored integer would reveal an error here.
            shift0 = 0; shift1 = 0; shift2 = 0; shift3 = 0
            for j in range(6):
                pc = cob_pc[mu, j]; sgn = cob_sign[mu, j]
                xp0 = (y0 + cob_off[mu, j, 0]) % N; xp1 = (y1 + cob_off[mu, j, 1]) % N
                xp2 = (y2 + cob_off[mu, j, 2]) % N; xp3 = (y3 + cob_off[mu, j, 3]) % N
                col = pc * V + ((xp0 * N + xp1) * N + xp2) * N + xp3
                shift0 += sgn * windingSensitivity[0, col]
                shift1 += sgn * windingSensitivity[1, col]
                shift2 += sgn * windingSensitivity[2, col]
                shift3 += sgn * windingSensitivity[3, col]
            windingBase = (log_winding_1d(winding[0], quantum, windingCoefficient)
                           + log_winding_1d(winding[1], quantum, windingCoefficient)
                           + log_winding_1d(winding[2], quantum, windingCoefficient)
                           + log_winding_1d(winding[3], quantum, windingCoefficient))
            center = int(round(-L / Kc))
            # Tolerance-driven window: the winding factor tilts the discrete Gaussian, so
            # a fixed half-width can truncate it.  Double until the edge weight is
            # negligible, exactly as the python reference does.
            H = Harr[mu]
            logw = np.empty(2 * H + 1)
            while True:
                nD = 2 * H + 1
                logw = np.empty(nD)
                best = -1.0e300
                for iD in range(nD):
                    D = center - H + iD
                    dQ = 0
                    for u in range(nu):
                        q0 = q[hh[u, 0], hh[u, 1], hh[u, 2], hh[u, 3]]
                        nv = q0 + D * vv[u]
                        dQ += (1 if nv != 0 else 0) - (1 if q0 != 0 else 0)
                    lw = (-twopi2k * (2.0 * D * L + D * D * Kc)
                          + (log_sector_weight(counts[1] + dQ, chargeLogWeight,
                                               chargeTailSlope, chargeHardWall)
                             - log_sector_weight(counts[1], chargeLogWeight,
                                                 chargeTailSlope, chargeHardWall))
                          + (log_winding_1d(winding[0] + D * shift0, quantum, windingCoefficient)
                             + log_winding_1d(winding[1] + D * shift1, quantum, windingCoefficient)
                             + log_winding_1d(winding[2] + D * shift2, quantum, windingCoefficient)
                             + log_winding_1d(winding[3] + D * shift3, quantum, windingCoefficient))
                          - windingBase
                          + cob_log_umbrella(D, nCharge, chargeList, qflat, hhFlat, vv,
                                              nu, N, pairUmbrellaLog, postList))
                    logw[iD] = lw
                    if lw > best:
                        best = lw
                edgeLow = np.exp(logw[0] - best)
                edgeHigh = np.exp(logw[nD - 1] - best)
                edge = edgeLow if edgeLow > edgeHigh else edgeHigh
                if edge < edgeTolerance or H > windowCap:
                    break
                H *= 2
            if H > windowCap:
                raise ValueError('coboundary window failed to reach tolerance within the cap')
            nD = 2 * H + 1
            ssum = 0.0
            for iD in range(nD):
                logw[iD] = np.exp(logw[iD] - best); ssum += logw[iD]
            r = np.random.random() * ssum; acc = 0.0; pick = 0
            for iD in range(nD):
                acc += logw[iD]
                if r <= acc:
                    pick = iD; break
            D = center - H + pick
            if D != 0:
                ctr[3] += 1
                dQ = 0
                nAffApplied = 0
                for u in range(nu):
                    q0 = q[hh[u, 0], hh[u, 1], hh[u, 2], hh[u, 3]]
                    nv = q0 + D * vv[u]
                    dQ += (1 if nv != 0 else 0) - (1 if q0 != 0 else 0)
                    q[hh[u, 0], hh[u, 1], hh[u, 2], hh[u, 3]] = nv
                    affIdxApplied[nAffApplied] = (((hh[u, 0] * N + hh[u, 1]) * N
                                                   + hh[u, 2]) * N + hh[u, 3])
                    nAffApplied += 1
                for j in range(6):
                    pc = cob_pc[mu, j]
                    xp0 = (y0 + cob_off[mu, j, 0]) % N; xp1 = (y1 + cob_off[mu, j, 1]) % N
                    xp2 = (y2 + cob_off[mu, j, 2]) % N; xp3 = (y3 + cob_off[mu, j, 3]) % N
                    F[pc, xp0, xp1, xp2, xp3] += D * cob_sign[mu, j]
                    periods[pc] += D * cob_sign[mu, j]
                    green_add(G[pc], g0, xp0, xp1, xp2, xp3, float(D * cob_sign[mu, j]), N)
                # Maintain the charged-cell list in step with q, by the same swap-with-last
                # trick openList uses.  If this drifts from q the umbrella silently weights
                # the wrong separation -- and would keep producing a plausible correlator.
                for kk in range(nAffApplied):
                    cell = affIdxApplied[kk]
                    pos = chargePos[cell]
                    if qflat[cell] != 0 and pos < 0:
                        chargeList[nCharge] = cell
                        chargePos[cell] = nCharge
                        nCharge += 1
                    elif qflat[cell] == 0 and pos >= 0:
                        last = chargeList[nCharge - 1]
                        chargeList[pos] = last
                        chargePos[last] = pos
                        chargePos[cell] = -1
                        nCharge -= 1
                counts[1] += dQ
                winding[0] += D * shift0; winding[1] += D * shift1
                winding[2] += D * shift2; winding[3] += D * shift3
        else:
            # ---------- plaquette ±1 ----------
            ctr[0] += 1
            if targetFraction > 0.0 and nOpen > 0 and np.random.random() < targetFraction:
                # Draw an open cell, then one of the 6 plaquettes incident on it.  The
                # forward stencil says toggling (c, x) moves cell (cc, x+off), so the
                # plaquette sits at x = a - off.
                ctr[4] += 1
                flat = openList[np.random.randint(nOpen)]
                cc = flat // V
                rem = flat - cc * V
                a3 = rem % N; rem = rem // N
                a2 = rem % N; rem = rem // N
                a1 = rem % N; rem = rem // N
                a0 = rem
                j = np.random.randint(6)
                c = revPc[cc, j]
                x0 = (a0 - revOff[cc, j, 0]) % N; x1 = (a1 - revOff[cc, j, 1]) % N
                x2 = (a2 - revOff[cc, j, 2]) % N; x3 = (a3 - revOff[cc, j, 3]) % N
            else:
                c = np.random.randint(6)
                x0 = np.random.randint(N); x1 = np.random.randint(N)
                x2 = np.random.randint(N); x3 = np.random.randint(N)
            s = 1 if np.random.random() < 0.5 else -1
            dC = 2.0 * s * G[c, x0, x1, x2, x3] + self_energy
            dD = 0
            openBefore = 0
            openAfter = 0
            for k in range(4):
                kcc = dsten_cc[c, k]
                a0 = (x0 + dsten_off[c, k, 0]) % N; a1 = (x1 + dsten_off[c, k, 1]) % N
                a2 = (x2 + dsten_off[c, k, 2]) % N; a3 = (x3 + dsten_off[c, k, 3]) % N
                old = dF[kcc, a0, a1, a2, a3]; new = old + s * dsten_sign[c, k]
                dD += (1 if new != 0 else 0) - (1 if old != 0 else 0)
                openBefore += 1 if old != 0 else 0
                openAfter += 1 if new != 0 else 0
                cellFlat[k] = kcc * V + ((a0 * N + a1) * N + a2) * N + a3
                cellNew[k] = new
            ng = w_ngroup[c]
            for gg in range(ng):
                gdq[gg] = 0
            for t in range(w_nterm[c]):
                p0 = (x0 + w_rel[c, t, 0]) % N; p1 = (x1 + w_rel[c, t, 1]) % N
                p2 = (x2 + w_rel[c, t, 2]) % N; p3 = (x3 + w_rel[c, t, 3]) % N
                gdq[w_gid[c, t]] += w_v[c, t] * F[w_pc[c, t], p0, p1, p2, p3]
            dQ = 0
            for gg in range(ng):
                dq = s * gdq[gg]
                if dq == 0:
                    continue
                h0 = (x0 + ghrel[c, gg, 0]) % N; h1 = (x1 + ghrel[c, gg, 1]) % N
                h2 = (x2 + ghrel[c, gg, 2]) % N; h3 = (x3 + ghrel[c, gg, 3]) % N
                q0 = q[h0, h1, h2, h3]; nv = q0 + dq
                dQ += (1 if nv != 0 else 0) - (1 if q0 != 0 else 0)
            col = c * V + ((x0 * N + x1) * N + x2) * N + x3
            dLogWinding = 0.0
            for mu2 in range(4):
                w0 = winding[mu2]
                w1 = w0 + s * windingSensitivity[mu2, col]
                dLogWinding += (log_winding_1d(w1, quantum, windingCoefficient)
                                - log_winding_1d(w0, quantum, windingCoefficient))
            dLogSector = (log_sector_weight(counts[0] + dD, sectorLogWeight, sectorTailSlope, sectorHardWall)
                          - log_sector_weight(counts[0], sectorLogWeight, sectorTailSlope, sectorHardWall))
            dLogProposal = (log_proposal_density_nb(openAfter, counts[0] + dD,
                                                    targetFraction, sites)
                            - log_proposal_density_nb(openBefore, counts[0],
                                                       targetFraction, sites))
            # Pair-separation umbrella.  Before: read the live charged list.  After: rebuild
            # the would-be charged set from the affected cells without mutating anything,
            # since the acceptance is decided before the move is applied.  Cells driven to
            # zero must be DROPPED, not carried as zeros -- a stale zero makes a two-defect
            # state look like three and silently switches the umbrella off for exactly the
            # configurations it exists to weight.
            nAff = 0
            for gg in range(ng):
                dq = s * gdq[gg]
                if dq == 0:
                    continue
                h0 = (x0 + ghrel[c, gg, 0]) % N; h1 = (x1 + ghrel[c, gg, 1]) % N
                h2 = (x2 + ghrel[c, gg, 2]) % N; h3 = (x3 + ghrel[c, gg, 3]) % N
                affIdx[nAff] = ((h0 * N + h1) * N + h2) * N + h3
                affNew[nAff] = qflat[affIdx[nAff]] + dq
                nAff += 1
            nPost = 0
            for jj in range(nCharge):
                cell = chargeList[jj]
                touched = False
                for kk in range(nAff):
                    if affIdx[kk] == cell:
                        touched = True
                        break
                if not touched:
                    postList[nPost] = cell
                    nPost += 1
            for kk in range(nAff):
                if affNew[kk] != 0:
                    postList[nPost] = affIdx[kk]
                    nPost += 1
            logUmbBefore = pair_log_umbrella(nCharge, chargeList, qflat, N, pairUmbrellaLog)
            logUmbAfter = 0.0
            if nPost == 2:
                v0 = qflat[postList[0]]
                for kk in range(nAff):
                    if affIdx[kk] == postList[0]:
                        v0 = affNew[kk]
                v1 = qflat[postList[1]]
                for kk in range(nAff):
                    if affIdx[kk] == postList[1]:
                        v1 = affNew[kk]
                if (v0 == 1 and v1 == -1) or (v0 == -1 and v1 == 1):
                    logUmbAfter = pairUmbrellaLog[sep2_flat(postList[0], postList[1], N)]
            dLogCharge = (log_sector_weight(counts[1] + dQ, chargeLogWeight,
                                           chargeTailSlope, chargeHardWall)
                          - log_sector_weight(counts[1], chargeLogWeight,
                                              chargeTailSlope, chargeHardWall))
            lnA = (-twopi2k * dC + dLogSector + dLogCharge + dLogWinding + dLogProposal
                   + logUmbAfter - logUmbBefore)
            if np.log(np.random.random()) < lnA:
                ctr[1] += 1
                for mu2 in range(4):
                    winding[mu2] += s * windingSensitivity[mu2, col]
                F[c, x0, x1, x2, x3] += s
                periods[c] += s
                for k in range(4):
                    kcc = dsten_cc[c, k]
                    a0 = (x0 + dsten_off[c, k, 0]) % N; a1 = (x1 + dsten_off[c, k, 1]) % N
                    a2 = (x2 + dsten_off[c, k, 2]) % N; a3 = (x3 + dsten_off[c, k, 3]) % N
                    dF[kcc, a0, a1, a2, a3] += s * dsten_sign[c, k]
                for gg in range(ng):
                    dq = s * gdq[gg]
                    if dq == 0:
                        continue
                    h0 = (x0 + ghrel[c, gg, 0]) % N; h1 = (x1 + ghrel[c, gg, 1]) % N
                    h2 = (x2 + ghrel[c, gg, 2]) % N; h3 = (x3 + ghrel[c, gg, 3]) % N
                    q[h0, h1, h2, h3] += dq
                green_add(G[c], g0, x0, x1, x2, x3, float(s), N)
                # Keep the occupancy list in step: swap-remove closures, append openings.
                for k in range(4):
                    flat = cellFlat[k]
                    pos = openPos[flat]
                    if cellNew[k] != 0 and pos < 0:
                        openList[nOpen] = flat
                        openPos[flat] = nOpen
                        nOpen += 1
                    elif cellNew[k] == 0 and pos >= 0:
                        last = openList[nOpen - 1]
                        openList[pos] = last
                        openPos[last] = pos
                        openPos[flat] = -1
                        nOpen -= 1
                nAffApplied = 0
                for kk in range(nAff):
                    affIdxApplied[nAffApplied] = affIdx[kk]
                    nAffApplied += 1
                # Maintain the charged-cell list in step with q, by the same swap-with-last
                # trick openList uses.  If this drifts from q the umbrella silently weights
                # the wrong separation -- and would keep producing a plausible correlator.
                for kk in range(nAffApplied):
                    cell = affIdxApplied[kk]
                    pos = chargePos[cell]
                    if qflat[cell] != 0 and pos < 0:
                        chargeList[nCharge] = cell
                        chargePos[cell] = nCharge
                        nCharge += 1
                    elif qflat[cell] == 0 and pos >= 0:
                        last = chargeList[nCharge - 1]
                        chargeList[pos] = last
                        chargePos[last] = pos
                        chargePos[cell] = -1
                        nCharge -= 1
                counts[0] += dD
                counts[1] += dQ
