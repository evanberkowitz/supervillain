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

Layout: tables and builders (this file, Task 3) first, so the batch ``njit``
kernel appended later (Task 7's :func:`gas_batch` and its helpers) has a natural
place to land after them.

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
