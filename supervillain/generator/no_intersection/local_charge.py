#!/usr/bin/env python
r"""
Local, single-link topological-charge response for the No-Intersection model.

Flipping one link $n_\ell \to n_\ell + c$ changes the charge density
$q = dn\wedge dn$ only on the handful of hypercubes the link touches, and --- because
the single-link self-wedge $d\delta_\ell\wedge d\delta_\ell \equiv 0$ --- the change is
*exactly linear* in the background $F = dn$:

.. math::
    \Delta q = c\,\bigl(F\wedge d\delta_\ell + d\delta_\ell\wedge F\bigr).

That linear map $F \mapsto \Delta q$ collapses to a small fixed **stencil**: for each
link direction $\mu$, a list of terms $(o, p, s, k)$ meaning ``the output hypercube at
offset o gets k * F[p, link_site + s]``.  Evaluating the stencil is $O(1)$ per link, so a
constrained-link sweep drops from $O(N^8)$ (one global $O(N^4)$ ``charge`` recompute per
link) to $O(N^4)$.

The stencil is *derived once* from the library :func:`~supervillain.lattice.wedge` on a
small probe lattice, so its sign and shift conventions are inherited from ``wedge`` itself
and never hand-transcribed here.  The companion ``d\delta_\ell`` pattern (needed to keep
$F$ current after an accepted flip) is read straight off ``e = d(delta_link)`` in the same
pass.

Restricted to $D = 4$ (the No-Intersection action hard-assumes it).
"""

from itertools import product

import numpy as np

from supervillain.lattice import Lattice, d, wedge

# Lazily-derived, process-global caches (the maps are universal in D=4: the component
# ordering of lattice.components[p] is fixed, so one derivation serves every lattice).
_DQ_STENCIL = {}   # mu -> list of (out_offset, F_component_index, F_site_offset, coeff)
_DF_STENCIL = {}   # mu -> list of (F_component_index, F_site_offset, value)  == d(delta_link)


def _derive():
    r"""
    Derive the single-link charge-response stencil and the $d\delta$ pattern for each of
    the four link directions, by probing the library :func:`wedge` on a scratch lattice.

    For each direction $\mu$ and each candidate background entry $F[p, \text{anchor}+s]$ set
    to a lone $1$, we read off $F\wedge e + e\wedge F$ (with $e = d(\delta_{\mu,\text{anchor}})$)
    and record every nonzero output hypercube as a stencil term.  The single-link response
    reads $F$ only within Chebyshev distance $1$ (asserted in
    ``test_local_charge::test_stencil_is_local``), so the probe box $s \in [-1, 1]^4$ is
    sufficient; the correctness tests re-verify the assembled stencil against the global
    recompute, so an undersized box would fail loudly rather than silently.
    """
    N = 7
    L0 = Lattice(4, N)
    anchor = (3, 3, 3, 3)
    n_planes = len(L0.components[2])
    box = list(product(range(-1, 2), repeat=4))

    def signed(x):
        # Map a raw modular coordinate difference into (-N/2, N/2].
        return ((x + N // 2) % N) - N // 2

    dq_stencil = {}
    df_stencil = {}
    for mu in range(4):
        de = L0.zeros(1, dtype=int)
        de[(mu,) + anchor] = 1
        e = d(de)                                 # e = d(delta_link), a 2-form Form
        e_arr = np.asarray(e)

        # d(delta_link) pattern: exactly the nonzero entries of e, as offsets from the site.
        df_terms = []
        for h in np.argwhere(e_arr != 0):
            p = int(h[0])
            s = tuple(signed(int(h[1 + k]) - anchor[k]) for k in range(4))
            df_terms.append((p, s, int(e_arr[tuple(h)])))
        df_stencil[mu] = df_terms

        # Charge-response stencil: probe each F entry in turn.
        dq_terms = []
        for p in range(n_planes):
            for s in box:
                y = tuple((anchor[k] + s[k]) % N for k in range(4))
                F = L0.zeros(2, dtype=int)
                F[(p,) + y] = 1
                r = np.asarray(wedge(F, e)) + np.asarray(wedge(e, F))   # 4-form, (1,)+dims
                for h in np.argwhere(r != 0):
                    o = tuple(signed(int(h[1 + k]) - anchor[k]) for k in range(4))
                    dq_terms.append((o, p, tuple(s), int(r[tuple(h)])))
        dq_stencil[mu] = dq_terms

    return dq_stencil, df_stencil


def _stencils():
    global _DQ_STENCIL, _DF_STENCIL
    if not _DQ_STENCIL:
        _DQ_STENCIL, _DF_STENCIL = _derive()
    return _DQ_STENCIL, _DF_STENCIL


def charge_change_from_link(F, mu, site, c, N):
    r"""
    The change in the charge density $q = dn\wedge dn$ from $n_{(\mu,\text{site})} \to
    n + c$, computed locally from the current field strength ``F`` = $d n$.

    Parameters
    ----------
    F : np.ndarray
        The current field strength $dn$, a raw 2-form array of shape ``(C(D,2),) + dims``.
    mu : int
        The link direction (index into ``lattice.components[1]``, i.e. just $\mu$).
    site : tuple of int
        The link's site.
    c : int
        The proposed shift $\Delta n_\ell$.
    N : int
        The lattice extent (for periodic wrap).

    Returns
    -------
    dict
        ``{hypercube_site: Delta q}`` on the $O(1)$ affected hypercubes, with zeros
        pruned.  An **empty** dict means the flip is *clean* ($\Delta q \equiv 0$): the
        constraint $q = 0$ is preserved.
    """
    dq_stencil, _ = _stencils()
    out = {}
    for o, p, s, k in dq_stencil[mu]:
        val = k * c * int(F[(p,) + tuple((site[j] + s[j]) % N for j in range(4))])
        if val:
            cell = tuple((site[j] + o[j]) % N for j in range(4))
            out[cell] = out.get(cell, 0) + val
    return {cell: v for cell, v in out.items() if v != 0}


def apply_link_to_F(F, mu, site, c, N):
    r"""
    Update the field strength ``F`` = $dn$ in place for an accepted flip
    $n_{(\mu,\text{site})} \to n + c$, touching only the $\sim 2(D-1)$ plaquettes the link
    borders.  ``F`` stays exactly ``d(n)`` without any global recompute.
    """
    _, df_stencil = _stencils()
    for p, s, val in df_stencil[mu]:
        F[(p,) + tuple((site[j] + s[j]) % N for j in range(4))] += c * val


# ============================================================================
# Vectorized, checkerboarded machinery for the whole-sweep update (M2).
# ============================================================================
#
# Two single-link flips interact --- through the bilinear cross term
# d(delta_i) ^ d(delta_j) + d(delta_j) ^ d(delta_i), the ONLY coupling between links,
# since each link's Delta S depends on its own n alone --- iff their sites lie within
# Chebyshev distance 1 of each other.  (This single-link **interaction reach = 1** is
# verified in test_local_charge::test_single_link_interaction_reach_is_one.)  So a set of
# same-direction links is mutually non-interacting as soon as every pair differs by >= 2
# in at least one axis, and such a set may be updated *simultaneously*: each link's clean
# check and Delta S read a background the others do not touch, so a simultaneous colour
# update is identical dynamics to visiting those links sequentially in any order (the
# standard checkerboard argument).
#
# "Differ by >= 2 in at least one axis" is achieved by colouring each axis's periodic
# chain so that same-colour coordinates are never adjacent -- a proper colouring of the
# cycle C_N: 2 colours when N is even, 3 when N is odd (the odd cycle's seam needs a
# third).  The full link colour is the tuple of per-axis colours; two links of one colour
# share every axis-colour, so they differ by 0-or->=2 in every axis, hence by >= 2 in some
# axis unless they are the same link.  This works for EVERY N, even or odd, with no
# divisibility requirement.


def axis_colors(N):
    r"""
    A proper colouring of the periodic chain of ``N`` sites: a list of index arrays whose
    union is ``range(N)`` and within which no two entries are adjacent (differ by 1) on the
    ring.  Two colours for even ``N`` (parity), three for odd ``N`` (the last site takes a
    third colour so the wrap-around seam stays properly coloured).
    """
    if N % 2 == 0:
        return [np.arange(0, N, 2), np.arange(1, N, 2)]
    return [np.arange(0, N - 1, 2), np.arange(1, N - 1, 2), np.array([N - 1])]


def _dq_grouped():
    r"""The DQ stencil regrouped by output offset ``o`` (for the vectorized clean check)."""
    global _DQ_GROUPED
    if _DQ_GROUPED is None:
        dq, _ = _stencils()
        grouped = {}
        for mu in range(4):
            gm = {}
            for o, p, s, k in dq[mu]:
                gm.setdefault(o, []).append((p, s, k))
            grouped[mu] = gm
        _DQ_GROUPED = grouped
    return _DQ_GROUPED


_DQ_GROUPED = None


def clean_mask_for_color(F, mu, idx, N):
    r"""
    For a whole colour --- direction ``mu`` and the link sites on the grid
    ``idx = [rows_0, rows_1, rows_2, rows_3]`` (one index array per axis, their Cartesian
    product) --- the boolean mask of which links are *clean* (flipping them preserves
    $q = 0$), computed in one shot by applying the charge-response stencil as shifted gathers
    of ``F``.  Cleanliness is independent of the shift magnitude, so ``c`` does not enter.

    ``F`` is the raw field-strength array $dn$; ``N`` the lattice extent.  Returns a boolean
    array shaped like the colour grid ``(len(idx_0), ..., len(idx_3))``.
    """
    block = tuple(len(i) for i in idx)
    clean = np.ones(block, dtype=bool)
    for terms in _dq_grouped()[mu].values():
        R = np.zeros(block, dtype=np.int64)
        for p, s, k in terms:
            rd = tuple((idx[a] + s[a]) % N for a in range(4))
            R += k * F[p][np.ix_(*rd)].astype(np.int64)
        clean &= (R == 0)
    return clean


def apply_color(n, F, mu, idx, N, flip):
    r"""
    Apply an accepted colour update in place: add ``flip`` (the per-link shift, zero where
    rejected) to $n$ on direction ``mu`` at the colour grid ``idx``, and patch $F = dn$ on
    the bordered plaquettes.  Because the colour is non-interacting the writes never alias
    within a stencil term, so the vectorized scatter reproduces the sequential updates.
    """
    n[mu][np.ix_(*idx)] += flip
    _, df_stencil = _stencils()
    for p, s, val in df_stencil[mu]:
        wr = tuple((idx[a] + s[a]) % N for a in range(4))
        F[p][np.ix_(*wr)] += val * flip


# ---------------------------------------------------------------------------
# numba colour kernel (M2.1): the whole clean-check + apply of one colour in
# compiled code, so the sweep pays no per-term Python/numpy dispatch at all.
# ---------------------------------------------------------------------------
#
# The broadcast sweep's cost was entirely per-colour, per-stencil-term Python/numpy
# dispatch (tens of thousands of ``np.ix_`` and fancy-index calls per sweep).  Here the
# whole integer clean-check and apply of a colour run in one compiled kernel over
# precomputed integer index data, so that dispatch is gone.
#
# The RNG draws and the float Delta-S / exp / accept stay in numpy (byte-identical to the
# broadcast sweep --- numba is not asked to reproduce numpy's vectorised exp); the kernel
# does only the INTEGER charge-response stencil and the integer flip/patch, which are exact.
# The per-colour metropolis pass ``metro`` and shift ``c`` are handed in already drawn and
# raveled in the colour's C-order, matching ``coords``.

from numba import njit  # noqa: E402  (kept next to the kernel it enables)


@njit(cache=True)
def _color_kernel(F2, n_mu, metro, prob, c, coords,
                  read_off, plane, coeff, group_ptr, df_off, df_plane, df_val, N):
    r"""
    Update one colour in place and return ``(naccept, nclean, accprob)``: the number of
    accepted flips, the number of constraint-preserving (clean) links, and the summed
    Metropolis acceptance probability ``prob`` over those clean links --- the ingredient of
    the *expected* acceptance diagnostic, which should track the true accepted fraction.

    ``F2`` is $F = dn$ reshaped to ``(n_planes, N**4)``; ``n_mu`` is $n$ on this colour's
    direction reshaped to ``(N**4,)``; ``coords`` are the colour's link sites ``(B, 4)`` in
    the same C-order as ``metro`` / ``prob`` / ``c``.  A link is flipped iff its charge
    response is clean on every output group AND its precomputed metropolis coin passed.
    Offsets are made non-negative before the modulo so the wrap is correct regardless of the
    sign convention of integer ``%`` in nopython mode.
    """
    B = coords.shape[0]
    ngroups = group_ptr.shape[0] - 1
    Td = df_plane.shape[0]
    naccept = 0
    nclean = 0
    accprob = 0.0
    for j in range(B):
        x0 = coords[j, 0]; x1 = coords[j, 1]; x2 = coords[j, 2]; x3 = coords[j, 3]
        ok = True
        for g in range(ngroups):
            R = 0
            for t in range(group_ptr[g], group_ptr[g + 1]):
                r0 = (x0 + read_off[t, 0] + N) % N
                r1 = (x1 + read_off[t, 1] + N) % N
                r2 = (x2 + read_off[t, 2] + N) % N
                r3 = (x3 + read_off[t, 3] + N) % N
                R += coeff[t] * F2[plane[t], ((r0 * N + r1) * N + r2) * N + r3]
            if R != 0:
                ok = False
                break
        if ok:
            nclean += 1
            accprob += prob[j]           # only clean links are Metropolis-tested
            if metro[j]:
                cc = c[j]
                n_mu[((x0 * N + x1) * N + x2) * N + x3] += cc
                for t in range(Td):
                    w0 = (x0 + df_off[t, 0] + N) % N
                    w1 = (x1 + df_off[t, 1] + N) % N
                    w2 = (x2 + df_off[t, 2] + N) % N
                    w3 = (x3 + df_off[t, 3] + N) % N
                    F2[df_plane[t], ((w0 * N + w1) * N + w2) * N + w3] += df_val[t] * cc
                naccept += 1
    return naccept, nclean, accprob


_NUMBA_STENCILS = {}   # mu -> (read_off, plane, coeff, group_ptr, df_off, df_plane, df_val)
_NUMBA_PLANS = {}      # N  -> list of (mu, base_grid, block_shape, coords)


def _numba_stencil(mu):
    r"""Flat integer arrays describing direction ``mu``'s stencil for :func:`_color_kernel`."""
    if mu in _NUMBA_STENCILS:
        return _NUMBA_STENCILS[mu]
    dq_grouped = _dq_grouped()[mu]
    _, df_stencil = _stencils()
    read_off, plane, coeff, group_ptr = [], [], [], [0]
    for terms in dq_grouped.values():
        for p, s, k in terms:
            plane.append(p); read_off.append(s); coeff.append(k)
        group_ptr.append(len(plane))
    df_off = [s for _p, s, _v in df_stencil[mu]]
    df_plane = [_p for _p, _s, _v in df_stencil[mu]]
    df_val = [_v for _p, _s, _v in df_stencil[mu]]
    out = (np.array(read_off, dtype=np.int64).reshape(-1, 4),
           np.array(plane, dtype=np.int64),
           np.array(coeff, dtype=np.int64),
           np.array(group_ptr, dtype=np.int64),
           np.array(df_off, dtype=np.int64).reshape(-1, 4),
           np.array(df_plane, dtype=np.int64),
           np.array(df_val, dtype=np.int64))
    _NUMBA_STENCILS[mu] = out
    return out


def numba_plan(N):
    r"""
    Per-colour plan for the numba sweep (cached per ``N``): a list, in the canonical colour
    order, of ``(mu, base_grid, block_shape, coords)``.  ``base_grid`` is the ``np.ix_`` tuple
    used to gather the colour's $d\phi$ and $n$ (for the numpy $\Delta S$); ``coords`` are the
    colour's link sites ``(B, 4)`` int64 in C-order, matching a raveled block.
    """
    if N in _NUMBA_PLANS:
        return _NUMBA_PLANS[N]
    axis = axis_colors(N)
    plan = []
    for mu in range(4):
        for choice in product(range(len(axis)), repeat=4):
            idx = [axis[a] for a in choice]
            base = np.ix_(*idx)
            block = tuple(len(i) for i in idx)
            coords = np.array(list(product(*[[int(x) for x in i] for i in idx])),
                              dtype=np.int64).reshape(-1, 4)
            plan.append((mu, base, block, coords))
    _NUMBA_PLANS[N] = plan
    return plan
