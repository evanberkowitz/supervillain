#!/usr/bin/env python
r"""
Compiled clean-set evaluation for :class:`~.TwoLinkAdaptiveWorm`.

The two-link worm's cost is its ``clean_set`` enumeration: for a drawn direction it must
test every shape in a fixed ~$10^{4}$-shape family against the current field strength
$F = dn$.  In pure Python that is ~15k dict-building :func:`_local_dq` calls per proposal.
Here the whole family is flattened to integer arrays once, and a single ``njit`` kernel
evaluates every shape's charge change $\Delta q$ against $F$ and returns a boolean *clean*
mask.  The (few) clean shapes are then turned into change dicts in Python, so the compiled
path reproduces the pure-Python ``clean_set`` **bit-for-bit** (same clean shapes, same
order, same dedup).

For a shape placed at ``anchor`` with per-link coefficient ``factor * c`` (``factor`` is
$+1$ for a forward move, $-1$ for the negated reverse), the charge change is

.. math::
    \Delta q = \text{factor}\sum_{\ell} c_{\ell}\, L_{\ell}(F) + \Delta n \wedge \Delta n,

the first term the per-link background-linear stencil response (the same
``dq_stencil`` :mod:`~supervillain.generator.no_intersection.local_charge` derives) and the
second the precomputed self-charge (sign-independent, quadratic in the coefficients).  A
shape is a clean **mover** iff $\Delta q$ is exactly the dipole
$\{\text{head}: -1, \text{target}: +1\}$, a clean **idle** iff $\Delta q \equiv 0$.

Restricted to $D = 4$.
"""

import numpy as np
from numba import njit

from supervillain.generator.no_intersection import local_charge


def dq_stencil_arrays():
    r"""
    The single-link charge-response stencil (``local_charge`` ``dq_stencil``) flattened for
    the kernel: for each direction $\mu$ a list of terms $(o, p, s, k)$ meaning the output
    hypercube at offset ``o`` gets ``k * c * F[p, site + s]``.  Returned as
    ``(out_off, plane, read_off, coeff, mu_ptr)`` with ``mu_ptr`` the per-$\mu$ CSR bounds.
    """
    dq, _ = local_charge._stencils()
    out_off, plane, read_off, coeff, mu_ptr = [], [], [], [], [0]
    for mu in range(4):
        for o, p, s, k in dq[mu]:
            out_off.append(o)
            plane.append(p)
            read_off.append(s)
            coeff.append(k)
        mu_ptr.append(len(plane))
    return (np.array(out_off, dtype=np.int64).reshape(-1, 4),
            np.array(plane, dtype=np.int64),
            np.array(read_off, dtype=np.int64).reshape(-1, 4),
            np.array(coeff, dtype=np.int64),
            np.array(mu_ptr, dtype=np.int64))


def flatten_family(shapes, self_charge):
    r"""
    Flatten a shape family (a list of shapes, each a tuple of ``(mu, rel_site, coeff)``
    triples) and its registered self-charges to CSR integer arrays for :func:`clean_mask`.
    ``self_charge`` maps each shape to ``((offset, value), ...)``.  Shape order is preserved,
    so the kernel's clean mask indexes ``shapes`` directly.
    """
    link_ptr, link_mu, link_r, link_c = [0], [], [], []
    sc_ptr, sc_off, sc_val = [0], [], []
    for shape in shapes:
        for mu, r, c in shape:
            link_mu.append(mu)
            link_r.append(r)
            link_c.append(c)
        link_ptr.append(len(link_mu))
        for off, val in self_charge[shape]:
            sc_off.append(off)
            sc_val.append(val)
        sc_ptr.append(len(sc_val))
    return (np.array(link_ptr, dtype=np.int64),
            np.array(link_mu, dtype=np.int64),
            np.array(link_r, dtype=np.int64).reshape(-1, 4) if link_r
            else np.zeros((0, 4), dtype=np.int64),
            np.array(link_c, dtype=np.int64),
            np.array(sc_ptr, dtype=np.int64),
            np.array(sc_off, dtype=np.int64).reshape(-1, 4) if sc_off
            else np.zeros((0, 4), dtype=np.int64),
            np.array(sc_val, dtype=np.int64))


@njit(cache=True)
def clean_mask(F2, N, factor, anchor, head_rav, target_rav, mode,
               link_ptr, link_mu, link_r, link_c,
               sc_ptr, sc_off, sc_val,
               st_o, st_p, st_s, st_k, st_ptr):
    r"""
    Boolean *clean* mask over a flattened shape family evaluated against ``F2`` (=$F = dn$
    reshaped to ``(n_planes, N**4)``).  ``mode == 1`` checks the mover dipole
    ``{head: -1, target: +1}``; ``mode == 0`` checks an idle (``Delta q`` identically zero).
    ``factor`` is applied to every link coefficient (self-charge is sign-independent).
    Offsets are made non-negative before the modulo so the wrap is correct in nopython mode.
    """
    nshapes = link_ptr.shape[0] - 1
    clean = np.zeros(nshapes, dtype=np.bool_)
    # Scratch for one shape's dq.  A shape has <= 4 links (the same4 library template) and
    # each link's stencil touches 8 output cells, so <= 32 link cells plus a small
    # self-charge support -- well under 45.  cap = 256 is a ~6x margin; a shape touching
    # more than cap distinct cells would silently drop terms, so keep cap above the family's
    # true maximum if larger templates are ever added.
    cap = 256
    cells = np.empty(cap, dtype=np.int64)
    vals = np.empty(cap, dtype=np.int64)
    for si in range(nshapes):
        nc = 0
        # Background-linear response of each link, via the single-link dq stencil.
        for li in range(link_ptr[si], link_ptr[si + 1]):
            mu = link_mu[li]
            c = factor * link_c[li]
            s0 = (anchor[0] + link_r[li, 0] + N) % N
            s1 = (anchor[1] + link_r[li, 1] + N) % N
            s2 = (anchor[2] + link_r[li, 2] + N) % N
            s3 = (anchor[3] + link_r[li, 3] + N) % N
            for ti in range(st_ptr[mu], st_ptr[mu + 1]):
                r0 = (s0 + st_s[ti, 0] + N) % N
                r1 = (s1 + st_s[ti, 1] + N) % N
                r2 = (s2 + st_s[ti, 2] + N) % N
                r3 = (s3 + st_s[ti, 3] + N) % N
                fval = F2[st_p[ti], ((r0 * N + r1) * N + r2) * N + r3]
                if fval == 0:
                    continue
                val = st_k[ti] * c * fval
                o0 = (s0 + st_o[ti, 0] + N) % N
                o1 = (s1 + st_o[ti, 1] + N) % N
                o2 = (s2 + st_o[ti, 2] + N) % N
                o3 = (s3 + st_o[ti, 3] + N) % N
                cell = ((o0 * N + o1) * N + o2) * N + o3
                found = False
                for b in range(nc):
                    if cells[b] == cell:
                        vals[b] += val
                        found = True
                        break
                if not found and nc < cap:
                    cells[nc] = cell
                    vals[nc] = val
                    nc += 1
        # Precomputed self-charge (dn wedge dn), sign-independent.
        for ci in range(sc_ptr[si], sc_ptr[si + 1]):
            o0 = (anchor[0] + sc_off[ci, 0] + N) % N
            o1 = (anchor[1] + sc_off[ci, 1] + N) % N
            o2 = (anchor[2] + sc_off[ci, 2] + N) % N
            o3 = (anchor[3] + sc_off[ci, 3] + N) % N
            cell = ((o0 * N + o1) * N + o2) * N + o3
            val = sc_val[ci]
            found = False
            for b in range(nc):
                if cells[b] == cell:
                    vals[b] += val
                    found = True
                    break
            if not found and nc < cap:
                cells[nc] = cell
                vals[nc] = val
                nc += 1
        # Classify against the target charge pattern.
        if mode == 0:
            good = True
            for b in range(nc):
                if vals[b] != 0:
                    good = False
                    break
            clean[si] = good
        else:
            ok_head = False
            ok_target = False
            bad = False
            for b in range(nc):
                if vals[b] == 0:
                    continue
                if cells[b] == head_rav and vals[b] == -1:
                    ok_head = True
                elif cells[b] == target_rav and vals[b] == 1:
                    ok_target = True
                else:
                    bad = True
                    break
            clean[si] = ok_head and ok_target and not bad
    return clean


@njit(cache=True)
def classify_mask(F2, N, anchor, head_rav,
                  link_ptr, link_mu, link_r, link_c,
                  sc_ptr, sc_off, sc_val,
                  st_o, st_p, st_s, st_k, st_ptr):
    r"""
    Per-shape classification over a flattened family evaluated against ``F2``
    ($F = dn$ reshaped to ``(n_planes, N**4)``), for the free-target worm: ``-2`` if the
    shape's $\Delta q$ is neither, ``-1`` for an idle ($\Delta q \equiv 0$), else the
    raveled cell of the ``+1`` defect of the clean dipole rooted at ``head_rav``.
    Shapes carry their own signed coefficients (no ``factor``) and anchor at the head.
    """
    nshapes = link_ptr.shape[0] - 1
    status = np.empty(nshapes, dtype=np.int64)
    cap = 256
    cells = np.empty(cap, dtype=np.int64)
    vals = np.empty(cap, dtype=np.int64)
    for si in range(nshapes):
        nc = 0
        for li in range(link_ptr[si], link_ptr[si + 1]):
            mu = link_mu[li]
            c = link_c[li]
            s0 = (anchor[0] + link_r[li, 0] + N) % N
            s1 = (anchor[1] + link_r[li, 1] + N) % N
            s2 = (anchor[2] + link_r[li, 2] + N) % N
            s3 = (anchor[3] + link_r[li, 3] + N) % N
            for ti in range(st_ptr[mu], st_ptr[mu + 1]):
                r0 = (s0 + st_s[ti, 0] + N) % N
                r1 = (s1 + st_s[ti, 1] + N) % N
                r2 = (s2 + st_s[ti, 2] + N) % N
                r3 = (s3 + st_s[ti, 3] + N) % N
                fval = F2[st_p[ti], ((r0 * N + r1) * N + r2) * N + r3]
                if fval == 0:
                    continue
                val = st_k[ti] * c * fval
                o0 = (s0 + st_o[ti, 0] + N) % N
                o1 = (s1 + st_o[ti, 1] + N) % N
                o2 = (s2 + st_o[ti, 2] + N) % N
                o3 = (s3 + st_o[ti, 3] + N) % N
                cell = ((o0 * N + o1) * N + o2) * N + o3
                found = False
                for b in range(nc):
                    if cells[b] == cell:
                        vals[b] += val
                        found = True
                        break
                if not found and nc < cap:
                    cells[nc] = cell
                    vals[nc] = val
                    nc += 1
        for ci in range(sc_ptr[si], sc_ptr[si + 1]):
            o0 = (anchor[0] + sc_off[ci, 0] + N) % N
            o1 = (anchor[1] + sc_off[ci, 1] + N) % N
            o2 = (anchor[2] + sc_off[ci, 2] + N) % N
            o3 = (anchor[3] + sc_off[ci, 3] + N) % N
            cell = ((o0 * N + o1) * N + o2) * N + o3
            val = sc_val[ci]
            found = False
            for b in range(nc):
                if cells[b] == cell:
                    vals[b] += val
                    found = True
                    break
            if not found and nc < cap:
                cells[nc] = cell
                vals[nc] = val
                nc += 1
        # Classify: count the nonzero cells and pattern-match the dipole.
        minus_ok = False
        plus_cell = np.int64(-1)
        bad = False
        nz = 0
        for b in range(nc):
            if vals[b] == 0:
                continue
            nz += 1
            if cells[b] == head_rav and vals[b] == -1:
                minus_ok = True
            elif vals[b] == 1 and plus_cell < 0:
                plus_cell = cells[b]
            else:
                bad = True
        if nz == 0:
            status[si] = -1
        elif nz == 2 and minus_ok and plus_cell >= 0 and not bad:
            status[si] = plus_cell
        else:
            status[si] = -2
    return status
