#!/usr/bin/env python
r"""
Compiled whole-batch tick loop for :class:`~.DefectGas`.

The defect gas's cost is pure per-proposal Python dispatch: each tick builds a small
$\Delta q$ dict from the single-link stencil, prices it against a sparse defect dict, and
tallies a sector --- a few dozen integer operations buried under interpreter overhead.
Here the whole pre-drawn proposal batch runs in one ``njit`` kernel over flat integer
state: the charge density $q$ kept as a **dense** array plus a compact list of its
nonzero cells (at most ``max_defects`` of them on a tuned chain, updated incrementally --- an
accepted flip touches $O(1)$ hypercubes), so $\Delta D$, the accept/reject, the $F$
patch, and the sector classification are all $O(1)$ per proposal with no Python in the
loop.

The kernel consumes the *same* pre-drawn proposal arrays (link, shift, uniform, plus
``comps``/``ucells``/``uedges`` -- the mixture-component, target-cell, and edge draws
the defect-adjacent mixture proposal needs) in the same order as the pure-Python
:meth:`~.DefectGas.step_reference` tick loop and applies the same floating-point accept
test, so the compiled chain reproduces the reference chain **bit-for-bit** (asserted in
``test_defect_gas_kernel.py``).

Restricted to $D = 4$.
"""

import numpy as np
from numba import njit

from supervillain.generator.no_intersection import local_charge
from supervillain.generator.no_intersection.two_link_kernel import dq_stencil_arrays

_PACK = None


def stencil_pack():
    r"""
    The flat integer stencils :func:`tick_batch` needs, cached process-wide: the
    single-link charge-response CSR ``(st_o, st_p, st_s, st_k, st_ptr)`` (from
    :func:`~.two_link_kernel.dq_stencil_arrays`) and the $d\delta_\ell$ field-strength
    patch regrouped per direction as ``(df_off, df_plane, df_val, df_ptr)``.
    """
    global _PACK
    if _PACK is None:
        st_o, st_p, st_s, st_k, st_ptr = dq_stencil_arrays()
        _, df = local_charge._stencils()
        df_off, df_plane, df_val, df_ptr = [], [], [], [0]
        for mu in range(4):
            for p, s, v in df[mu]:
                df_plane.append(p)
                df_off.append(s)
                df_val.append(v)
            df_ptr.append(len(df_val))
        _PACK = (st_o, st_p, st_s, st_k, st_ptr,
                 np.array(df_off, dtype=np.int64).reshape(-1, 4),
                 np.array(df_plane, dtype=np.int64),
                 np.array(df_val, dtype=np.int64),
                 np.array(df_ptr, dtype=np.int64))
    return _PACK


@njit(cache=True)
def _pair_rsq(a, b, N):
    # Min-image separation squared between two raveled cells.
    r = 0
    for _ in range(4):
        da = a % N
        db = b % N
        a //= N
        b //= N
        d = da - db
        if d < 0:
            d = -d
        if N - d < d:
            d = N - d
        r += d * d
    return r


@njit(cache=True)
def _cell_edge(cell, e, N):
    # Edge e in [0, 32) of the hypercube at raveled corner `cell`: direction
    # mu = e // 8; bits of b = e % 8 offset the three non-mu coordinates by +1.
    c3 = cell % N
    r = cell // N
    c2 = r % N
    r //= N
    c1 = r % N
    c0 = r // N
    mu = e // 8
    b = e % 8
    j = 0
    for nu in range(4):
        if nu == mu:
            continue
        if (b >> j) & 1:
            if nu == 0:
                c0 = (c0 + 1) % N
            elif nu == 1:
                c1 = (c1 + 1) % N
            elif nu == 2:
                c2 = (c2 + 1) % N
            else:
                c3 = (c3 + 1) % N
        j += 1
    return mu, c0, c1, c2, c3


@njit(cache=True)
def _link_charge_sum(mu, x0, x1, x2, x3, q, N):
    # s(l, q): sum of |q| over the 8 hypercubes containing link (mu, x) --- the
    # corners x - b over the three non-mu directions.
    s = 0
    for b in range(8):
        c0 = x0
        c1 = x1
        c2 = x2
        c3 = x3
        j = 0
        for nu in range(4):
            if nu == mu:
                continue
            if (b >> j) & 1:
                if nu == 0:
                    c0 = (c0 - 1 + N) % N
                elif nu == 1:
                    c1 = (c1 - 1 + N) % N
                elif nu == 2:
                    c2 = (c2 - 1 + N) % N
                else:
                    c3 = (c3 - 1 + N) % N
            j += 1
        qq = q[((c0 * N + c1) * N + c2) * N + c3]
        s += qq if qq >= 0 else -qq
    return s


@njit(cache=True)
def _link_charge_sum_delta(mu, x0, x1, x2, x3, q, cells, vals, nc, N):
    # s(l, q + Delta q): as _link_charge_sum but with the proposal's touched-cell
    # deltas (cells[:nc], vals[:nc]) applied on the fly.
    s = 0
    for b in range(8):
        c0 = x0
        c1 = x1
        c2 = x2
        c3 = x3
        j = 0
        for nu in range(4):
            if nu == mu:
                continue
            if (b >> j) & 1:
                if nu == 0:
                    c0 = (c0 - 1 + N) % N
                elif nu == 1:
                    c1 = (c1 - 1 + N) % N
                elif nu == 2:
                    c2 = (c2 - 1 + N) % N
                else:
                    c3 = (c3 - 1 + N) % N
            j += 1
        cell = ((c0 * N + c1) * N + c2) * N + c3
        qq = q[cell]
        for t in range(nc):
            if cells[t] == cell:
                qq += vals[t]
                break
        s += qq if qq >= 0 else -qq
    return s


@njit(cache=True)
def _geo_weight(cells, charges, count, w2, rsq_shell, N):
    # W of a defect multiset: w2 at a two-cell +-1 pair, the averaged Wick sum
    # (half the sum of the two contraction products, so a trivial all-ones
    # table is neutral there too) at four unit cells, and 1.0 for everything
    # else (or with the umbrella off).
    if w2.size == 0:
        return 1.0
    if count == 2:
        if charges[0] * charges[1] == -1:
            return w2[rsq_shell[_pair_rsq(cells[0], cells[1], N)]]
        return 1.0
    if count == 4:
        pos = np.empty(2, dtype=np.int64)
        neg = np.empty(2, dtype=np.int64)
        npos = 0
        nneg = 0
        for i in range(4):
            c = charges[i]
            if c == 1:
                if npos == 2:
                    return 1.0
                pos[npos] = cells[i]
                npos += 1
            elif c == -1:
                if nneg == 2:
                    return 1.0
                neg[nneg] = cells[i]
                nneg += 1
            else:
                return 1.0
        return 0.5 * (w2[rsq_shell[_pair_rsq(pos[0], neg[0], N)]]
                      * w2[rsq_shell[_pair_rsq(pos[1], neg[1], N)]]
                      + w2[rsq_shell[_pair_rsq(pos[0], neg[1], N)]]
                      * w2[rsq_shell[_pair_rsq(pos[1], neg[0], N)]])
    return 1.0


@njit(cache=True)
def tick_batch(F2, n2, dphi2, q, nzc, D, nnz,
               mus, sites, cs, us, comps, ucells, uedges, i0,
               kappa, fugacity, w, w2, rsq_shell, gamma, max_defects, N,
               st_o, st_p, st_s, st_k, st_ptr,
               df_off, df_plane, df_val, df_ptr,
               H_pair, H_four, tally, vac_stop,
               tstate, exc_hist, t_sector):
    r"""
    Run the pre-drawn proposal batch from index ``i0`` to its end --- or until
    ``vac_stop`` vacuum ticks have been seen, if ``vac_stop > 0`` --- mutating the chain
    state in place and tallying the sector dwell.

    State (mutated in place): ``F2`` is $F = dn$ as ``(n_planes, N**4)``; ``n2`` is $n$
    as ``(4, N**4)``; ``q`` is the dense charge density ``(N**4,)``; ``nzc[:nnz]`` are
    the raveled cells where ``q`` is nonzero (order arbitrary); ``dphi2`` is $d\phi$ as
    ``(4, N**4)``, read only.  ``D`` and ``nnz`` ride in and out as return values since
    scalars cannot be mutated.  The proposal arrays ``mus, sites, cs, us`` and the
    accept test --- ``u < e^{-\Delta S} \zeta^{\Delta D}``, or with a nonempty sector
    table ``w`` instead ``u < e^{-\Delta S}\, w[(D+\Delta D)/2]/w[D/2]`` --- match
    :meth:`~.DefectGas.step_reference` exactly.  With a gamma table the accept test
    carries the full Metropolis--Hastings factor p(l|n')/p(l|n) for the mixture
    proposal (uniform with probability gamma_k, else charge-weighted defect-adjacent).

    Tallies: each tick lands in a sector --- vacuum ($D = 0$, counted toward the return
    value and ``vac_stop``), the single-pair sector (``H_pair`` at the raveled $+$-to-$-$
    displacement), one of the four $D = 4$ classes (``H_four``; the sorted-charge
    classes $\{+1,+1,-1,-1\}$, $\{+2,-1,-1\}$, $\{+1,+1,-2\}$, $\{+2,-2\}$ are keyed
    here by ``nnz`` and the sign of the doubled charge, equivalent by neutrality) ---
    or is scaffolding.  Pair and four tallies are recorded only when ``tally``.

    Transport instrumentation (mutated in place, always on): ``tstate`` is
    ``[current excursion length in ticks, completed excursion count, max single-pair
    min-image separation squared, touched-top-sector flag, round trips]`` (a round
    trip is a vacuum return after visiting ``D == max_defects`` since the previous vacuum
    tick); ``exc_hist[b]`` counts completed excursions whose
    length had bit-length $b$ (power-of-two bins, top bin saturating).  An *excursion*
    is a maximal stretch of nonvacuum ticks; a zero-count far bin plus a small
    ``tstate[2]`` shows the run was transport-censored there, not that the dwell is
    zero.

    Returns
    -------
    (i, D, nnz, accepted, vacuum)
        ``i`` the index after the last consumed proposal; ``vacuum`` the vacuum ticks
        seen this call.
    """
    n_props = mus.shape[0]
    acc = 0
    vac = 0
    # Scratch for one proposal's distinct touched hypercubes (a single-link stencil
    # reaches at most 8; 64 is a wide margin).
    cells = np.empty(64, dtype=np.int64)
    vals = np.empty(64, dtype=np.int64)
    i = i0
    while i < n_props:
        mu = mus[i]
        x0 = sites[i, 0]; x1 = sites[i, 1]; x2 = sites[i, 2]; x3 = sites[i, 3]
        c = cs[i]
        if gamma.size > 0 and D > 0:
            if comps[i] >= gamma[D // 2]:
                # Adjacent draw: cell with probability |q_c|/D, then one of its
                # 32 edges uniformly.
                target = ucells[i] * D
                cum = 0.0
                cell = nzc[0]
                for b in range(nnz):
                    qq = q[nzc[b]]
                    cum += qq if qq >= 0 else -qq
                    if target < cum:
                        cell = nzc[b]
                        break
                e = int(uedges[i] * 32)
                mu, x0, x1, x2, x3 = _cell_edge(cell, e, N)
        # Delta q on the touched hypercubes, deduplicated into (cells, vals).
        nc = 0
        for t in range(st_ptr[mu], st_ptr[mu + 1]):
            r0 = (x0 + st_s[t, 0] + N) % N
            r1 = (x1 + st_s[t, 1] + N) % N
            r2 = (x2 + st_s[t, 2] + N) % N
            r3 = (x3 + st_s[t, 3] + N) % N
            fv = F2[st_p[t], ((r0 * N + r1) * N + r2) * N + r3]
            if fv == 0:
                continue
            v = st_k[t] * c * fv
            o0 = (x0 + st_o[t, 0] + N) % N
            o1 = (x1 + st_o[t, 1] + N) % N
            o2 = (x2 + st_o[t, 2] + N) % N
            o3 = (x3 + st_o[t, 3] + N) % N
            cell = ((o0 * N + o1) * N + o2) * N + o3
            found = False
            for b in range(nc):
                if cells[b] == cell:
                    vals[b] += v
                    found = True
                    break
            if not found:
                cells[nc] = cell
                vals[nc] = v
                nc += 1
        # Price the mess: Delta D against the dense q.
        dD = 0
        for b in range(nc):
            if vals[b] != 0:
                q0 = q[cells[b]]
                dD += abs(q0 + vals[b]) - abs(q0)
        if max_defects < 0 or D + dD <= max_defects:
            srav = ((x0 * N + x1) * N + x2) * N + x3
            A = dphi2[mu, srav] - 2 * np.pi * n2[mu, srav]
            dS = (kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2)
            # Price the sector change: the table w[k], k = D/2, when present (its
            # emptiness selects the geometric path, kept expression-identical so the
            # fugacity chain reproduces bit-for-bit).
            if w.size > 0:
                ratio = w[(D + dD) // 2] / w[D // 2]
            else:
                ratio = fugacity ** np.float64(dD)
            if w2.size > 0:
                newD = D + dD
                if D == 2 or D == 4 or newD == 2 or newD == 4:
                    # Current multiset from the tracked nonzeros.
                    ccur = np.empty(nnz, dtype=np.int64)
                    qcur = np.empty(nnz, dtype=np.int64)
                    mcur = 0
                    for b2 in range(nnz):
                        ccur[mcur] = nzc[b2]
                        qcur[mcur] = q[nzc[b2]]
                        mcur += 1
                    # Candidate multiset: current cells with deltas applied,
                    # plus touched cells that turn on.  The candidate can never
                    # exceed the current nonzeros plus the touched cells.
                    cnew = np.empty(nnz + nc, dtype=np.int64)
                    qnew = np.empty(nnz + nc, dtype=np.int64)
                    mnew = 0
                    for b2 in range(nnz):
                        cell = nzc[b2]
                        qq = q[cell]
                        for t in range(nc):
                            if cells[t] == cell:
                                qq += vals[t]
                                break
                        if qq != 0:
                            cnew[mnew] = cell
                            qnew[mnew] = qq
                            mnew += 1
                    for t in range(nc):
                        if vals[t] == 0:
                            continue
                        if q[cells[t]] == 0:      # turns on
                            cnew[mnew] = cells[t]
                            qnew[mnew] = vals[t]
                            mnew += 1
                    Wcur = _geo_weight(ccur, qcur, mcur, w2, rsq_shell, N)
                    Wnew = _geo_weight(cnew, qnew, mnew, w2, rsq_shell, N)
                    ratio = ratio * (Wnew / Wcur)
            if gamma.size > 0:
                nl = np.float64(mus.shape[0])
                pu = 1.0 / nl
                if D > 0:
                    g = gamma[D // 2]
                    s_fwd = _link_charge_sum(mu, x0, x1, x2, x3, q, N)
                    p_fwd = g * pu + (1.0 - g) * s_fwd / (32.0 * D)
                else:
                    p_fwd = pu
                newD2 = D + dD
                if newD2 > 0:
                    g2 = gamma[newD2 // 2]
                    s_rev = _link_charge_sum_delta(mu, x0, x1, x2, x3,
                                                   q, cells, vals, nc, N)
                    p_rev = g2 * pu + (1.0 - g2) * s_rev / (32.0 * newD2)
                else:
                    p_rev = pu
                ratio = ratio * (p_rev / p_fwd)
            if us[i] < np.exp(-dS) * ratio:
                n2[mu, srav] += c
                for t in range(df_ptr[mu], df_ptr[mu + 1]):
                    fp0 = (x0 + df_off[t, 0] + N) % N
                    fp1 = (x1 + df_off[t, 1] + N) % N
                    fp2 = (x2 + df_off[t, 2] + N) % N
                    fp3 = (x3 + df_off[t, 3] + N) % N
                    F2[df_plane[t], ((fp0 * N + fp1) * N + fp2) * N + fp3] += df_val[t] * c
                # Dense q and the compact nonzero list, maintained incrementally.
                for b in range(nc):
                    if vals[b] == 0:
                        continue
                    cell = cells[b]
                    q0 = q[cell]
                    q1 = q0 + vals[b]
                    q[cell] = q1
                    if q0 == 0:
                        nzc[nnz] = cell
                        nnz += 1
                    elif q1 == 0:
                        for b2 in range(nnz):
                            if nzc[b2] == cell:
                                nzc[b2] = nzc[nnz - 1]
                                nnz -= 1
                                break
                D += dD
                acc += 1
        i += 1
        # ---- classify the sector at this tick (accepted or not), as in step_reference.
        if t_sector.size > 0:
            # Per-tick sector histogram (empty t_sector: not tallied).
            t_sector[D // 2] += 1
        if D == 0:
            if tstate[0] > 0:
                # Close the excursion: power-of-two length bin, top bin saturating.
                x = tstate[0]
                b = 0
                while x > 0:
                    x >>= 1
                    b += 1
                if b > 31:
                    b = 31
                exc_hist[b] += 1
                tstate[1] += 1
                tstate[0] = 0
            if tstate[3] == 1:
                # A vacuum return after touching the top sector: one round trip.
                tstate[4] += 1
                tstate[3] = 0
            vac += 1
            if vac_stop > 0 and vac == vac_stop:
                break
        else:
            tstate[0] += 1
            if max_defects >= 0 and D == max_defects:
                tstate[3] = 1
            if D == 2 and nnz == 2:
                v1 = q[nzc[0]]
                v2 = q[nzc[1]]
                if v1 == -v2 and (v1 == 1 or v1 == -1):
                    if v1 == 1:
                        plus = nzc[0]; minus = nzc[1]
                    else:
                        plus = nzc[1]; minus = nzc[0]
                    p3 = plus % N; pr = plus // N
                    p2 = pr % N; pr //= N
                    p1 = pr % N; p0 = pr // N
                    m3 = minus % N; mr = minus // N
                    m2 = mr % N; mr //= N
                    m1 = mr % N; m0 = mr // N
                    d0 = (p0 - m0 + N) % N
                    d1 = (p1 - m1 + N) % N
                    d2 = (p2 - m2 + N) % N
                    d3 = (p3 - m3 + N) % N
                    # Min-image separation squared: the transport ceiling.
                    mw0 = d0 if d0 <= N - d0 else N - d0
                    mw1 = d1 if d1 <= N - d1 else N - d1
                    mw2 = d2 if d2 <= N - d2 else N - d2
                    mw3 = d3 if d3 <= N - d3 else N - d3
                    rsq = mw0 * mw0 + mw1 * mw1 + mw2 * mw2 + mw3 * mw3
                    if rsq > tstate[2]:
                        tstate[2] = rsq
                    if tally:
                        H_pair[((d0 * N + d1) * N + d2) * N + d3] += 1
            elif tally and D == 4:
                # Neutrality (sum q = 0 identically) pins each nnz to one sorted-charge
                # class: nnz=4 -> {+1,+1,-1,-1}; nnz=3 -> {+2,-1,-1} or {+1,+1,-2} by
                # the doubled charge's sign; nnz=2 -> {+2,-2}.  Each class tallies the
                # inverse of the state's umbrella weight (1.0 with the umbrella off).
                inv = 1.0
                if w2.size > 0:
                    ccur = np.empty(8, dtype=np.int64)
                    qcur = np.empty(8, dtype=np.int64)
                    for b2 in range(nnz):
                        ccur[b2] = nzc[b2]
                        qcur[b2] = q[nzc[b2]]
                    inv = 1.0 / _geo_weight(ccur, qcur, nnz, w2, rsq_shell, N)
                if nnz == 4:
                    H_four[0] += inv
                elif nnz == 3:
                    for b in range(3):
                        vv = q[nzc[b]]
                        if vv == 2:
                            H_four[1] += inv
                            break
                        if vv == -2:
                            H_four[2] += inv
                            break
                elif nnz == 2:
                    v1 = q[nzc[0]]
                    if v1 == 2 or v1 == -2:
                        H_four[3] += inv
    return i, D, nnz, acc, vac
