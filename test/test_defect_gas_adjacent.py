#!/usr/bin/env python
r"""
Defect-adjacent proposal targeting: a mixture proposal (gamma_k uniform /
charge-weighted defect-adjacent) with the full Metropolis-Hastings ratio.
gamma is a (K+1)-vector indexed by sector; the empty array is the legacy
sentinel (bit-for-bit today's sampler, same RNG stream).
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _cold(S):
    L = S.Lattice
    return (np.zeros((1,) + tuple(L.dims)),
            np.zeros((4,) + tuple(L.dims), dtype=np.int64))


W = (1.0, 0.04, 2.4e-3)          # a known-healthy N=4 kappa=0.05 table, D_max=4


def test_gamma_default_is_legacy():
    g = DefectGas(_action(), weights=W)
    assert g.gamma.size == 0


def test_gamma_scalar_broadcasts():
    g = DefectGas(_action(), weights=W, gamma=0.5)
    assert g.gamma.shape == (3,)               # K+1 = D_max/2 + 1 = 3
    assert np.all(g.gamma == 0.5)


def test_gamma_vector_accepted():
    g = DefectGas(_action(), weights=W, gamma=(1.0, 0.5, 0.25))
    assert np.array_equal(g.gamma, [1.0, 0.5, 0.25])


def test_gamma_needs_cap():
    try:
        DefectGas(_action(), fugacity=0.1, gamma=0.5)      # D_max None
    except ValueError:
        pass
    else:
        assert False, 'gamma without a capped gas must raise ValueError'


def test_gamma_works_with_capped_fugacity():
    g = DefectGas(_action(), fugacity=0.1, D_max=4, gamma=0.5)
    assert g.gamma.shape == (3,)


def test_gamma_validation():
    S = _action()
    for bad in (0.0, -0.5, 1.5, (0.5, 0.5), (1.0, 0.5, 0.0), (1.0, 0.5, 2.0)):
        try:
            DefectGas(S, weights=W, gamma=bad)
        except ValueError:
            pass
        else:
            assert False, f'gamma={bad} must raise ValueError'


def test_adjacency_geometry_and_normalization():
    # sum_l s(l, q) = 32 D exactly, and the python and kernel helpers agree,
    # on states with a pair, a doubled charge, and a quartic multiset.
    from supervillain.generator.no_intersection.defect_gas import link_charge_sum, cell_edge
    from supervillain.generator.no_intersection import defect_gas_kernel as K

    N = 4
    rng = np.random.default_rng(7)
    for charges in ([1, -1], [2, -2], [1, 1, -1, -1], [2, -1, -1]):
        q = np.zeros(N**4, dtype=np.int64)
        cells = rng.choice(N**4, size=len(charges), replace=False)
        for cell, val in zip(cells, charges):
            q[cell] = val
        D = int(np.abs(q).sum())
        total = 0
        for mu in range(4):
            for site in np.ndindex((N,) * 4):
                s_py = link_charge_sum(mu, site, q, N)
                s_k = K._link_charge_sum(mu, site[0], site[1], site[2], site[3], q, N)
                assert s_py == s_k
                total += s_py
        assert total == 32 * D


def test_cell_edge_roundtrip():
    # Every edge of a cell contains that cell among its 8 containing hypercubes,
    # python and kernel decoders agree, and the 32 edges of a cell are distinct.
    from supervillain.generator.no_intersection.defect_gas import cell_edge
    from supervillain.generator.no_intersection import defect_gas_kernel as K

    N = 4
    for cell in (0, 37, N**4 - 1):
        seen = set()
        c3 = cell % N; r = cell // N
        c2 = r % N; r //= N
        c1 = r % N; c0 = r // N
        for e in range(32):
            mu, site = cell_edge(cell, e, N)
            k_mu, x0, x1, x2, x3 = K._cell_edge(cell, e, N)
            assert (mu, site) == (k_mu, (x0, x1, x2, x3))
            seen.add((mu, site))
            # cell must be among the containing hypercubes of (mu, site):
            containing = set()
            for b in range(8):
                c = list(site)
                j = 0
                for nu in range(4):
                    if nu == mu:
                        continue
                    if (b >> j) & 1:
                        c[nu] = (c[nu] - 1) % N
                    j += 1
                containing.add(((c[0] * N + c[1]) * N + c[2]) * N + c[3])
            assert cell in containing
        assert len(seen) == 32


def test_delta_charge_sum():
    # s(l, q + dq) via the delta helper equals s(l, q') with q' built explicitly.
    from supervillain.generator.no_intersection.defect_gas import link_charge_sum
    from supervillain.generator.no_intersection import defect_gas_kernel as K

    N = 4
    q = np.zeros(N**4, dtype=np.int64)
    q[3] = 1; q[77] = -1
    cells = np.array([3, 40], dtype=np.int64)
    vals = np.array([-1, 1], dtype=np.int64)
    qprime = q.copy(); qprime[3] -= 1; qprime[40] += 1
    for mu in range(4):
        for site in ((0, 0, 0, 0), (0, 0, 0, 3), (2, 1, 0, 1)):
            expect = link_charge_sum(mu, site, qprime, N)
            got = K._link_charge_sum_delta(mu, site[0], site[1], site[2], site[3],
                                           q, cells, vals, 2, N)
            dq = {(0, 0, 0, 3): -1, (0, 2, 2, 0): 1}    # raveled 3 and 40 at N=4
            got_py = link_charge_sum(mu, site, q, N, dq=dq)
            assert got == expect == got_py
