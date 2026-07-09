#!/usr/bin/env python
r"""
The compiled DefectGas tick kernel must reproduce the pure-python reference chain
bit-for-bit: same proposals, same accepts, same fields, same sector tallies.  These
tests drive the two paths from identical seeds and demand exact equality, through both
the standalone run() driver and the Generator step()/step_reference() API.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _start(S, seed=5):
    L = S.Lattice
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, size=(1,) + tuple(L.dims))
    n = np.zeros((4,) + tuple(L.dims), dtype=np.int64)
    return phi, n


def _cold(S):
    # step() freezes phi, so a rough random phi background makes vacuum returns
    # needlessly rare; the step tests start cold instead.
    L = S.Lattice
    return (np.zeros((1,) + tuple(L.dims)),
            np.zeros((4,) + tuple(L.dims), dtype=np.int64))


def _twins(S, seed=17, **kwargs):
    return (DefectGas(S, rng=np.random.default_rng(seed), **kwargs),
            DefectGas(S, rng=np.random.default_rng(seed), **kwargs))


def _assert_blocks_equal(fast, slow):
    assert len(fast.blocks) == len(slow.blocks)
    for (hp_f, hz_f, h4_f), (hp_s, hz_s, h4_s) in zip(fast.blocks, slow.blocks):
        assert np.array_equal(hp_f, hp_s)
        assert hz_f == hz_s
        assert np.array_equal(h4_f, h4_s)


def test_run_matches_reference():
    S = _action()
    fast, slow = _twins(S, zeta=0.2, D_max=8)
    phi, n = _start(S)

    pf, nf = fast.run(phi, n, sweeps=5)
    ps, ns = slow.run(phi, n, sweeps=5, compiled=False)
    fast.close_block()
    slow.close_block()

    assert np.array_equal(nf, ns)
    assert np.array_equal(pf, ps)
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted
    assert fast.D_trace == slow.D_trace
    _assert_blocks_equal(fast, slow)
    # Not vacuous: the chain moved and visited both tallied sectors' feeders.
    assert fast.accepted > 0
    assert fast.blocks[0][1] > 0            # vacuum dwell
    assert fast.blocks[0][0].sum() > 0      # single-pair dwell


def test_run_matches_reference_quartic_sector():
    # A fugacity high enough that the D = 4 classes are actually populated, so the
    # kernel's nnz-based classification is exercised against the sorted-charge dict.
    S = _action()
    fast, slow = _twins(S, seed=23, zeta=0.5, D_max=8)
    phi, n = _start(S)

    fast.run(phi, n, sweeps=4)
    slow.run(phi, n, sweeps=4, compiled=False)
    fast.close_block()
    slow.close_block()

    _assert_blocks_equal(fast, slow)
    assert fast.blocks[0][2].sum() > 0      # four-defect dwell


def test_run_matches_reference_uncapped():
    S = _action()
    fast, slow = _twins(S, seed=29, zeta=0.1, D_max=None)
    phi, n = _start(S)

    pf, nf = fast.run(phi, n, sweeps=3)
    ps, ns = slow.run(phi, n, sweeps=3, compiled=False)

    assert np.array_equal(nf, ns)
    assert np.array_equal(pf, ps)
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted


def test_step_matches_reference():
    S = _action()
    fast, slow = _twins(S, seed=3, zeta=0.1, D_max=8, emit_every=300)
    phi, n = _cold(S)
    cfg_f = {'phi': phi, 'n': n}
    cfg_s = {'phi': phi, 'n': n}

    for _ in range(3):
        cfg_f = fast.step(cfg_f)
        cfg_s = slow.step_reference(cfg_s)
        assert np.array_equal(np.asarray(cfg_f['n']), np.asarray(cfg_s['n']))
        assert np.array_equal(cfg_f['Theta_Theta'], cfg_s['Theta_Theta'])
        assert cfg_f['Vacuum_Ticks'] == cfg_s['Vacuum_Ticks']
        assert np.array_equal(cfg_f['Four_Defect'], cfg_s['Four_Defect'])

    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted
    assert fast.D_trace == slow.D_trace


def test_paths_interleave():
    # step and step_reference maintain one coherent chain state, so mixing them on a
    # single instance equals an all-reference twin.
    S = _action()
    mixed, pure = _twins(S, seed=11, zeta=0.1, D_max=8, emit_every=300)
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}

    for advance in (mixed.step, mixed.step_reference, mixed.step):
        a = advance(a)
        b = pure.step_reference(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert np.array_equal(a['Theta_Theta'], b['Theta_Theta'])
        assert np.array_equal(a['Four_Defect'], b['Four_Defect'])

    assert mixed.proposed == pure.proposed
    assert mixed.accepted == pure.accepted
