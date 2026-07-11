#!/usr/bin/env python
r"""
The compiled DefectGas tick kernel must reproduce the pure-python reference chain
bit-for-bit: same proposals, same accepts, same fields, same sector tallies.  These
tests drive the two paths from identical seeds and demand exact equality, through the
Generator step()/step_reference() API.
"""

import numpy as np

import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection import DefectGas


def _action(N=4, kappa=0.05):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _cold(S):
    # step() freezes phi, so a rough random phi background makes vacuum returns
    # needlessly rare; the step tests start cold instead.
    L = S.Lattice
    return (np.zeros((1,) + tuple(L.dims)),
            np.zeros((4,) + tuple(L.dims), dtype=np.int64))


def _twins(S, seed=17, **kwargs):
    return (DefectGas(S, rng=np.random.default_rng(seed), **kwargs),
            DefectGas(S, rng=np.random.default_rng(seed), **kwargs))


def test_step_matches_reference_quartic_sector():
    # A fugacity high enough that the D = 4 classes populate, so the kernel's
    # nnz-based classification is exercised against the sorted-charge dict --- but
    # low enough that the chain still comes home to emit.
    S = _action()
    fast, slow = _twins(S, seed=23, fugacity=0.2, D_max=8, emit_every=100)
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}

    four = 0.
    for _ in range(3):
        a = fast.step(a)
        b = slow.step_reference(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert np.array_equal(a['Theta_Theta'], b['Theta_Theta'])
        assert np.array_equal(a['Four_Defect'], b['Four_Defect'])
        four += a['Four_Defect'].sum()
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted
    assert four > 0                             # quartic sector actually visited


def test_step_matches_reference_uncapped():
    S = _action()
    fast, slow = _twins(S, seed=29, fugacity=0.1, D_max=None, emit_every=300)
    phi, n = _cold(S)
    a = {'phi': phi, 'n': n}
    b = {'phi': phi, 'n': n}

    for _ in range(2):
        a = fast.step(a)
        b = slow.step_reference(b)
        assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
        assert np.array_equal(a['Theta_Theta'], b['Theta_Theta'])
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted


def test_step_matches_reference():
    S = _action()
    fast, slow = _twins(S, seed=3, fugacity=0.1, D_max=8, emit_every=3000)
    phi, n = _cold(S)
    cfg_f = {'phi': phi, 'n': n}
    cfg_s = {'phi': phi, 'n': n}

    excursions = 0.
    max_rsq = 0.
    for _ in range(3):
        cfg_f = fast.step(cfg_f)
        cfg_s = slow.step_reference(cfg_s)
        assert np.array_equal(np.asarray(cfg_f['n']), np.asarray(cfg_s['n']))
        assert np.array_equal(cfg_f['Theta_Theta'], cfg_s['Theta_Theta'])
        assert cfg_f['Vacuum_Ticks'] == cfg_s['Vacuum_Ticks']
        assert np.array_equal(cfg_f['Four_Defect'], cfg_s['Four_Defect'])
        assert cfg_f['Pair_Excursions'] == cfg_s['Pair_Excursions']
        assert cfg_f['Max_Pair_RSq'] == cfg_s['Max_Pair_RSq']
        assert np.array_equal(cfg_f['Excursion_Lengths'], cfg_s['Excursion_Lengths'])
        assert cfg_f['Ticks'] == cfg_s['Ticks']
        assert cfg_f['Ticks'] >= cfg_f['Vacuum_Ticks'] > 0
        excursions += cfg_f['Pair_Excursions']
        max_rsq = max(max_rsq, cfg_f['Max_Pair_RSq'])

    assert 'Ticks' in fast.inline_observables(1)
    assert fast.proposed == slow.proposed
    assert fast.accepted == slow.accepted
    assert fast.D_trace == slow.D_trace
    # Transport happened over the run and both paths saw it identically.
    assert excursions > 0
    assert max_rsq > 0


def test_paths_interleave():
    # step and step_reference maintain one coherent chain state, so mixing them on a
    # single instance equals an all-reference twin.
    S = _action()
    mixed, pure = _twins(S, seed=11, fugacity=0.1, D_max=8, emit_every=300)
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
