#!/usr/bin/env python

import supervillain


def test_neighborhood_update_D3():
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    G = supervillain.generator.villain.NeighborhoodUpdate(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape


def test_neighborhood_update_D2_works():
    L = supervillain.lattice.Lattice(D=2, N=6)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    G = supervillain.generator.villain.NeighborhoodUpdate(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape
