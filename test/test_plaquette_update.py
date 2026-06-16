#!/usr/bin/env python

import numpy as np
import supervillain


def test_plaquette_update_D2_valid():
    L = supervillain.lattice.Lattice(D=2, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)
    G = supervillain.generator.worldline.PlaquetteUpdate(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert S.valid(result)
    assert result['m'].shape == cfg['m'].shape
    assert result['v'].shape == cfg['v'].shape


def test_plaquette_update_D3_valid():
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)
    G = supervillain.generator.worldline.PlaquetteUpdate(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert S.valid(result)
    assert result['m'].shape == cfg['m'].shape
    assert result['v'].shape == cfg['v'].shape


def test_plaquette_update_uses_all_2form_components():
    """After many steps, all C(D,2) plaquette components of v must have been touched."""
    np.random.seed(7)
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)
    G = supervillain.generator.worldline.PlaquetteUpdate(S)
    cfg = S.configurations(1)[0]
    for _ in range(200):
        cfg = G.step(cfg)
    v = cfg['v']
    n_comps = len(L.components[2])
    for c in range(n_comps):
        assert (v[c] != 0).any(), f"v component {c} ({L.components[2][c]}) never changed"
