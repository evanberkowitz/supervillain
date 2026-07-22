#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.lattice import Form


def _action(D=2, N=6, kappa=0.5, W=1):
    L = supervillain.lattice.Lattice(D=D, N=N)
    return supervillain.action.Villain(L, kappa=kappa, W=W)


def _S(S, cfg):
    L = S.Lattice
    return S(Form(np.asarray(cfg['phi']), degree=0, lattice=L),
             Form(np.asarray(cfg['n']), degree=1, lattice=L))


def _randomize(S, cfg, seed):
    r = np.random.default_rng(seed)
    L = S.Lattice
    cfg = dict(cfg)
    cfg['phi'] = cfg['phi'] + Form(r.normal(size=np.asarray(cfg['phi']).shape),
                                   degree=0, lattice=L)
    cfg['n'] = cfg['n'] + r.integers(-1, 2, size=np.asarray(cfg['n']).shape)
    return cfg


def test_overrelaxation_shapes():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.SiteOverrelaxation(S)
    cfg = S.configurations(1)[0]
    result = G.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape


def test_overrelaxation_leaves_n_untouched():
    S = _action(D=2, N=6)
    G = supervillain.generator.villain.SiteOverrelaxation(S, rng=np.random.default_rng(0))
    cfg = S.configurations(1)[0]
    cfg['n'] = cfg['n'] + 1
    before = np.asarray(cfg['n']).copy()
    result = G.step(cfg)
    assert (np.asarray(result['n']) == before).all()


def test_overrelaxation_preserves_action():
    # The defining property: reflection about the conditional mean is action-neutral.
    for D, N, kappa in [(2, 6, 0.5), (3, 4, 0.7)]:
        S = _action(D=D, N=N, kappa=kappa)
        G = supervillain.generator.villain.SiteOverrelaxation(
            S, applications=3, rng=np.random.default_rng(1))
        cfg = _randomize(S, S.configurations(1)[0], seed=2)
        before = _S(S, cfg)
        after = _S(S, G.step(cfg))
        assert abs(after - before) < 1e-8 * (1 + abs(before))


def test_overrelaxation_changes_phi():
    # Non-trivial move: phi actually moves (guards against an accidental no-op).
    S = _action(D=2, N=6, kappa=0.5)
    G = supervillain.generator.villain.SiteOverrelaxation(S, applications=1,
                                                          rng=np.random.default_rng(4))
    cfg = _randomize(S, S.configurations(1)[0], seed=5)
    before = np.asarray(cfg['phi']).copy()
    after = np.asarray(G.step(cfg)['phi'])
    assert np.abs(after - before).max() > 1e-6


def test_two_applications_is_not_the_identity():
    # A full sweep composes non-commuting per-color reflections -> not an involution,
    # so applications>1 is meaningful, not self-cancelling.
    S = _action(D=2, N=6, kappa=0.5)
    G = supervillain.generator.villain.SiteOverrelaxation(S, applications=2,
                                                          rng=np.random.default_rng(7))
    cfg = _randomize(S, S.configurations(1)[0], seed=8)
    before = np.asarray(cfg['phi']).copy()
    after = np.asarray(G.step(cfg)['phi'])
    assert np.abs(after - before).max() > 1e-3
