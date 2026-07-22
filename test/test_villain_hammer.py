#!/usr/bin/env python

import pytest
import supervillain


def test_villain_hammer_includes_worm_in_D2():
    L = supervillain.lattice.Lattice(D=2, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    H = supervillain.generator.villain.Hammer(S)
    assert 'ClassicWorm' in str(H)


def test_villain_hammer_uses_heatbaths_finite_W():
    # The default ergodic sampler uses the exact heatbaths, not the Metropolis
    # SiteUpdate/LinkUpdate.
    L = supervillain.lattice.Lattice(D=2, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    s = str(supervillain.generator.villain.Hammer(S))
    assert 'SiteHeatbath' in s
    assert 'LinkHeatbath' in s
    assert 'ExactHeatbath' in s
    assert 'CohomologyHeatbath' in s
    assert 'SiteUpdate' not in s
    assert 'LinkUpdate' not in s
    # The closed-n moves are heatbaths too now, not Metropolis.
    assert 'ExactUpdate' not in s
    assert 'CohomologyUpdate' not in s


def test_villain_hammer_uses_site_heatbath_at_W_infinity():
    # When W=∞ there is no link update (LinkHeatbath's coset is undefined), but
    # φ is still updated by the exact heatbath.
    L = supervillain.lattice.Lattice(D=2, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=float('inf'))
    s = str(supervillain.generator.villain.Hammer(S))
    assert 'SiteHeatbath' in s
    assert 'SiteUpdate' not in s
    assert 'LinkHeatbath' not in s
    assert 'LinkUpdate' not in s


def test_villain_hammer_produces_valid_configs_W2():
    # The heatbath-based Hammer must still maintain dn ≡ 0 (mod W) at W>1.
    L = supervillain.lattice.Lattice(D=2, N=4)
    S = supervillain.action.Villain(L, kappa=0.3, W=2)
    e = supervillain.Ensemble(S).generate(30, supervillain.generator.villain.Hammer(S),
                                          start='cold')
    for i in range(len(e)):
        assert S.valid(e.configuration[i])


def test_villain_hammer_includes_overrelaxation_by_default():
    L = supervillain.lattice.Lattice(D=2, N=6)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    assert 'SiteOverrelaxation' in str(supervillain.generator.villain.Hammer(S))


def test_villain_hammer_overrelax_must_be_positive():
    # Overrelaxation is unconditional in the Hammer; overrelax must be >= 1.
    L = supervillain.lattice.Lattice(D=2, N=6)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.villain.Hammer(S, overrelax=0)


def test_villain_hammer_omits_worm_in_D3():
    # The ClassicWorm is only implemented for D=2; Hammer should still build
    # (and step) in higher D by omitting it, rather than raising.
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    H = supervillain.generator.villain.Hammer(S)
    assert 'ClassicWorm' not in str(H)

    cfg = S.configurations(1)[0]
    result = H.step(cfg)
    assert result['phi'].shape == cfg['phi'].shape
    assert result['n'].shape == cfg['n'].shape
