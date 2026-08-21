#!/usr/bin/env python

import supervillain


def test_villain_hammer_includes_worm_in_D2():
    L = supervillain.lattice.Lattice(D=2, N=4)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    H = supervillain.generator.villain.Hammer(S)
    assert 'ClassicWorm' in str(H)


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
