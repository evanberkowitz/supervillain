#!/usr/bin/env python

import numpy as np
import supervillain
import supervillain.ensemble


def test_autocorrelation_time_falls_back_when_nothing_fluctuates(monkeypatch):
    L = supervillain.lattice.Lattice2D(3)
    S = supervillain.action.Villain(L, kappa=0.5, W=1)
    G = supervillain.generator.villain.Hammer(S)
    e = supervillain.Ensemble(S).generate(8, G, start='cold')

    # Emulate an ensemble in which no observable fluctuates enough: every
    # per-observable estimate raises, so the collected dictionary is empty.
    def too_small(*args, **kwargs):
        raise ValueError('The fluctuations are too small to reliably determine an autocorrelation.')

    monkeypatch.setattr(supervillain.ensemble, 'autocorrelation_time', too_small)

    # Should warn and fall back rather than raise on an empty max().
    tau = e.autocorrelation_time()
    assert tau == int(np.ceil(len(e) / 2))
