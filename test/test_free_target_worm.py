#!/usr/bin/env python

import numpy as np
import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice, Form, d
from supervillain.generator.no_intersection.charge import charge


def _worm(N=6, kappa=0.3):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    return gen.FreeTargetWorm(S)


def _key(shape):
    return frozenset(shape)


def test_family_is_deduped_and_negation_closed():
    w = _worm()
    keys = [_key(s) for s in w._family]
    assert len(keys) == len(set(keys))                     # no duplicate placements
    keyset = set(keys)
    for shape in w._family:
        neg = _key(tuple((mu, r, -c) for mu, r, c in shape))
        assert neg in keyset                               # negation-closed
        assert shape in w._self_charge                     # self-charge registered


def test_family_touches_the_head():
    # Support-anchoring: every family member's charge-reach support contains the
    # head (the origin, in relative coordinates) through at least one link.
    w = _worm()
    origin = (0, 0, 0, 0)
    for shape in w._family:
        assert any(origin in w._slot_support(mu, r) for mu, r, _c in shape)


def test_family_contains_singles_pairs_and_library():
    w = _worm()
    sizes = {len(shape) for shape in w._family}
    assert 1 in sizes and 2 in sizes                       # singles and pairs
    assert max(sizes) >= 3                                 # library 3-/4-link templates
