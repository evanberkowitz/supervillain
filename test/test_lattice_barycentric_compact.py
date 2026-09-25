import itertools
import numpy as np
import pytest

import supervillain.lattice.barycentric.doubled as dbl
import supervillain.lattice.barycentric.compact as cpt


def test_component_counts_match_simplices_per_site():
    """S_p is the number of p-simplices per original site: a component occurs
    exactly once per site, which is what makes the layout 100% dense."""
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = dbl.Lattice(D, N)
        c = cpt._compact_structures(D)
        for p in range(D + 1):
            assert c.count[p] == lat.count(p) // N**D, (D, p)


def test_four_dimensional_component_counts():
    c = cpt._compact_structures(4)
    assert [c.count[p] for p in range(5)] == [16, 240, 800, 960, 384]


def test_components_are_valid_and_lexicographic():
    """A type is realized only where the directions it releases are all even."""
    D = 4
    s = dbl._structures(D)
    c = cpt._compact_structures(D)
    for p in range(D + 1):
        for parity, t in c.components[p]:
            S = {mu for mu in range(D) if parity[mu]}
            assert not (s.touched[p][t] & S), (p, parity, t)
        keys = [(sum(b << mu for mu, b in enumerate(par)), t)
                for par, t in c.components[p]]
        assert keys == sorted(keys), f'p={p}: components not in canonical order'


def test_component_index_inverts_components():
    c = cpt._compact_structures(4)
    for p in range(5):
        for i, comp in enumerate(c.components[p]):
            assert c.component_index[p][comp] == i
