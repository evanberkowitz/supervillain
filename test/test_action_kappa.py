#!/usr/bin/env python

import pytest
import supervillain
from supervillain.lattice import Lattice


def test_villain_rejects_negative_kappa():
    L = Lattice(2, 5)
    with pytest.raises(ValueError):
        supervillain.action.Villain(L, kappa=-0.1)


def test_nointersections_rejects_negative_kappa():
    L = Lattice(4, 5)
    with pytest.raises(ValueError):
        supervillain.action.NoIntersections(L, kappa=-0.1)


def test_nointersections_negative_kappa_names_itself():
    # The guard lives in Villain.__init__ but reports type(self), so a NoIntersections
    # error message names NoIntersections, not its Villain base.
    L = Lattice(4, 5)
    with pytest.raises(ValueError, match='NoIntersections'):
        supervillain.action.NoIntersections(L, kappa=-1.0)


def test_zero_kappa_is_allowed():
    # kappa = 0 is degenerate but a legal action; only kappa < 0 is rejected.
    supervillain.action.Villain(Lattice(2, 5), kappa=0.0)
    supervillain.action.NoIntersections(Lattice(4, 5), kappa=0.0)
