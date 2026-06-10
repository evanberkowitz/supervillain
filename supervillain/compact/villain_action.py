#!/usr/bin/env python

import numpy as np

from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.configurations import Configurations
from supervillain.lattice.compact import Form, d
from supervillain.compact.villain import Villain as _CompactVillain
import supervillain.layout as layout


class Villain(ReadWriteable):
    r'''
    Villain action on a compact :class:`~supervillain.compact.Lattice`, wired for
    :class:`~supervillain.Ensemble` (``Batch`` columns of :class:`~supervillain.compact.Form`).
    '''

    def __init__(self, lattice2d, kappa, W=1):
        self.Lattice2D = lattice2d
        self.Lattice = layout.compact_lattice(lattice2d)
        self.kappa = kappa
        self.W = W
        self._core = _CompactVillain(kappa)

    def __str__(self):
        return f'compact.Villain({self.Lattice2D}, κ={self.kappa}, W={self.W})'

    def __call__(self, phi, n):
        return self._core(phi, n)

    def local(self, phi, n):
        r'''Per-link contribution :math:`\frac{\kappa}{2}(d\phi - 2\pi n)^2` as a 1-form.'''
        return (self.kappa / 2) * (d(phi) - 2 * np.pi * n)**2

    def transform(self, phi, n, m):
        return self._core.transform(phi, n, m)

    def configurations(self, count):
        L = self.Lattice
        return Configurations({
            'phi': Batch(count, cls=Form, degree=0, lattice=L),
            'n':   Batch(count, cls=Form, degree=1, lattice=L, dtype=int),
        })
