#!/usr/bin/env python
# coding: utf-8

import numpy as np
from supervillain.compact import Form, d

class Villain:

    def __init__(self, kappa):
        self.kappa = kappa

    def __call__(self, phi, n):

        return (self.kappa / 2) * ((d(phi) - 2*np.pi*n)**2).sum()

    def transform(self, phi, n, m):
        return phi + 2*np.pi*m, n + d(m)
