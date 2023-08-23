#!/usr/bin/env python

import numpy as np

from supervillain.observable import Observable

class TopologicalSusceptibility(Observable):

    @staticmethod
    def Villain(S, phi, n):

        L = S.Lattice
        return np.mean(L.d(1, n)**2)



