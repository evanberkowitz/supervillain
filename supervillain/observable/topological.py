#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.lattice import d, wedge
from supervillain.observable import Observable, Scalar


class AbsoluteTopologicalChargeDensity(Scalar, Observable):
    r'''The absolute topological-charge density in the four-dimensional
    :math:`W=1` Villain model.

    For the integer-valued Villain 1-form :math:`n`, define the local
    topological charge as the 4-form

    .. math::

       Q(x) = (dn \wedge dn)(x).

    On the periodic lattice, :math:`Q=d(n\wedge dn)` is exact, so its signed
    total vanishes configuration by configuration.  The absolute local charge
    remains nontrivial: opposite-sign charge defects contribute rather than
    cancel.  Mirroring :class:`~.ActionDensity`, this observable reports the
    intensive quantity

    .. math::

       \texttt{AbsoluteTopologicalChargeDensity}
       = \frac{1}{\Lambda}\sum_x \left|Q(x)\right|,

    where :math:`\Lambda` is the number of four-cells (equivalently, sites) on
    the periodic four-dimensional hypercubic lattice.
    '''

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        The observable is only defined for the four-dimensional :math:`W=1`
        Villain model.  Including it in the autocorrelation computation on any
        other ensemble would trigger a measurement that raises
        :class:`NotImplementedError`, so we restrict it to the ensembles where it
        can actually be measured before deferring to the usual :class:`~.Scalar`
        decision.
        '''
        S = ensemble.Action
        return (
            isinstance(S, supervillain.action.Villain)
            and S.Lattice.D == 4
            and S.W == 1
            and super().autocorrelation(ensemble)
        )

    @staticmethod
    def Villain(S, n):
        r'''Measure :math:`\Lambda^{-1}\sum_x |(dn\wedge dn)(x)|`.

        This observable is intentionally restricted to the four-dimensional
        :math:`W=1` model.  The package does not currently define the combined
        :math:`W>1` and topological-charge-constrained theory.
        '''

        L = S.Lattice
        if L.D != 4:
            raise NotImplementedError(
                'AbsoluteTopologicalChargeDensity requires a four-dimensional lattice.'
            )
        if S.W != 1:
            raise NotImplementedError(
                'AbsoluteTopologicalChargeDensity is currently defined only for W=1.'
            )

        field_strength = d(n)
        charge = wedge(field_strength, field_strength)
        return np.abs(charge).sum() / L.cells_of_degree[4]
