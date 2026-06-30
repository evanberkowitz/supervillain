#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.lattice import d, wedge
from supervillain.observable import Observable, Scalar, DerivedQuantity


def _topological_charge(L, n):
    r'''The local topological charge.

    For the integer-valued Villain 1-form :math:`n`, the local topological
    charge is the 4-form

    .. math::

       Q_x = (dn \wedge dn)_x.

    It is a 4-form, so it is only the top form---and therefore the global
    topological charge density---on a four-dimensional lattice.  We restrict to
    :math:`D=4`; the :math:`W` constraint :math:`[dn \equiv 0 \bmod W]` is
    irrelevant to the *measurement*: when :math:`W>1` the field strength
    :math:`dn` is quantized in units of :math:`W` (so :math:`Q` comes in units
    of :math:`W^2`), and when :math:`W=\infty` the constraint forces
    :math:`dn=0` so :math:`Q\equiv 0`.

    Parameters
    ----------
    L : supervillain.lattice.Lattice
    n : np.ndarray
        An integer-valued 1-form.

    Returns
    -------
    np.ndarray
        The charge 4-form, shape ``(1,) + L.dims``.
    '''
    if L.D != 4:
        raise NotImplementedError(
            'Topological-charge observables require a four-dimensional lattice.'
        )
    field_strength = d(n)
    return wedge(field_strength, field_strength)


class TopologicalChargeDensity(Observable):
    r'''The local topological-charge density in the four-dimensional Villain
    model,

    .. math::

       \texttt{TopologicalChargeDensity}_x = Q_x = (dn\wedge dn)_x,

    a field carrying one value per four-cell.
    It is the per-configuration ingredient from which the other
    topological-charge observables are built: :class:`~.TopologicalCharge` sums
    it, :class:`~.AbsoluteTopologicalChargeDensity` averages its magnitude,
    :class:`~.TopologicalChargeSquared` averages its square, and
    :class:`~.TopologicalTwoPoint` autocorrelates it.

    Because :math:`Q=d(n\wedge dn)` is exact, its sum over the lattice vanishes
    configuration by configuration, so :math:`\langle Q_x\rangle = 0` pointwise
    by translation invariance.
    '''

    @staticmethod
    def Villain(S, n):
        r'''Measure the charge field :math:`Q_x = (dn\wedge dn)_x`.'''

        charge = _topological_charge(S.Lattice, n)
        return np.asarray(charge).sum(axis=0)


class AbsoluteTopologicalChargeDensity(Scalar, Observable):
    r'''The absolute topological-charge density in the four-dimensional Villain
    model.

    On the periodic lattice the signed charge :math:`Q` (see
    :class:`~.TopologicalChargeDensity`) is exact, so its signed total vanishes
    configuration by configuration.  The absolute local charge remains
    nontrivial: opposite-sign charge defects contribute rather than cancel.
    Mirroring :class:`~.ActionDensity`, this observable reports the intensive
    quantity

    .. math::

       \texttt{AbsoluteTopologicalChargeDensity}
       = \frac{1}{\Lambda}\sum_x \left|Q_x\right|,

    where :math:`\Lambda` is the number of four-cells (equivalently, sites) on
    the periodic four-dimensional hypercubic lattice.
    '''

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        The observable is only defined for the four-dimensional Villain model, so
        we restrict the autocorrelation decision to those ensembles before
        deferring to the usual :class:`~.Scalar` choice.  When :math:`W=\infty`
        the constraint forces :math:`dn=0`, making the charge an identically-zero
        constant with no autocorrelation to estimate, so we exclude it there.
        '''
        S = ensemble.Action
        return (
            isinstance(S, supervillain.action.Villain)
            and S.Lattice.D == 4
            and S.W < float('inf')
            and super().autocorrelation(ensemble)
        )

    @staticmethod
    def Villain(S, TopologicalChargeDensity):
        r'''Measure :math:`\Lambda^{-1}\sum_x |Q_x|` from the charge field.'''

        return np.abs(TopologicalChargeDensity).sum() / S.Lattice.cells_of_degree[4]


class TopologicalCharge(Scalar, Observable):
    r'''The signed total topological charge in the four-dimensional Villain
    model,

    .. math::

       \texttt{TopologicalCharge} = \sum_x Q_x,

    the lattice sum of :class:`~.TopologicalChargeDensity`.  Because the charge
    density is exact this vanishes identically on the periodic lattice---
    configuration by configuration, for every :math:`W`.  It is therefore
    pointless to measure on its own, but it is the :math:`\langle Q\rangle` that
    would form the quantum-disconnected piece of
    :class:`~.Topological_Topological` were the charge to develop a nonzero
    expectation value (for instance under a topological chemical potential).
    '''

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        The total charge is identically zero on every configuration, so it has
        no fluctuations and is never included in the autocorrelation-time
        computation.
        '''
        return False

    @staticmethod
    def Villain(S, TopologicalChargeDensity):
        r'''Measure :math:`\sum_x Q_x`, the lattice sum of the charge field
        (always zero).'''

        return TopologicalChargeDensity.sum()


class TopologicalChargeSquared(Scalar, Observable):
    r'''The same-site topological-charge correlator in the four-dimensional
    Villain model,

    .. math::

       \texttt{TopologicalChargeSquared}
       = \frac{1}{\Lambda}\sum_x Q_x^2,

    with :math:`\Lambda` the number of four-cells, the intensive same-site value
    :math:`\langle Q_x^2\rangle`.  Because :math:`\langle Q\rangle = 0` (see
    :class:`~.TopologicalCharge`), this is the local topological-charge
    fluctuation---the proper, sign-respecting counterpart of
    :class:`~.AbsoluteTopologicalChargeDensity`.  It equals
    :class:`~.TopologicalTwoPoint` (equivalently :class:`~.Topological_Topological`)
    evaluated at the :py:attr:`~supervillain.lattice.Lattice.origin`,
    configuration by configuration.
    '''

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        Defined only for the four-dimensional Villain model; excluded when
        :math:`W=\infty` because the charge is then identically zero.  See
        :meth:`AbsoluteTopologicalChargeDensity.autocorrelation`.
        '''
        S = ensemble.Action
        return (
            isinstance(S, supervillain.action.Villain)
            and S.Lattice.D == 4
            and S.W < float('inf')
            and super().autocorrelation(ensemble)
        )

    @staticmethod
    def Villain(S, TopologicalChargeDensity):
        r'''Measure :math:`\Lambda^{-1}\sum_x Q_x^2` from the charge field.  The
        field has one entry per four-cell, so the mean is the average over the
        :math:`\Lambda` four-cells.'''

        return np.mean(TopologicalChargeDensity ** 2)


class TopologicalTwoPoint(Observable):
    r'''The translation-averaged two-point function of the local topological
    charge in the four-dimensional Villain model,

    .. math::

       \texttt{TopologicalTwoPoint}_{\Delta x}
       = \frac{1}{\Lambda}\sum_x Q_x\,Q_{x-\Delta x},

    the autocorrelation of :class:`~.TopologicalChargeDensity` computed with the
    Fourier-accelerated :meth:`~supervillain.lattice.Lattice.correlation`.

    Its value at the :py:attr:`~supervillain.lattice.Lattice.origin` equals
    :class:`~.TopologicalChargeSquared`, and because the total charge vanishes
    its sum over :math:`\Delta x` is :math:`\Lambda^{-1}(\sum_x Q_x)^2 = 0`.
    '''

    @staticmethod
    def Villain(S, TopologicalChargeDensity):
        r'''Measure :math:`\Lambda^{-1}\sum_x Q_x\,Q_{x-\Delta x}` from the charge field.'''

        return S.Lattice.correlation(TopologicalChargeDensity, TopologicalChargeDensity)


class Topological_Topological(DerivedQuantity):
    r'''The connected topological-charge correlator: the translation average of
    the connected two-point function,

    .. math::

        \begin{aligned}
            \texttt{Topological\_Topological}_{\Delta x}
            &= \frac{1}{\Lambda}\sum_x \left(
                \langle Q_x Q_{x-\Delta x}\rangle - \langle Q_x\rangle\langle Q_{x-\Delta x}\rangle
                \right)
            \\
            &= \texttt{TopologicalTwoPoint}_{\Delta x}
                - \texttt{correlation}(\langle Q\rangle, \langle Q\rangle)_{\Delta x}.
        \end{aligned}

    Both terms are normalized the same way.

    On the periodic lattice the total charge :math:`\sum_x Q_x = 0` configuration by
    configuration (see :class:`~.TopologicalCharge`), so translation invariance forces
    :math:`\langle Q_x\rangle = 0` and the disconnected piece vanishes in expectation;
    the subtraction is retained so that it is restored automatically should the charge
    ever acquire a nonzero expectation value.
    '''

    @staticmethod
    def default(S, TopologicalTwoPoint, TopologicalChargeDensity):
        L = S.Lattice
        return TopologicalTwoPoint - L.correlation(TopologicalChargeDensity, TopologicalChargeDensity)
