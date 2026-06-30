#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.lattice import d, wedge
from supervillain.observable import Observable, Scalar, DerivedQuantity


def _topological_charge(L, n):
    r'''The local topological-charge density.

    For the integer-valued Villain 1-form :math:`n`, the local topological-charge
    density is the 4-form

    .. math::

       q_x = (dn \wedge dn)_x.

    It is a 4-form, so it is the top form only on a four-dimensional lattice,
    where its lattice sum is the global topological charge :math:`Q = \sum_x q_x`.
    We restrict to :math:`D=4`; the :math:`W` constraint
    :math:`[dn \equiv 0 \bmod W]` is irrelevant to the *measurement*: when
    :math:`W>1` the field strength :math:`dn` is quantized in units of :math:`W`
    (so :math:`q` comes in units of :math:`W^2`), and when :math:`W=\infty` the
    constraint forces :math:`dn=0` so :math:`q\equiv 0`.

    Parameters
    ----------
    L : supervillain.lattice.Lattice
    n : np.ndarray
        An integer-valued 1-form.

    Returns
    -------
    np.ndarray
        The charge-density 4-form, shape ``(1,) + L.dims``.
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

       \texttt{TopologicalChargeDensity}_x = q_x = (dn\wedge dn)_x,

    a field carrying one value per four-cell.
    It is the per-configuration ingredient from which the other
    topological-charge observables are built: :class:`~.TopologicalCharge` sums
    it, :class:`~.TopologicalChargeDensitySquared` averages its square, and
    :class:`~.TopologicalTwoPoint` autocorrelates it.

    Because :math:`q=d(n\wedge dn)` is exact, its lattice sum---the total charge
    :math:`Q`---vanishes configuration by configuration, so
    :math:`\langle q_x\rangle = 0` pointwise by translation invariance.
    '''

    @staticmethod
    def Villain(S, n):
        r'''Measure the charge-density field :math:`q_x = (dn\wedge dn)_x`.'''

        charge = _topological_charge(S.Lattice, n)
        return np.asarray(charge).sum(axis=0)


class TopologicalCharge(Scalar, Observable):
    r'''The signed total topological charge in the four-dimensional Villain
    model,

    .. math::

       \texttt{TopologicalCharge} = Q = \sum_x q_x,

    the lattice sum of :class:`~.TopologicalChargeDensity`.  Because the density
    :math:`q` is exact this vanishes identically on the periodic lattice---
    configuration by configuration, for every :math:`W`.  It is therefore
    pointless to measure on its own; a nonzero :math:`\langle Q\rangle`
    (equivalently a nonzero density expectation
    :math:`\langle q_x\rangle = \langle Q\rangle/\Lambda`) is what would let the
    quantum-disconnected piece of :class:`~.Topological_Topological` survive (for
    instance under a topological chemical potential).
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
        r'''Measure :math:`Q = \sum_x q_x`, the lattice sum of the charge-density
        field (always zero).'''

        return TopologicalChargeDensity.sum()


class TopologicalChargeDensitySquared(Scalar, Observable):
    r'''The same-site topological-charge correlator in the four-dimensional
    Villain model,

    .. math::

       \texttt{TopologicalChargeDensitySquared}
       = \frac{1}{\Lambda}\sum_x q_x^2,

    with :math:`\Lambda` the number of four-cells, the intensive same-site value
    :math:`\langle q_x^2\rangle`.  Because :math:`\langle q_x\rangle = 0` (the
    total charge :math:`Q` vanishes; see :class:`~.TopologicalCharge`), this is
    the local topological-charge fluctuation---a proper, sign-respecting measure
    of local topological activity.  It equals :class:`~.TopologicalTwoPoint`
    (equivalently :class:`~.Topological_Topological`) evaluated at the
    :py:attr:`~supervillain.lattice.Lattice.origin`, configuration by
    configuration.
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
        r'''Measure :math:`\Lambda^{-1}\sum_x q_x^2` from the charge-density field.
        The field has one entry per four-cell, so the mean is the average over
        the :math:`\Lambda` four-cells.'''

        return np.mean(TopologicalChargeDensity ** 2)


class TopologicalTwoPoint(Observable):
    r'''The translation-averaged two-point function of the local topological-charge
    density in the four-dimensional Villain model,

    .. math::

       \texttt{TopologicalTwoPoint}_{\Delta x}
       = \frac{1}{\Lambda}\sum_x q_x\,q_{x-\Delta x},

    the autocorrelation of :class:`~.TopologicalChargeDensity` computed with the
    Fourier-accelerated :meth:`~supervillain.lattice.Lattice.correlation`.

    Its value at the :py:attr:`~supervillain.lattice.Lattice.origin` equals
    :class:`~.TopologicalChargeDensitySquared`, and because the total charge
    vanishes its sum over :math:`\Delta x` is
    :math:`\Lambda^{-1}(\sum_x q_x)^2 = \Lambda^{-1}Q^2 = 0`.
    '''

    @staticmethod
    def Villain(S, TopologicalChargeDensity):
        r'''Measure :math:`\Lambda^{-1}\sum_x q_x\,q_{x-\Delta x}` from the charge-density field.'''

        return S.Lattice.correlation(TopologicalChargeDensity, TopologicalChargeDensity)


class Topological_Topological(DerivedQuantity):
    r'''The connected topological-charge correlator: the translation average of
    the connected two-point function,

    .. math::

        \begin{aligned}
            \texttt{Topological\_Topological}_{\Delta x}
            &= \frac{1}{\Lambda}\sum_x \left(
                \langle q_x q_{x-\Delta x}\rangle - \langle q_x\rangle\langle q_{x-\Delta x}\rangle
                \right)
            \\
            &= \texttt{TopologicalTwoPoint}_{\Delta x}
                - \texttt{correlation}(\langle q\rangle, \langle q\rangle)_{\Delta x}.
        \end{aligned}

    Both terms are normalized the same way.

    On the periodic lattice the total charge :math:`Q = \sum_x q_x = 0`
    configuration by configuration (see :class:`~.TopologicalCharge`), so
    translation invariance forces :math:`\langle q_x\rangle = 0` and the
    disconnected piece vanishes in expectation; the subtraction is retained so
    that it is restored automatically should the charge ever acquire a nonzero
    expectation value.
    '''

    @staticmethod
    def default(S, TopologicalTwoPoint, TopologicalChargeDensity):
        L = S.Lattice
        return TopologicalTwoPoint - L.correlation(TopologicalChargeDensity, TopologicalChargeDensity)
