from supervillain.observable import Scalar, Observable
import numpy as np

class TorusWrapping(Observable):
    r'''
    Both the :class:`~.Villain` and :class:`~.Worldline` formulations have integer-valued numbers that wrap the spatial torus.

    .. warning ::
        Unlike most observables these will NOT match${}^*$ between the different formulations.
        However, since they are both related to the global topology, we expect them to evolve quite slowly and suffer the longest autocorrelation times.

        ${}^*$They both have 0 expectation value by the discrete rotational lattice symmetries, but they don't have related physical motivation.
    '''


    @staticmethod
    def Villain(S, phi, n):
        r'''
        The total wrapping in direction $\mu$ is given by the gauge holonomy

        .. math ::
            \texttt{TorusWrapping}_{\mu} = \sum n_\mu
        '''

        return n.sum(axis=tuple(range(1, n.ndim)))

    @staticmethod
    def Worldline(S, m):
        r'''
        The total wrapping in direction $\mu$ is given by the net particle flux in that direction

        .. math ::
            \texttt{TorusWrapping}_{\mu} = \frac{1}{N} \sum m_\mu

        where we divide by $N$ (the linear lattice size) because a single torus-wrapping worldline
        contributes one unit for each of the $N$ positions along direction $\mu$.
        '''

        return m.sum(axis=tuple(range(1, m.ndim))) / S.Lattice.N

class WrappingSquared(Scalar, Observable):
    r'''
    The sum of squared torus-wrapping numbers over all directions,

    .. math ::

        \texttt{WrappingSquared} = \sum_\mu \texttt{TorusWrapping}_\mu^2.

    Unlike :class:`~.TorusWrapping` itself (which vanishes in expectation by symmetry),
    this is positive semi-definite and carries information about the size of
    topological fluctuations.  It is D-agnostic: in D=2 it reduces to
    $\texttt{TWrapping}^2 + \texttt{XWrapping}^2$.
    '''

    @staticmethod
    def default(S, TorusWrapping):
        return (TorusWrapping**2).sum()


class TWrapping(Scalar, Observable):
    r'''
    Just the time component of :class:`~.TorusWrapping`.
    '''

    @staticmethod
    def default(S, TorusWrapping):
        return TorusWrapping[0]


class XWrapping(Scalar, Observable):
    r'''
    Just the space component of :class:`~.TorusWrapping`.
    '''

    @staticmethod
    def default(S, TorusWrapping):
        return TorusWrapping[1]
