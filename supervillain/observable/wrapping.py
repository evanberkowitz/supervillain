from supervillain.observable import Observable
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

        L = S.Lattice
        return n.sum(axis=(-2,-1))

    @staticmethod
    def Worldline(S, m):
        r'''
        The total wrapping in direction $\mu$ is given by the net particle flux in that direction

        .. math ::
            \texttt{TorusWrapping}_{\mu} = \frac{1}{|\mu|} \sum m_\mu

        where we divide by the length of the dimension because a torus-wrapping $m$ will get contributions from every μ-slice.
        '''

        return m.sum(axis=(-2,-1)) / S.Lattice.dims
