from supervillain.observable import Observable
import numpy as np

class Links(Observable):
    r'''
    Both the :class:`~.Villain` and :class:`~.Worldline` formulations have the notion of invariant links.

    .. warning ::
        Unlike most observables these will NOT match${}^*$ between the different formulations.

        ${}^*$They both have 0 expectation value by the discrete rotational lattice symmetries, but they don't have related physical motivation; they are dual to one another under Poisson resummation.
    '''


    @staticmethod
    def Villain(S, phi, n):
        r'''
        .. math ::
            \texttt{Links}_{\ell} = (d\phi - 2\pi n)_\ell
        '''

        L = S.Lattice
        return L.d(0, phi) - 2*np.pi*n

    @staticmethod
    def Worldline(S, m, v):
        r'''
        .. math ::
            \texttt{Links}_{\ell} = m

        '''

        L = S.Lattice
        if S.W == 1:
            return m
        else:
            return m - L.delta(2, v) / S.W
