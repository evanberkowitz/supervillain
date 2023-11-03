from supervillain.observable import Observable
import numpy as np

class Links(Observable):
    r'''
    Both the :class:`~.Villain` and :class:`~.Worldline` formulations have the notion of invariant links.
    The main purpose of ``Links`` is to ensure that we always write functions of the correct combination of raw fields.

    .. warning ::
        Unlike most observables these will NOT match${}^*$ between the different formulations.

        ${}^*$They both have 0 expectation value by the discrete rotational lattice symmetries, but they don't have related physical motivation; they are dual to one another under Poisson resummation.
    '''


    @staticmethod
    def Villain(S, phi, n):
        r'''
        The only gauge-invariant link variables in the :class:`~.Villain` formulation are

        .. math ::
            \texttt{Links}_{\ell} = (d\phi - 2\pi n)_\ell

        and these are what show up in any observables that talk to links.
        You can see that this combination arises in the action itself and in the :class:`~.ActionTwoPoint`, for example.
        Some observables, like the :class:`~.Spin_Spin` correlation function have 'bare' 0-form $\phi$s but they come in exponentials and are therefore gauge invariant.
        Others, like :class:`~.XWrapping` live on loops where the $d\phi$ integrates away but are gauge invariant.
        '''

        L = S.Lattice
        return L.d(0, phi) - 2*np.pi*n

    @staticmethod
    def Worldline(S, m, v):
        r'''
        In the :class:`~.action.Worldline` action we always have the combination

        .. math ::
            \texttt{Links}_{\ell} = (m - \delta v / W)_\ell

        and observables are generically functions of this combination, though there are some exceptional observables like :class:`~.TorusWrapping` where the $\delta v$ pieces cancel exactly.

        '''

        L = S.Lattice
        if S.W == 1:
            return m
        else:
            return m - L.delta(2, v) / S.W
