from supervillain.observable import Observable
import numpy as np

class ActionDensity(Observable):
    r'''The expectation value of the action density can be calculated as

    .. math::
        \begin{align}
        \mathcal{S} &= -  \kappa \partial_\kappa \log Z
        \\
          &= \left\langle - \kappa  \partial_\kappa (-S) \right\rangle
        \\
          &= \left\langle \kappa  \partial_\kappa S \right\rangle
        \end{align}

    It is extensive in the spacetime volume, so we calculate the density

    .. math ::
       \texttt{ActionDensity} =  \mathcal{S} / \Lambda

    where $\Lambda$ is the number of sites in our spacetime.
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` case differentiating and then multiplying by $\kappa$ gives the action $S_0$ itself!
        '''

        L = S.Lattice
        return S(phi, n) / (L.sites)


    @staticmethod
    def Worldline(S, m):
        r'''
        In the :class:`~.Worldline` formulation we differentiate to find

        .. math ::
           \begin{align}
            \mathcal{S} &= \left\langle \kappa \partial_\kappa S \right\rangle = - \frac{1}{2\kappa} \sum_{\ell} m_\ell^2 + \frac{|\ell|}{2}.
           \end{align}

        '''
        
        L = S.Lattice
        return (L.links / 2 - 0.5 / S.kappa * (m**2).sum()) / (L.sites)


