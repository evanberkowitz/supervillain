from supervillain.observable import Observable
import numpy as np

class InternalEnergyDensity(Observable):
    r'''If we think of $\kappa$ like a thermodynamic $\beta$, then we may compute the internal energy $U$

    .. math::
        \begin{align}
        U &= -  \partial_\kappa \log Z
        \\
          &= \left\langle -  \partial_\kappa (-S) \right\rangle
        \\
          &= \left\langle  \partial_\kappa S \right\rangle
        \end{align}

    It is extensive in the spacetime volume, so we calculate the density

    .. math ::
       \texttt{InternalEnergyDensity} =  U / \Lambda

    where $\Lambda$ is the number of sites in our spacetime.
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` case differentiating the action $S_0$ is the same as dividing it by $\kappa$!
        '''
        L = S.Lattice
        return S(phi, n) / (L.sites * S.kappa)


    @staticmethod
    def Worldline(S, m):
        r'''
        In the :class:`~.Worldline` formulation we differentiate to find

        .. math ::
           \begin{align}
            U &= \left\langle \partial_\kappa S \right\rangle = - \frac{1}{2\kappa^2} \sum_{\ell} m_\ell^2 + \frac{|\ell|}{2 \kappa}.
           \end{align}

        '''

        L = S.Lattice
        return (L.links / 2 - 0.5 / S.kappa * (m**2).sum()) / (L.sites * S.kappa)

class InternalEnergyDensitySquared(Observable):
    r'''
    If we think of $\kappa$ as a thermodynamic $\beta$, then we
    may compute the expectation value of the square of the internal
    energy $U$

    .. math::
        \begin{align}
        \langle U^2 \rangle &= \frac{1}{Z} \partial^2_\kappa  Z = \left\langle (\partial_\kappa S)^2 - \partial^2_\kappa S \right\rangle
        \end{align}

    and to the intensive density squared is

    .. math::
        \texttt{InternalEnergyDensitySquared} = \langle U^2 \rangle / \Lambda^2

    where $\Lambda$ is the number of sites in our spacetime. 
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` case,

        .. math ::
            \begin{align}
                \partial_\kappa S &= S/\kappa
                &
                \partial^2_\kappa S &= 0
            \end{align}
        '''

        L = S.Lattice
        return (S(phi, n) / (L.sites * S.kappa))**2


    @staticmethod
    def Worldline(S, m):
        r'''
        In the :class:`~.Worldline` formulation we differentiate to find

        .. math ::
            \begin{align}
                \partial_\kappa S &= - \frac{1}{2\kappa^2} \sum_{\ell} m_\ell^2 + \frac{|\ell|}{2 \kappa}
                &
                \partial^2_\kappa S &= \frac{1}{\kappa} \sum_{\ell} m_\ell^2 - \frac{|\ell|}{2\kappa^2}.
            \end{align}

        '''

        L = S.Lattice
        partial_kappa_S = (L.links / 2 - 0.5 / S.kappa * (m**2).sum()) / S.kappa
        partial_2_kappa_S = ((m**2).sum() - L.links / S.kappa) / S.kappa
        
        return (partial_kappa_S**2 - partial_2_kappa_S) / L.sites**2
