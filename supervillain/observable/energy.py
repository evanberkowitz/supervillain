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
            \mathcal{S} &= \left\langle \partial_\kappa S \right\rangle = - \frac{1}{2\kappa^2} \sum_{\ell} m_\ell^2 + \frac{|\ell|}{2 \kappa}.
           \end{align}

        '''
        
        L = S.Lattice
        return (L.links / 2 - 0.5 / S.kappa * (m**2).sum()) / (L.sites)

class InternalEnergyDensitySquared(Observable):
    r'''If we think of $\kappa$ as a thermodynamic $\beta$, then we
     may compute the expectation value of the square of the internal
     energy density $U$ as

    .. math::
        \begin{align}
        \langle U^2 \rangle &= \frac{1}{\Lambda^2} \frac{1}{Z} \partial^2_\kappa  Z
        \end{align}

    where $\Lambda$ is the number of sites in our spacetime. 
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` case differentiating by $\kappa$ gives the energy.
        '''

        L = S.Lattice
        return (S(phi, n) / (L.sites * S.kappa))**2


    @staticmethod
    def Worldline(S, m):
        r'''
        In the :class:`~.Worldline` formulation we differentiate to find

        .. math ::
           \begin{align}
            \langle U^2 \rangle &= \frac{1}{Z}\frac{1}{\Lambda^2} \sum_{i} \left[(E_{i}(\beta)-\beta E_{i}'(\beta))^2 -2 E'_{i}(\beta) - \beta E_{i}''(\beta)\right] e^{-\beta E_{i}(\beta)}
           \end{align}
        
        where
        .. math ::
       E_{i}(\beta) =  \sum_{\ell} \left[\frac{1}{2\beta^2} +\frac{1}{2
       \beta} \log(2\pi \beta) \right]

       is the effective worldline `energy'. 
        '''

        L = S.Lattice
        beta = S.kappa
        en = 0.5 * (m**2).sum() / (beta**2) + L.links * 0.5 * np.log(2*np.pi*beta)/beta
        enPrime = -1.0 * (m**2).sum()/(beta**3 )+ L.links * 0.5 * (1 - np.log(2 * np.pi * beta))/(beta ** 2)
        enPrimePrime = 3.0 * (m**2).sum()/(beta ** 4) + L.links * 0.5 *( -3.0  + 2.0 *  np.log(2 * np.pi * beta))/(beta ** 3 )
        return ( (en + beta * enPrime)**2 - 2.0 * enPrime  - beta * enPrimePrime )/(L.sites ** 2)