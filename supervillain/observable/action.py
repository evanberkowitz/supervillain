from supervillain.observable import Observable, DerivedQuantity
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

class ActionTwoPoint(Observable):
    r'''
    In :class:`~.Action_Action` we need the volume-averaged

    .. math ::
        S^2_{x,y}
        =
        \left.\left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) -  \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S \right\rangle\right|_{\kappa_x = \kappa_y = \kappa}.

    We compute 

    .. math ::
        \texttt{ActionTwoPoint} = \frac{1}{\Lambda} \sum_x S^2_{x, x-\Delta x}
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` formulation one finds

        .. math ::
            \left.\kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S\right|_{\kappa_x = \kappa_y = \kappa} = 0

        while

        .. math ::
            \left.(\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S)\right|_{\kappa_x = \kappa_y = \kappa}
            =
            \left(\frac{\kappa}{2} \sum_{\ell \text{ from }y} (d\phi - 2\pi n)^2_{\ell}\right)
            \left(\frac{\kappa}{2} \sum_{\ell \text{ from }x} (d\phi - 2\pi n)^2_{\ell}\right)

        '''
        
        L = S.Lattice
        density = 0.5 * S.kappa * ((L.d(0, phi) - 2*np.pi*n)**2).sum(axis=0)

        return L.correlation(density, density)

    @staticmethod
    def Worldline(S, m):
        r'''
        In the :class:`~.Worldline` formulation one has

        .. math ::
            \left.(\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S)\right|_{\kappa_x = \kappa_y = \kappa}
            =
            \left(- \frac{1}{2\kappa} \sum_{\ell \text{ from } y} m_\ell^2 + \frac{1}{2}\right)
            \left(- \frac{1}{2\kappa} \sum_{\ell \text{ from } x} m_\ell^2 + \frac{1}{2}\right)

        while

        .. math ::
            \left.\kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S\right|_{\kappa_x = \kappa_y = \kappa}
            =
            \delta_{xy} \left(\frac{1}{\kappa} \sum_{\ell \text{ from } x} m_\ell^2 - 1\right) 

        '''
        
        L = S.Lattice
        kappa = S.kappa
        m_squared = (m**2).sum(axis=0)

        derivative = 1 -0.5 / kappa * m_squared

        result = L.correlation(derivative, derivative)

        result[0,0] -= (m_squared / kappa - 1).mean()

        return result

class Action_Action(DerivedQuantity):
    r'''
    If we imagine rewriting the actions' sums over links as a sum over sites and a sum over directions we can associate a value of κ with each site.
    Then we may compute the correlations of the action density by evaluating

    .. math::
        \begin{align}
            \mathcal{S}_{x,y} = \left.\left(\kappa_y \frac{\delta}{\delta \kappa_y}\right) \left(\kappa_x \frac{\delta}{\delta \kappa_x}\right) \log Z\right|_{\kappa_{x,y} = \kappa} =
            & 
            \left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) - \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S  - \kappa_y \delta_{yx} \partial_{\kappa_x} S \right\rangle
            \\ &
            - \left\langle \kappa_x \partial_{\kappa_x} S \right\rangle \left\langle \kappa_y \partial_{\kappa_y} S \right\rangle.
        \end{align}

    Using translational invariance the quantum-disconnected piece is independent of $x$ and $y$ and can be replaced by $\left\langle\texttt{ActionDensity}\right\rangle^2$.
    The piece explicitly local can also be simplified $\left\langle \delta_{xy} \kappa_y \partial_{\kappa_x} S\right\rangle = \delta_{xy} \left\langle \texttt{ActionDensity} \right\rangle$ by translational invariance.
    So, we find the simplification

    .. math ::
        \begin{align}
            \mathcal{S}_{x,y} = 
            & 
            \left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) -  \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S \right\rangle
            \\ &
            - \delta_{xy} \left\langle \texttt{ActionDensity} \right\rangle
            - \left\langle \texttt{ActionDensity} \right\rangle^2
        \end{align}

    We define the spacetime-dependent correlator
    
    .. math ::
        S^2_{x,y}
        =
        \left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) -  \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S \right\rangle

    so that $\mathcal{S}_{x,y} = S^2_{xy} - \left\langle \texttt{ActionDensity} \right\rangle^2$

    We can reduce to a function of a single relative coordinate,

    .. math ::
        \begin{align}
            \texttt{Action_Action}_{\Delta x} = \mathcal{S}_{\Delta x} = \frac{1}{\Lambda} \sum_{x} \mathcal{S}_{x, x-\Delta x}
        \end{align}


    The last term is $\Delta x$ independent, and penultimate term only contributes to $\Delta x = 0$.

    '''

    @staticmethod
    def default(S, ActionTwoPoint, ActionDensity):
        result = ActionTwoPoint.copy()
        result[0,0] -= ActionDensity
        result -= ActionDensity**2

        return result
