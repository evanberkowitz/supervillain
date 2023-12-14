from supervillain.observable import Scalar, Observable, DerivedQuantity
import numpy as np

class ActionDensity(Scalar, Observable):
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
    def Worldline(S, Links):
        r'''
        In the :class:`~.Worldline` formulation we differentiate to find

        .. math ::
           \begin{align}
            \mathcal{S} &= \left\langle \kappa \partial_\kappa S \right\rangle = - \frac{1}{2\kappa} \sum_{\ell} (m-\delta v/W)_\ell^2 + \frac{|\ell|}{2}.
           \end{align}

        '''
        
        L = S.Lattice
        return (L.links / 2 - 0.5 / S.kappa * (Links**2).sum()) / (L.sites)

class ActionTwoPoint(Observable):
    r'''
    In :class:`~.Action_Action` we need the translation-averaged

    .. math ::
        S^2_{x,y}
        =
        \left.\left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) -  \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S - \delta_{xy} \kappa_x \partial_{\kappa_x} S \right\rangle\right|_{\kappa_{x,y} = \kappa}

    given by

    .. math ::
        \texttt{ActionTwoPoint} = \frac{1}{\Lambda} \sum_x S^2_{x, x-\Delta x}.
    '''

    @staticmethod
    def Villain(S, Links):
        r'''
        In the :class:`~.Villain` formulation one finds

        .. math ::
            \left.\kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S\right|_{\kappa_{x,y} = \kappa} = 0

        while

        .. math ::
            \left.\delta_{xy} \kappa_x \partial_{\kappa_x} S \right|_{\kappa_x = \kappa}
            =
            \delta_{xy} \frac{\kappa}{2} \sum_{\ell \text{ from } x} (d\phi - 2\pi n)_\ell^2

        and

        .. math ::
            \left.(\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S)\right|_{\kappa_{x,y} = \kappa}
            =
            \left(\frac{\kappa}{2} \sum_{\ell \text{ from }y} (d\phi - 2\pi n)^2_{\ell}\right)
            \left(\frac{\kappa}{2} \sum_{\ell \text{ from }x} (d\phi - 2\pi n)^2_{\ell}\right)

        '''
        
        L = S.Lattice
        density = 0.5 * S.kappa * (Links**2).sum(axis=0)

        result = L.correlation(density, density)

        # The averaging over x of the δ term just modifies Δx=0.
        # We can simplify 1/Λ ∑_x δ_{x,x-Δx} f_x = δ_{Δx, 0} 1/Λ ∑_x f_x which means the Δx=0
        # piece of the correlator needs adjustment by the average f.
        # In this case f = density.

        result[0,0] -= density.mean()

        return result

    @staticmethod
    def Worldline(S, Links):
        r'''
        In the :class:`~.Worldline` formulation one has to carefully treat the $|\ell|/2 \log 2\pi \kappa$ contribution.
        We should really imagine $|\ell|/2$ as arising from a sum over sites of independent $\log 2\pi \kappa$s.
        That term contributes constant pieces,

        .. math ::
            \left.(\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S)\right|_{\kappa_{x,y} = \kappa}
            =
            \left(1 - \frac{1}{2\kappa} \sum_{\ell \text{ from } y} (m-\delta v/W)_\ell^2 \right)
            \left(1 - \frac{1}{2\kappa} \sum_{\ell \text{ from } x} (m-\delta v/W)_\ell^2 \right)

        while
        
        .. math ::
            \left.\delta_{xy} \kappa_x \partial_{\kappa_x} S \right|_{\kappa_x = \kappa}
            =
            \delta_{xy} \left(
                1 - \frac{1}{2\kappa} \sum_{\ell \text{ from } x} (m-\delta v/W)_\ell^2
            \right)

        and

        .. math ::
            \left.\kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S\right|_{\kappa_{x,y} = \kappa}
            =
            \delta_{xy} \left(-1 + \frac{1}{\kappa} \sum_{\ell \text{ from } x} (m-\delta v/W)_\ell^2\right).

        '''
        
        L = S.Lattice
        kappa = S.kappa
        m_squared = (Links**2).sum(axis=0)

        derivative = 1 -0.5 / kappa * m_squared

        result = L.correlation(derivative, derivative)

        # The averaging over x of the δ terms just modifies Δx=0.
        # We can simplify 1/Λ ∑_x δ_{x,x-Δx} f_x = δ_{Δx, 0} 1/Λ ∑_x f_x which means the Δx=0
        # piece of the correlator needs adjustment by the average f.
        # In this case f = (m-δv/W)^2 / 2κ,
        # what is left from cancelling the local one-derivative against the two-derivative term.

        delta = m_squared / 2 / kappa

        result[0,0] -= delta.mean()

        return result

class Action_Action(DerivedQuantity):
    r'''
    If we imagine rewriting the actions' sums over links as a sum over sites and a sum over directions we can associate a value of κ with each site.
    Then we may compute the correlations of the action density by evaluating

    .. math::
        \begin{align}
            \mathcal{S}_{x,y} =& \left.\left(-\kappa_y \frac{\delta}{\delta \kappa_y}\right) \left(-\kappa_x \frac{\delta}{\delta \kappa_x}\right) \log Z\right|_{\kappa_{x,y} = \kappa}
            \\
            =& 
            \left.\left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) - \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S  - \kappa_y \delta_{yx} \partial_{\kappa_x} S \right\rangle\right|_{\kappa_{x,y} = \kappa}
            \\ &
            - \left.\left\langle \kappa_x \partial_{\kappa_x} S \right\rangle\right|_{\kappa_x = \kappa}
              \left.\left\langle \kappa_y \partial_{\kappa_y} S \right\rangle\right|_{\kappa_y = \kappa}.
        \end{align}

    Using translational invariance the quantum-disconnected piece is independent of $x$ and $y$ and can be replaced by $\left\langle\texttt{ActionDensity}\right\rangle^2$.
    So, we find the simplification

    .. math ::
        \begin{align}
            \mathcal{S}_{x,y} = 
            & 
            \left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) -  \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S - \delta_{xy} \kappa_x \partial_{\kappa_x} S\right\rangle
            \\ &
            - \left\langle \texttt{ActionDensity} \right\rangle^2
        \end{align}

    We define the spacetime-dependent correlator
    
    .. math ::
        S^2_{x,y}
        =
        \left\langle (\kappa_y \partial_{\kappa_y} S) (\kappa_x \partial_{\kappa_x} S) -  \kappa_y \kappa_x \partial_{\kappa_y} \partial_{\kappa_x} S - \delta_{xy} \kappa_x \partial_{\kappa_x} S \right\rangle

    so that $\mathcal{S}_{x,y} = S^2_{xy} - \left\langle \texttt{ActionDensity} \right\rangle^2$

    We can reduce to a function of a single relative coordinate,

    .. math ::
        \begin{align}
            \texttt{Action_Action}_{\Delta x} = \mathcal{S}_{\Delta x} = \frac{1}{\Lambda} \sum_{x} \mathcal{S}_{x, x-\Delta x}
        \end{align}


    and define a primary observable :class:`~.ActionTwoPoint`

    .. math ::
        \begin{align}
            \texttt{ActionTwoPoint}_{\Delta x} = \frac{1}{\Lambda} \sum_{x} S^2_{x,x-\Delta x}.
        \end{align}

    The quantum-disconnected term is $\Delta x$ independent, so

    .. math ::
        \texttt{Action_Action}_{\Delta x} = \texttt{ActionTwoPoint}_{\Delta x} - \left\langle \texttt{ActionDensity} \right\rangle^2.

    '''

    @staticmethod
    def default(S, ActionTwoPoint, ActionDensity):
        return ActionTwoPoint - ActionDensity**2
