from supervillain.observable import Observable

class InternalEnergyDensity(Observable):
    r'''If we think of $\kappa$ like a thermodynamic $\beta$, then we may compute the internal energy $U$

    .. math::
        \begin{align}
        U &= - \kappa \partial_\kappa \log Z
        \\
          &= \left\langle - \kappa \partial_\kappa (-S) \right\rangle
        \\
          &= \left\langle \kappa \partial_\kappa S \right\rangle
        \end{align}

    It is extensive in the spacetime volume, so we calculate the density

    .. math ::
       \texttt{InternalEnergyDensity} =  U / \Lambda

    where $\Lambda$ is the number of sites in our spacetime.
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        In the :class:`~.Villain` case differentiating and then multiplying by $\kappa$ gives the action $S_0$ itself!
        '''

        L = S.Lattice
        return S(phi, n) / L.sites


    @staticmethod
    def Worldline(S, m):
        r'''
        In the :class:`~.Worldline` formulation we differentiate to find

        .. math ::
           \begin{align}
            U &= \left\langle \kappa \partial_\kappa S \right\rangle = - \frac{1}{2\kappa} \sum_{\ell} m_\ell^2 + \frac{|\ell|}{2}.
           \end{align}

        '''
        
        L = S.Lattice
        return L.links / 2 - 0.5 / self.kappa * np.sum(m**2)
