from supervillain.observable import Observable

class ActionDensity(Observable):
    r'''The :ref:`action` fixes the Boltzmann weight of field configurations.
    It is extensive in the spacetime volume, so we calculate the density

    .. math ::
       \texttt{ActionDensity} = S / \Lambda

    where $\Lambda$ is the number of sites in our spacetime.
    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        Evaluates the :class:`~.Villain` action normalized to the spacetime volume.
        '''

        L = S.Lattice
        return S(phi, n) / L.sites


    @staticmethod
    def Worldline(S, m):
        r'''
        Evaluates the :class:`~.Worldline` action normalized to the spacetime volume.
        '''
        
        L = S.Lattice
        return S(m) / L.sites
