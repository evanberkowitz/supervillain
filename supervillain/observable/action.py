from supervillain.observable import Observable

class ActionDensity(Observable):

    @staticmethod
    def Villain(S, phi, n):

        L = S.Lattice
        return S(phi, n) / L.sites


    @staticmethod
    def Worldline(S, m):
        
        L = S.Lattice
        return S(m) / L.sites
