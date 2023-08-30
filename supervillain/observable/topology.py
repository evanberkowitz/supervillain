#!/usr/bin/env python

import numpy as np

from supervillain.observable import Observable

class WindingSquared(Observable):
    r'''
    Given periodic boundary conditions the total topological charge vanishes $\partial_J Z = 0$.
    Translational invariance is strong enough to conclude that in expectation the winding number on any plaquette also vanishes.

    However we can treat $J$ as a local source and ask about the square of the winding number on any plaquette

    .. math ::
        w_p^2 = -\frac{\delta^2 \log Z}{\delta J_p^2}

    and we can drop the quantum-disconnected pieces because $\delta Z / \delta J_p = 0$ for any $p$ when $J=0$.

    To increase statistics we calculate the local winding number squared and average over the lattice

    .. math ::
       \texttt{WindingSquared} = \frac{1}{\Lambda} \sum_p w_p^2

    where $\Lambda$ is the number of plaquettes in the lattice.

    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        Differentiating with respect to $J_p$ gives a $-i\,dn_p$; differentiating twice gives $-dn_p^2$, so that $w_p = dn_p^2$.
        '''

        L = S.Lattice
        return np.mean(L.d(1, n)**2)


class Winding_Winding(Observable):
    r'''
    Beyond just the :class:`same-site-squared <WindingSquared>` we can compute correlations of the plaquette winding number.

    .. math ::
        W_{p,q} = -\frac{\delta^2 \log Z}{\delta J_p \delta J_q}

    By translation invariance we can average so that only the separation remains

    .. math ::
        W_{\Delta p} = \frac{1}{\Lambda} \sum_{p} W_{p, p-\Delta p}.

    .. note ::
        You can check that $\texttt{Winding_Winding[0,0]} = \texttt{WindingSquared}$ configuration by configurations.

    '''

    @staticmethod
    def Villain(S, phi, n):
        r'''
        Differentiating twice gives $W_{p,q} = \left\langle dn_p dn_q \right\rangle$; the quantum-disconnected piece vanishes when $J=0$.
        '''

        L = S.Lattice
        dn = L.d(1, n)
        return L.correlation(dn, dn)

