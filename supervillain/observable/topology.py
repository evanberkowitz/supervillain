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

    @staticmethod
    def Worldline(S, m):
        r'''
        Reusing the derivation from :class:`~.Winding_Winding` and setting $p=q$ we get

        .. math::

            w_p =  \frac{1}{\pi^2 \kappa}-\frac{1}{(2\pi \kappa)^2}\left\langle (dm)_p^2 \right\rangle.
        '''
        return 1/(np.pi**2 * S.kappa)-np.mean(S.Lattice.d(1, m)**2) / (2*np.pi*S.kappa)**2

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

    @staticmethod
    def Worldline(S, m):
        r'''
        .. collapse :: The Worldline observable is trickier.
            :open:
            :class: note:

            Expanding the :class:`~.Worldline` action (and dropping the irrelevant contants)

            .. math::
                \begin{align}
                S_J[m]
                    &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 + \text{constants}
                    \\
                    &= \frac{1}{2\kappa} \sum_\ell \left(m_\ell^2 - \frac{1}{\pi} m_\ell (\delta J)_\ell + \frac{1}{4\pi^2} (\delta J)_\ell^2 \right)+ \text{constants}
                    \\
                    &= \frac{1}{2\kappa} \left(\sum_\ell m_\ell^2 + \sum_p - \frac{1}{\pi} (dm)_p J_p + \frac{1}{4\pi^2} J_p (d \delta J)_p \right)+ \text{constants}
                \end{align}

            where we integrated two terms by parts to get $J$ undecorated.

            Differentiating $\log Z$ once gives

            .. math::
                \begin{align}
                \frac{\delta}{\delta J_p} \log Z
                    &= \frac{1}{Z} \sum Dm\; [\delta m = 0] e^{-S_J[m]} \frac{\delta S}{\delta J_p}
                    \\
                    &= \frac{1}{Z} \sum Dm\; [\delta m = 0] e^{-S_J[m]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (dm)_p + \frac{1}{4\pi^2} (d\delta J)_p \right)
                \end{align}

            Differentiating again we must hit both the $1/Z$ term and the path integral upstairs giving two terms,

            .. math::
                \begin{align}
                \frac{\delta}{\delta J_q}\frac{\delta}{\delta J_p} \log Z
                    =& \frac{\delta}{\delta J_q} \left[\frac{1}{Z} \sum Dm\; [\delta m = 0] e^{S_J[m]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (dm)_p + \frac{1}{4\pi^2} (d\delta J)_p \right) \right]
                    \\
                    =& -\frac{1}{Z^2} \left[\sum Dm\; [\delta m = 0] e^{S_J[m]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (dm)_p + \frac{1}{4\pi^2} (d\delta J)_p \right) \right]
                    \\
                    & \phantom{-\frac{1}{Z^2}}\times \left[\sum Dm\; [\delta m = 0] e^{S_J[m]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (dm)_q + \frac{1}{4\pi^2} (d\delta J)_q \right) \right]
                    \\
                    &+ \frac{1}{Z} \sum Dm\; [\delta m = 0] e^{S_J[m]} \Bigg[
                    \\
                    & \phantom{-\frac{1}{Z^2}} \left(\frac{-1}{2\kappa}\right)^2\left( -\frac{1}{\pi} (dm)_q + \frac{1}{4\pi^2} (d\delta J)_q \right)\left( -\frac{1}{\pi} (dm)_p + \frac{1}{4\pi^2} (d\delta J)_p \right)
                    \\
                    & \phantom{-\frac{1}{Z^2}} +\frac{-1}{8\pi^2 \kappa}\frac{\delta}{\delta J_q} (d \delta J)_p 
                    \Bigg]
                \\
                \end{align}

            In the second term we differentiate both the action [the first term in the brackets] and what was already brought down [the second].
            Plugging in $J=0$ gives

            .. math ::
                \begin{align}
                -\left.\frac{\delta}{\delta J_q}\frac{\delta}{\delta J_p} \log Z \right|_{J=0}
                = \left(\frac{1}{2\pi \kappa}\right)^2\Bigg\{&
                \left\langle (dm)_p \right\rangle \left\langle (dm)_q \right\rangle
                \\
                &- \left\langle (dm)_p (dm)_q \right\rangle
                \\
                & + \frac{\kappa}{2} \frac{\delta}{\delta J_q} (d \delta J)_p \Bigg\}
                \end{align}

            because $d\delta 0 = 0$.

            When $J=0$ the first (quantum-disconnected) term is proportional $\left\langle dm_p \right\rangle \left\langle dm_q \right\rangle$ and the individual expectation values vanish by symmetry so we are left with.

            .. math ::
                \begin{align}
                -\left.\frac{\delta}{\delta J_q}\frac{\delta}{\delta J_p} \log Z \right|_{J=0}
                = \left(\frac{1}{2\pi \kappa}\right)^2\Bigg\{&
                \frac{\kappa}{2} \frac{\delta}{\delta J_q} (d \delta J)_p 
                - \left\langle (dm)_p (dm)_q \right\rangle\Bigg\}
                \end{align}

            Careful evaluation shows that $d \delta J =0$ when $J=0$ and that we ought to treat $d \delta J$ as independent from $J$, so that the functional derivative in the final term vanishes.


        At the end of the day all we are left with is

        .. math::
            - \frac{\delta}{\delta J_p} \frac{\delta}{\delta J_q}\log Z = \frac{1}{(2\pi \kappa)^2}\left\{\frac{\kappa}{2}\frac{\delta}{\delta J_q}(d \delta J) - \left\langle (dm)_p (dm_q) \right\rangle\right\}

        which simplifies if the functional derivative vanishes.
        '''
        L = S.Lattice
        kappa = S.kappa
        dm = L.d(1, m)
        return - L.correlation(dn, dn) / (2*np.pi*kappa)**2

