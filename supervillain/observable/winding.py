#!/usr/bin/env python

import numpy as np

from supervillain.observable import Scalar, Observable
from supervillain.lattice import d, delta

class WindingSquared(Scalar, Observable):
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
    def Villain(S, n):
        r'''
        Differentiating with respect to $J_p$ gives a $-i\,dn_p$; differentiating twice gives $-dn_p^2$, so that $w_p = dn_p^2$.
        '''

        if S.Lattice.D < 2:
            raise NotImplementedError('Winding observables require $D \\geq 2$; there are no plaquettes for $D < 2$.')
        return np.mean(d(n)**2)

    @staticmethod
    def Worldline(S, Links):
        r'''
        Reusing the derivation from :class:`~.Winding_Winding` and setting $p=q$ we get

        .. math::

            w_p =  \frac{1}{\pi^2 \kappa}-\frac{1}{(2\pi \kappa)^2}\left\langle [d(m-\delta v/W)]_p^2 \right\rangle.

        because $\delta / \delta J_p ( d \delta J_p) = 4$ (the orientation-averaged diagonal of $d\delta$, in any $D$).
        '''
        if S.Lattice.D < 2:
            raise NotImplementedError('Winding observables require $D \\geq 2$; there are no plaquettes for $D < 2$.')
        return 1/(np.pi**2 * S.kappa)-np.mean(d(Links)**2) / (2*np.pi*S.kappa)**2

class Winding_Winding(Observable):
    r'''
    Beyond just the :class:`same-site-squared <WindingSquared>` we can compute correlations of the plaquette winding number.

    .. math ::
        W_{p,q} = -\frac{\delta^2 \log Z}{\delta J_p \delta J_q}

    By translation invariance we can average so that only the separation remains

    .. math ::
        W_{\Delta p} = \frac{1}{\Lambda} \sum_{p} W_{p, p-\Delta p}.

    .. note ::
        You can check that $\texttt{Winding\_Winding}$ at the :py:attr:`~supervillain.lattice.Lattice.origin` equals $\texttt{WindingSquared}$ configuration by configuration.

    For $D > 2$ the plaquette winding $dn$ is a 2-form with $\binom{D}{2}$ orientations.  By isotropy we average over them, so $\texttt{Winding\_Winding}$ is the orientation-averaged same-orientation correlator $\frac{1}{\binom{D}{2}} \sum_c \langle dn_c\, dn_c\rangle$ (a single correlator when $D=2$).

    '''

    # Cache of the J-independent dδ contact stencil, keyed by (D, N).
    _stencil = dict()

    @staticmethod
    def Villain(S, n):
        r'''
        Differentiating twice gives $W_{p,q} = \left\langle dn_p dn_q \right\rangle$; the quantum-disconnected piece vanishes when $J=0$.  For $D>2$ we average the same-orientation correlator over the $\binom{D}{2}$ plaquette orientations.
        '''

        L = S.Lattice
        if L.D < 2:
            raise NotImplementedError('Winding observables require $D \\geq 2$; there are no plaquettes for $D < 2$.')
        dn = d(n)
        return L.correlation(dn, dn).mean(axis=0)

    @staticmethod
    def Worldline(S, Links):
        r'''
        .. collapse :: The Worldline observable is trickier.
            :class: note:

            Expanding the :class:`~.Worldline` action (and grouping the irrelevant contants)

            .. math::
                \begin{aligned}
                S_J[m, v]
                    &= \frac{1}{2\kappa} \sum_\ell \left((m-\delta v/W) - \frac{\delta J}{2\pi}\right)_\ell^2 + \text{constants}
                    \\
                    &= \frac{1}{2\kappa} \sum_\ell \left((m-\delta v/W)_\ell^2 - \frac{1}{\pi} (m-\delta v/W)_\ell (\delta J)_\ell + \frac{1}{4\pi^2} (\delta J)_\ell^2 \right)+ \text{constants}
                    \\
                    &= \frac{1}{2\kappa} \left(\sum_\ell (m-\delta v/W)_\ell^2 + \sum_p - \frac{1}{\pi} (d(m-\delta v/W))_p J_p + \frac{1}{4\pi^2} J_p (d \delta J)_p \right)+ \text{constants}
                \end{aligned}

            where we integrated two terms by parts to get $J$ undecorated.

            Differentiating $\log Z$ once gives

            .. math::
                \begin{aligned}
                \frac{\delta}{\delta J_p} \log Z
                    &= -\frac{1}{Z} \sum Dm\; Dv\; [\delta m = 0] e^{-S_J[m, v]} \frac{\delta S}{\delta J_p}
                    \\
                    &= \frac{1}{Z} \sum Dm\; Dv\; [\delta m = 0] e^{-S_J[m, v]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (d(m-\delta v/W))_p + \frac{2}{4\pi^2} (d\delta J)_p \right)
                \end{aligned}

            where the factor of 2 on the $d \delta J$ term comes from the fact that $J d \delta J$ is quadratic in $J$.

            Differentiating again we must hit both the $1/Z$ term and the path integral upstairs giving two terms,

            .. math::
                \begin{aligned}
                \frac{\delta}{\delta J_q}\frac{\delta}{\delta J_p} \log Z
                    =& \frac{\delta}{\delta J_q} \left[\frac{1}{Z} \sum Dm\; Dv\; [\delta m = 0] e^{-S_J[m,v]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (d(m-\delta v/W))_p + \frac{1}{2\pi^2} (d\delta J)_p \right) \right]
                    \\
                    =& -\frac{1}{Z^2} \left[\sum Dm\; Dv\; [\delta m = 0] e^{-S_J[m,v]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (d(m-\delta v/W))_p + \frac{1}{2\pi^2} (d\delta J)_p \right) \right]
                    \\
                    & \phantom{-\frac{1}{Z^2}}\times \left[\sum Dm\; Dv\; [\delta m = 0] e^{-S_J[m,v]} \frac{-1}{2\kappa}\left( -\frac{1}{\pi} (d(m-\delta v/W))_q + \frac{1}{2\pi^2} (d\delta J)_q \right) \right]
                    \\
                    &+ \frac{1}{Z} \sum Dm\; Dv\; [\delta m = 0] e^{-S_J[m,v]} \Bigg[
                    \\
                    & \phantom{-\frac{1}{Z^2}} \left(\frac{-1}{2\kappa}\right)^2\left( -\frac{1}{\pi} (d(m-\delta v/W))_q + \frac{1}{2\pi^2} (d\delta J)_q \right)
                    \\
                    & \phantom{-\frac{1}{Z^2}} \times\left( -\frac{1}{\pi} (d(m-\delta v/W))_p + \frac{1}{2\pi^2} (d\delta J)_p \right)
                    \\
                    & \phantom{-\frac{1}{Z^2}} +\frac{-1}{4\pi^2 \kappa}\frac{\delta}{\delta J_q} (d \delta J)_p 
                    \Bigg]
                \\
                \end{aligned}

            In the second term we differentiate both the action [the first term in the brackets] and what was already brought down [the second].
            Plugging in $J=0$ gives

            .. math ::
                \begin{aligned}
                -\left.\frac{\delta}{\delta J_q}\frac{\delta}{\delta J_p} \log Z \right|_{J=0}
                = \left(\frac{1}{2\pi \kappa}\right)^2\Bigg\{&
                \left\langle (d(m-\delta v/W))_p \right\rangle \left\langle (d(m-\delta v/W))_q \right\rangle
                \\
                &- \left\langle (d(m-\delta v/W))_p (d(m-\delta v/W))_q \right\rangle
                \\
                & + \kappa \left.\frac{\delta}{\delta J_q} (d \delta J)_p\right|_{J=0} \Bigg\}
                \end{aligned}

            because $d\delta 0 = 0$.

            When $J=0$ the first (quantum-disconnected) term is proportional $\left\langle d(m-\delta v/W)_p \right\rangle \left\langle d(m-\delta v/W)_q \right\rangle$ and the individual expectation values vanish by symmetry so we are left with

            .. math ::
                \begin{aligned}
                -\left.\frac{\delta}{\delta J_q}\frac{\delta}{\delta J_p} \log Z \right|_{J=0}
                = \left(\frac{1}{2\pi \kappa}\right)^2\Bigg\{&
                \kappa \left.\frac{\delta}{\delta J_q} (d \delta J)_p  \right|_{J=0}
                - \left\langle (d(m-\delta v/W))_p (d(m-\delta v/W))_q \right\rangle\Bigg\}
                \end{aligned}

            The remaining functional derivative is a displacement-dependent constant.


        At the end of the day all we are left with is

        .. math::
            - \frac{\delta}{\delta J_p} \frac{\delta}{\delta J_q}\log Z = \frac{1}{(2\pi \kappa)^2}\left\{\kappa\left.\frac{\delta}{\delta J_q}(d \delta J_p)\right|_{J=0} - \left\langle (d(m-\delta v/W))_p (d(m-\delta v/W)_q) \right\rangle\right\}

        when $J=0$.  In 2D $\left.\frac{\delta}{\delta J_q}(d \delta J_p)\right|_{J=0} = 4 \delta_{pq} - \sum_{\hat{\mu}} \delta_{p+\hat{\mu},q}$ where $\hat{\mu}$ runs over the 4 cardinal directions, reproducing (minus) the standard `2D five-point Laplacian stencil <https://en.wikipedia.org/wiki/Five-point_stencil#In_two_dimensions>`_.

        In 2D the plaquette is the top cell, so $d\,dn = 0$ and $d\delta$ coincides with the Hodge Laplacian.  In $D>2$ that is no longer true: $d\delta$ mixes the $\binom{D}{2}$ plaquette orientations, and only its orientation *diagonal* $(d\delta)_{cc}$ pairs with the same-orientation correlator.  We therefore average that diagonal over orientations (its $J=0$ origin value is $4$ in any $D$).
        '''
        L = S.Lattice
        if L.D < 2:
            raise NotImplementedError('Winding observables require $D \\geq 2$; there are no plaquettes for $D < 2$.')
        kappa = S.kappa
        dm = d(Links)

        # δ/δJ_q(dδJ)_p is J-independent: the orientation-diagonal of dδ, which depends only
        # on the lattice (D, N), not on the configuration.  We measure the orientation-averaged
        # same-orientation correlator, so we use the orientation-average of (dδ)_cc: put a unit
        # source on component c at the origin, apply dδ, read back component c, average over c.
        # The stencil is the same for every measurement on a given lattice, so we cache it.
        key = (L.D, L.N)
        try:
            contact = Winding_Winding._stencil[key]
        except KeyError:
            orientations = len(L.components[2])
            contact = np.zeros(L.dims)
            for c in range(orientations):
                source = L.form(2)
                source[c][L.origin] = 1.
                contact += np.asarray(d(delta(source)))[c]
            contact /= orientations
            Winding_Winding._stencil[key] = contact

        return (kappa * contact - L.correlation(dm, dm).mean(axis=0)) / (2*np.pi*kappa)**2

