#!/usr/bin/env python

import numpy as np
from supervillain.h5 import H5able
from supervillain.configurations import Configurations

import logging
logger = logging.getLogger(__name__)

class Villain(H5able):
    r'''
    'The' Villain action is just the straightforward

    .. math::
       \begin{align}
       Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S_J[\phi, n]}
       &
       S_J[\phi, n] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + i \sum_p J_p (dn)_p
       \end{align}

    with $\phi$ a real-valued 0-form that lives on sites, $n$ an integer-valued one form that lives on links $l$, and $J$ a two-form that lives on plaquettes $p$.

    In this formulation, if $J$ is real and nonzero we expect a sign problem because the action is complex.  However, we can think of $J$ as an external source, take functional derivatives to get observables, and then set $J$ to zero so that we only need sample according to the first term.

    Parameters
    ----------
    lattice: supervillain.lattice.Lattice2D
        The lattice on which $\phi$ and $n$ live.
    kappa: float
        The $\kappa$ in the overall coefficient.
    '''

    def __init__(self, lattice, kappa):

        self.Lattice = lattice
        self.kappa = kappa

    def __str__(self):
        return f'Villain({self.Lattice}, κ={self.kappa})'

    def __call__(self, phi, n, **kwargs):
        r'''
        Parameters
        ----------
        phi: np.ndarray
            A real-valued 0-form.
        n: np.ndarray
            An integer-valued 1-form.

        Returns
        -------
        float
            $S_0[\phi, n]$
        '''
        return self.kappa / 2 * np.sum((self.Lattice.d(0, phi) - 2*np.pi*n)**2)

    def configurations(self, count):
        r'''
        Parameters
        ----------
        count: int

        Returns
        -------
        dict
            A dictionary of zeroed arrays at keys ``phi`` and ``n``, holding ``count`` 0- and 1-forms respectively.
        '''
        return Configurations({
            'phi': self.Lattice.form(0, count),
            'n':   self.Lattice.form(1, count, dtype=int),
            })

class Worldline(H5able):
    r'''
    The dual (worldline) action is

    .. math::
       \begin{align}
       Z[J] &= \sum Dm\; e^{-S_J[m]} \left[\delta m = 0\right]
       &
       S_J[m] &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 
       \end{align}

    In other words, it is a sum over all configurations where $\delta m$ vanishes on every site.

    This formulation has no obvious sign problem when $J\neq 0 $, but maintaining the constraint $\delta m = 0$ requires a nontrivial algorithm.

    Parameters
    ----------
    lattice: supervillain.Lattice2D
        The lattice on which $m$ lives.
    kappa: float
        The $\kappa$ in the overall coefficient.


    '''

    def __init__(self, lattice, kappa):

        self.Lattice = lattice
        self.kappaa = kappa

    def __str__(self):
        return f'Worldline({self.Lattice}, κ={self.kappa})'

    def valid(self, m):
        r'''
        Returns true if the constraint $[\delta m = 0]$ is satisfied everywhere and false otherwise.
        '''

        return (self.Lattice.delta(m) == 0).all()

    def __call__(self, m, **kwargs):
        r'''
        Parameters
        ----------
        m: np.ndarray
            An integer-valued 1-form.

        Returns
        -------
        float:
            $S_0[m]$

        Raises
        ------
        ValueError
            If $m$ does not satisfy the constraint.
        '''

        if not self.valid(m):
            raise ValueError(f'The one-form m does not satisfy the constraint δm = 0 everywhere.')
        return 0.5 / self.kappa * np.sum(m**2)

    def configurations(self, count):
        r'''
        Parameters
        ----------
        count: int
            How many configurations to return.

        Returns
        -------
        dict
            A dictionary of zeroed arrays at key ``m`` holding ``count`` 1-forms.
        '''

        return Configurations({
            'm': self.Lattice.form(1, count, dtype=int),
            })
