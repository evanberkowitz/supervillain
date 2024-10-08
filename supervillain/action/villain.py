#!/usr/bin/env python

import numpy as np
from supervillain.h5 import ReadWriteable
import supervillain.h5.extendable as extendable
from supervillain.configurations import Configurations

import logging
logger = logging.getLogger(__name__)

class Villain(ReadWriteable):
    r'''
    'The' Villain action is just the straightforward

    .. math::
       \begin{align}
       Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; Dv\; e^{-S_J[\phi, n, v]}
       \\
       S_J[\phi, n, v] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p \left(v/W + J/2\pi \right)_p (dn)_p
       \end{align}

    with $\phi$ a real-valued 0-form that lives on sites, $n$ an integer-valued one form that lives on links $l$, and $J$ a two-form that lives on plaquettes $p$.

    In this formulation, if $J$ is real and nonzero we expect a sign problem because the action is complex.  However, we can think of $J$ as an external source, take functional derivatives to get observables, and then set $J$ to zero so that we only need sample according to the first term.

    .. warning::
        Because $W\neq1$ suffers from a sign problem if we try to sample $v$, we assume an ensemble will be generated
        with a clever algorithm that that maintains :ref:`the winding constraint <winding constraint>`, so that $v$ need not be included in the field content.

    Parameters
    ----------
    lattice: supervillain.lattice.Lattice2D
        The lattice on which $\phi$ and $n$ live.
    kappa: float
        The $\kappa$ in the overall coefficient.
    W: int
        The constraint integer $W$.  When $W>1$ we must integrate out $v$ to avoid a horrible sign problem and sample carefully to maintain the winding constraint.
    '''

    def __init__(self, lattice, kappa, W=1):

        self.Lattice = lattice
        self.kappa = kappa
        self.W = W

    def __str__(self):
        return f'Villain({self.Lattice}, κ={self.kappa}, W={self.W})'

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
            $S_0[\phi, n]$.  We assume the path integration over $v$ implements :ref:`the winding constraint <winding constraint>` in some clever way,
            so the action does not depend on $v$.
        '''
        return self.kappa / 2 * np.sum((self.Lattice.d(0, phi) - 2*np.pi*n)**2) # + nothing that depends on v since we will implement the constraint directly.

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
            'phi': extendable.array(self.Lattice.form(0, count)),
            'n':   extendable.array(self.Lattice.form(1, count, dtype=int)),
            })

    def gauge_transform(self, configuration, k):
        r'''
        The Villain formulation has the gauge symmetry 

        .. math::
           \phi &\rightarrow\; \phi + 2\pi k
           \\
           n &\rightarrow\; n + dk

        with the gauge-invariant combination $(d\phi - 2\pi n)$.

        Parameters
        ----------
        configuration: dict
            A dictionary with the fields.
        k: np.array
            The gauge transformation parameter $k$ which must be of integer dtype.

        Returns
        -------
        dict:
            A dictionary with the fields transformed by $k$.
        '''

        if not issubclass(k.dtype.type, np.integer):
            raise ValueError('The gauge transformation is generated by integer k; it must be of integer dtype.')

        return {
            'phi': configuration['phi'] + 2*np.pi*k,
            'n':   configuration['n']   + self.Lattice.d(0, k),
        }

    def valid(self, configuration):
        r'''
        Returns true if the constraint $[dn \equiv 0 \text{ mod } W]$ is satisfied everywhere.

        When $W=\infty$, check that $dn = 0$ everywhere.

        Parameters
        ----------
        configuration: dict
            A dictionary that at least contains n.

        Returns
        -------
        bool:
            Is the constraint satisfied everywhere?
        '''

        n = configuration['n']
        zero = (np.mod(self.Lattice.d(1, n), self.W) if self.W < float('inf') else self.Lattice.d(1, n))
        return (zero == 0).all()
