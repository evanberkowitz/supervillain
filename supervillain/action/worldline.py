#!/usr/bin/env python

import numpy as np
from supervillain.h5 import ReadWriteable
import supervillain.h5.extendable as extendable
from supervillain.configurations import Configurations

import logging
logger = logging.getLogger(__name__)

class Worldline(ReadWriteable):
    r'''
    The dual (worldline) action is

    .. math::
       \begin{align}
       Z[J] &= \sum Dm\; Dv\; e^{-S_J[m, v]} \left[\delta m = 0\right]
       \\
       S_J[m, v] &= \frac{1}{2\kappa} \sum_\ell \left(m - \delta\left(\frac{v}{W} + \frac{J}{2\pi} \right)\right)_\ell^2 + \frac{|\ell|}{2} \ln (2\pi \kappa) - |x| \ln 2\pi
       \end{align}

    In other words, it is a sum over all configurations where $\delta m$ vanishes on every site.

    This formulation has no obvious sign problem when $W\neq 1$, but maintaining the constraint $\delta m = 0$ requires a nontrivial algorithm.

    Parameters
    ----------
    lattice: supervillain.Lattice2D
        The lattice on which $m$ lives.
    kappa: float
        The $\kappa$ in the overall coefficient.
    W: int
        The winding symmetry is $\mathbb{Z}_W$.  If $W=1$ the vortices are completely unconstrained.


    '''

    def __init__(self, lattice, kappa, W=1):

        self.Lattice = lattice
        self.kappa = kappa
        self.W = W
        self._constant_offset = self.Lattice.links / 2 * np.log(2*np.pi*kappa) - self.Lattice.sites * np.log(2*np.pi)

    def __str__(self):
        return f'Worldline({self.Lattice}, κ={self.kappa}, W={self.W})'

    def valid(self, m):
        r'''
        Returns true if the constraint $[\delta m = 0]$ is satisfied everywhere and false otherwise.
        '''

        return (self.Lattice.delta(1, m) == 0).all()

    def __call__(self, m, v, **kwargs):
        r'''
        Parameters
        ----------
        m: np.ndarray
            An integer-valued 1-form.
        v: np.ndarray
            An integer-valued 2-form.

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
        return 0.5 / self.kappa * np.sum((m - self.Lattice.delta(2, v) / self.W)**2) + self._constant_offset

    def configurations(self, count):
        r'''
        Parameters
        ----------
        count: int
            How many configurations to return.

        Returns
        -------
        Configurations
            ``count`` configurations of a zeroed 1-form ``m`` a zeroed 2-form ``v``.
        '''

        return Configurations({
            'm': extendable.array(self.Lattice.form(1, count, dtype=int)),
            'v': extendable.array(self.Lattice.form(2, count, dtype=int)),
            })

    def equivalence_class_v(self, configuration):
        r'''
        The constrained model has a gauge symmetry $v \rightarrow v \pm W$ with the gauge-invariant combination $m-\delta v / W$.

        We can take any configuration and send

        .. math ::
            \begin{align}
                v &\rightarrow v + \lambda W
                &
                m &\rightarrow m - \delta \lambda
            \end{align}

        for integer $\lambda$.  We fix $\lambda$ on every plaquette so that after the transformation $v\in[0,W)$.

        .. seealso ::
            test/equivalence-class-v.py

        Parameters
        ----------
        configuration: dict
            A dictionary with a one-form ``m`` and two-form ``v``.

        Returns
        -------
        dict:
            A dictionary with the equivalent fields but with $v \in [0, W)$.

        '''

        L = self.Lattice

        return {
                'm': configuration['m'] - L.delta(2, np.floor_divide(configuration['v'], self.W)),
                'v': np.mod(configuration['v'], self.W),
        }

    def count_loops(self, cfg):
        cfgabs = np.abs(cfg)
        #Takes the absolute value of a configuration's links to determine where a worm perturbation is
        #This must be applied to a worm on a zero background field, (or just subtract out the background field from before the worm)
        #Here I am checking each site for whether it lies on the worm (has an adjacent non-zero link),
        #and if so then give it an integer value. This is basically just a unique identifier, so the integer value 
        #increments each time
        q = 1
        ledger = self.Lattice.form(0)
        for i in range(self.Lattice.dims[0]):
            for j in range(self.Lattice.dims[1]):
                if (np.array([cfgabs[0,i,j],cfgabs[1,i,j],cfgabs[0,i-1,j],cfgabs[1,i,j-1]])>0).any():
                    ledger[i,j]=q
                    q+=1

        oldledger = np.empty_like(ledger)
        while (ledger != oldledger).any():  #As long as we are not in a steady state
            oldledger = ledger.copy()
            for i in range(ledger.shape[0]):
                for j in range(ledger.shape[1]):
                    if ledger[i,j] != 0:
                        for adj in [1,2,3,4]:   #Sift through each non-zero point in the ledger and its 
                            #adjacent links
                            #[1,2,3,4] -> [East,N,W,S]
                            if adj == 1:
                                test_link = cfgabs[0,i,j]
                                if test_link > 0:
                                    #Here we check if the worm passes through a neighboring link, and if so we take
                                    #the maximum of the two points connected by that link and adopt that for both
                                    ledger[i,j] = max(ledger[i,j],ledger[self.Lattice.mod([i+1,j])[0],self.Lattice.mod([i+1,j])[1]])
                                    ledger[self.Lattice.mod([i+1,j])[0],self.Lattice.mod([i+1,j])[1]] = ledger[i,j]
                            if adj == 2:
                                test_link = cfgabs[1,i,j]
                                if test_link > 0:
                                    ledger[i,j] = max(ledger[i,j],ledger[self.Lattice.mod([i,j+1])[0],self.Lattice.mod([i,j+1])[1]])
                                    ledger[self.Lattice.mod([i,j+1])[0],self.Lattice.mod([i,j+1])[1]] = ledger[i,j]
                            if adj == 3:
                                test_link = cfgabs[0,i-1,j]
                                if test_link > 0:
                                    ledger[i,j] = max(ledger[i,j],ledger[self.Lattice.mod([i-1,j])[0],self.Lattice.mod([i-1,j])[1]])
                                    ledger[self.Lattice.mod([i-1,j])[0],self.Lattice.mod([i-1,j])[1]] = ledger[i,j]
                            if adj == 4:
                                test_link = cfgabs[1,i,j-1]
                                if test_link > 0:
                                    ledger[i,j] = max(ledger[i,j],ledger[self.Lattice.mod([i,j-1])[0],self.Lattice.mod([i,j-1])[1]])
                                    ledger[self.Lattice.mod([i,j-1])[0],self.Lattice.mod([i,j-1])[1]] = ledger[i,j]
        #Once the process reaches a steady state, ie. the maximums in all connected loops have proliferated
        #we can just count the number of unique maximums that we obtain (other than 0)
        return np.sum(np.unique(ledger) > 0)

