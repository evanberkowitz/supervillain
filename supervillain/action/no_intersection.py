#!/usr/bin/env python

import numpy as np
from supervillain.action.villain import Villain
from supervillain.lattice import Lattice, d, wedge

import logging
logger = logging.getLogger(__name__)


class NoIntersections(Villain):
    r'''
    The No-Intersection model is the modified-Villain action

    .. math::
       S = \frac{\kappa}{2} \sum_\ell (d\phi - 2\pi n)_\ell^2

    restricted to configurations obeying the *no-intersection constraint*

    .. math::
       Q = (dn \wedge dn) = 0 \quad\text{on every hypercube.}

    The constraint is the path integral of a Lagrange-multiplier top-form
    $\theta$ entering the action as $S = S_\text{Villain} + i\,\theta\,(dn\wedge dn)$;
    $\theta$ is never sampled, so the field content is the Villain content
    $\{\phi, n\}$.  Because $dn\wedge dn$ is a 4-form the model is only defined
    (and only interesting --- it carries a mixed axial-vector-vector anomaly) in
    $D = 4$, which this class hard-assumes.

    Parameters
    ----------
    lattice: supervillain.lattice.Lattice
        A four-dimensional lattice on which $\phi$ and $n$ live.
    kappa: float
        The $\kappa$ in the overall coefficient.
    '''

    def __init__(self, lattice, kappa):
        if not isinstance(lattice, Lattice):
            raise TypeError(f'NoIntersections requires a supervillain.lattice.Lattice, got {type(lattice).__name__}')
        if lattice.D != 4:
            raise ValueError(f'The No-Intersection model is only defined in D = 4, got D = {lattice.D}.')
        super().__init__(lattice, kappa, W=1)

    def __str__(self):
        return f'NoIntersections({self.Lattice}, κ={self.kappa})'

    def valid(self, configuration):
        r'''
        Returns true if the no-intersection constraint $dn \wedge dn = 0$ holds on
        every hypercube.

        Parameters
        ----------
        configuration: dict
            A dictionary that at least contains ``n``.

        Returns
        -------
        bool:
            Is the constraint satisfied everywhere?
        '''
        dn = d(configuration['n'])
        return bool(np.isclose(wedge(dn, dn), 0).all())
