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

    restricted to configurations obeying the *no-intersection constraint* that the
    topological-charge density $q$ vanish on every hypercube,

    .. math::
       q = (dn \wedge dn) = 0 \quad\text{on every hypercube.}

    The constraint is the path integral of a Lagrange-multiplier top-form
    $\theta$ entering the action as $S = S_{\text{Villain}} + i\,\theta\,(dn\wedge dn)$;
    $\theta$ is never sampled, so the field content is the Villain content
    $\{\phi, n\}$.  Because $dn\wedge dn$ is a 4-form the model is only defined
    (and only interesting --- it carries a mixed axial-vector-vector anomaly) in
    $D = 4$, which this class hard-assumes.

    Parameters
    ----------
    lattice: supervillain.lattice.Lattice
        A four-dimensional lattice with $N \geq 3$ on which $\phi$ and $n$ live.  $N = 2$ is
        rejected: a single link's charge response is a $2^4$ hypercube block, which at $N = 2$
        fills the whole lattice, so the local update moves lose their locality.
    kappa: float
        The $\kappa$ in the overall coefficient.
    '''

    def __init__(self, lattice, kappa):
        if not isinstance(lattice, Lattice):
            raise TypeError(f'NoIntersections requires a supervillain.lattice.Lattice, got {type(lattice).__name__}')
        if lattice.D != 4:
            raise ValueError(f'The No-Intersection model is only defined in D = 4, got D = {lattice.D}.')
        if lattice.N < 3:
            raise ValueError(
                f'The No-Intersection model needs N >= 3, got N = {lattice.N}.  A single link\'s '
                'charge response spans a 2^4 block of hypercubes, so at N = 2 it wraps around the '
                'whole lattice: the local moves lose locality and the two diagonal move families '
                'collapse (mod 2, e_mu - e_nu and e_mu + e_nu coincide).')
        super().__init__(lattice, kappa, W=1)

    def __str__(self):
        return f'NoIntersections({self.Lattice}, κ={self.kappa})'

    def valid(self, configuration):
        r'''
        Returns true if the no-intersection constraint $q = dn \wedge dn = 0$ holds on
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

    def admissible_symmetries(self):
        r'''
        Which factors of the hypercubic-torus space group (see
        :ref:`the lattice symmetries <space-group>`) this action admits, as
        keyword arguments for
        :meth:`~supervillain.generator.symmetry.Symmetry.all` /
        :meth:`~supervillain.generator.symmetry.Symmetry.draw`.

        .. warning ::
            A genuine OVERRIDE of :meth:`Villain.admissible_symmetries
            <supervillain.action.Villain.admissible_symmetries>`, not the
            inherited answer, even though ``NoIntersections`` *is* a
            ``Villain`` --- the action is invariant under the full space
            group exactly as :class:`~supervillain.action.Villain` is, but
            the no-intersection constraint $q = dn \wedge dn = 0$ is built
            from a CUP PRODUCT, natural only under order-preserving cubical
            maps.  A reflection carries $\cup$ into the opposite cup
            product, which differs by a coboundary and moves $q$ hypercube
            by hypercube even though $q \equiv 0$ on average survives.
            Measured on 25 configurations of the $N=6$ transport ensemble:
            every single, double, and triple axis flip breaks $q \equiv 0$
            on 25 of 25; translations, permutations, and the full inversion
            $x \to -x$ break it on none.

        .. note ::
            $\{I, -I\}$ is a genuine subgroup ($-I$ is central), so the
            restricted set is still a group: uniform draws, closure under
            inverses, and detailed balance all survive.  What is lost is
            reach, not correctness.

        Returns
        -------
        dict
            Keyword arguments ``translations``, ``flip_sets``, and ``perms``
            for :class:`~supervillain.generator.symmetry.Symmetry`, with
            ``flip_sets`` restricted to the identity and the full inversion
            $\{(), (0, \ldots, D-1)\}$.
        '''
        D = self.Lattice.D
        return dict(translations=True, flip_sets=[(), tuple(range(D))], perms=True)
