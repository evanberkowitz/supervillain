#!/usr/bin/env python

from supervillain.h5 import ReadWriteable


class Action(ReadWriteable):
    r'''
    An action assigns a weight to a configuration, declares what fields a
    configuration is made of, and says which of them are legal.  It lives on a
    :class:`~supervillain.lattice.Lattice`, and it knows which
    :ref:`lattice symmetries <space-group>` it admits.

    Nothing here is implemented.  This class records the surface that
    :class:`~supervillain.action.Villain`,
    :class:`~supervillain.action.Worldline`, and
    :class:`~supervillain.action.NoIntersections` all provide, so that
    generators and observables can be written against the interface rather
    than against a particular formulation.

    .. note ::
        It subclasses :class:`~supervillain.h5.ReadWriteable`, so an action
        that inherits from :class:`Action` is storable in HDF5 with the
        :class:`~supervillain.Ensemble` it generated without saying so
        separately.

    Attributes
    ----------
    Lattice: supervillain.lattice.Lattice
        The lattice the action lives on.  Every action has one; it is what
        fixes the dimension $D$ and the number of lattice sites in each direction $N$, and what the field degrees of freedom are reckoned against.
    '''

    Lattice = None

    def __call__(self, *fields, **kwargs):
        r'''
        The value of the action on a configuration.

        .. note ::
            The signature is deliberately left to each formulation, since the
            fields differ: :class:`~supervillain.action.Villain` takes
            $(\varphi, n)$ and :class:`~supervillain.action.Worldline` takes
            $(m, v)$.  Code that holds a configuration dictionary and does not
            know the formulation should call ``S(**configuration)``.  Every
            action must accept ``**kwargs``, so that entries a configuration carries
            beyond the fields --- inline observables, or an
            :ref:`importance weight <importance-weights>` --- are ignored.

        Returns
        -------
        float
            $S$ evaluated on the given fields.
        '''
        raise NotImplementedError

    def configurations(self, count):
        r'''
        A set of :class:`~supervillain.configurations.Configurations`, which declares the field content (the name, degree, and dtype of every :class:`~supervillain.lattice.Form`) and holds ``count`` copies of each.

        Parameters
        ----------
        count: int
            How many configurations the batch should hold.

        Returns
        -------
        supervillain.configurations.Configurations
        '''
        raise NotImplementedError

    def valid(self, configuration):
        r'''
        Whether a configuration provides the action's field contents and whether those fields satisfy the action's constraints.

        Parameters
        ----------
        configuration: dict
            At least the fields the action and its constraints are built from.

        Returns
        -------
        bool
        '''
        raise NotImplementedError

    def admissible_symmetries(self):
        r'''
        Which factors of the hypercubic-torus space group (see
        :ref:`the lattice symmetries <space-group>`) this action admits, as
        keyword arguments for
        :meth:`~supervillain.generator.symmetry.Symmetry.all` /
        :meth:`~supervillain.generator.symmetry.Symmetry.draw`.

        .. warning ::
            There is deliberately no default.  A :class:`~supervillain.generator.symmetry.Symmetry` can be applied
            as an always-accepted Monte Carlo move, so an element the action does not in
            fact admit corrupts the ensemble silently.
            An action that has not declared its symmetries therefore gets none, rather than being assumed to admit everything.

        Returns
        -------
        dict
            Suitable for ``Symmetry.all(lattice, **S.admissible_symmetries())``.
        '''
        raise NotImplementedError(
            f'{type(self).__name__} has not declared which lattice symmetries '
            'it admits.  Symmetry moves are always accepted, so assuming a '
            'group it does not admit would corrupt the ensemble with nothing '
            'to reveal it; declare admissible_symmetries instead.')
