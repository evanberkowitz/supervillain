#!/usr/bin/env python
r'''
Shared Villain observables in production ``Lattice2D`` layout.

Used to compare ensembles generated via production infrastructure vs compact
:class:`~supervillain.compact.Form` fields (convert with :mod:`supervillain.layout`
before calling these functions).
'''

import numpy as np


def links(lattice, phi, n):
    r'''Gauge-invariant link variables :math:`d\phi - 2\pi n`.'''
    return lattice.d(0, phi) - 2 * np.pi * n


def action(lattice, phi, n, kappa):
    return 0.5 * kappa * np.sum(links(lattice, phi, n)**2)


def action_density(lattice, phi, n, kappa, W=1):
    return action(lattice, phi, n, kappa) / lattice.cells_of_degree[0]


def internal_energy_density(lattice, phi, n, kappa, W=1):
    return action_density(lattice, phi, n, kappa, W) / kappa


def winding_squared(lattice, phi, n, kappa, W=1):
    r'''Matches :class:`~supervillain.observable.WindingSquared`.'''
    return np.mean(lattice.d(1, n)**2)


def links_squared_mean(lattice, phi, n, kappa, W=1):
    L = links(lattice, phi, n)
    return np.mean(L**2)



OBSERVABLES = {
    'ActionDensity': action_density,
    'InternalEnergyDensity': internal_energy_density,
    'WindingSquared': winding_squared,
    'LinksSquaredMean': links_squared_mean,
}


def measure_configuration(lattice, phi, n, kappa, W=1, *, names=None):
    names = names or OBSERVABLES.keys()
    return {
        name: OBSERVABLES[name](lattice, phi, n, kappa, W)
        for name in names
    }


def measure_ensemble(ensemble, *, names=None, bridge_from_form=False):
    r'''
    Per-draw values for each named observable.

    Parameters
    ----------
    bridge_from_form:
        If true, convert compact :class:`~supervillain.compact.Form` fields with
        :func:`~supervillain.layout.from_form` before measuring.
    '''
    import supervillain.layout as layout

    S = ensemble.Action
    lattice = getattr(S, 'Lattice2D', S.Lattice)
    names = list(names or OBSERVABLES.keys())
    columns = {name: [] for name in names}

    for cfg in ensemble.configuration:
        if bridge_from_form:
            phi = layout.from_form(cfg['phi'])
            n = layout.from_form(cfg['n'])
        else:
            phi = cfg['phi']
            n = cfg['n']
        row = measure_configuration(lattice, phi, n, S.kappa, S.W, names=names)
        for name in names:
            columns[name].append(row[name])

    return {name: np.asarray(columns[name]) for name in names}
