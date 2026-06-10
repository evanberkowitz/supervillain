#!/usr/bin/env python

import numpy as np

from supervillain.compact.compact import d
from supervillain.generator import Generator
import supervillain.generator.combining as combining


class SiteUpdate(Generator):
    r'''
    Checkerboarded single-site :math:`\phi` updates using compact :meth:`~supervillain.compact.Form.face_sum`.
    '''

    def __init__(self, action, interval_phi=np.pi, rng=None):
        self.Action = action
        self.Lattice = action.Lattice
        self.interval_phi = interval_phi
        self.rng = np.random.default_rng(rng)

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def step(self, cfg):
        S = self.Action
        L = self.Lattice
        y = {key: value.copy() for key, value in cfg.items()}

        self.sweeps += 1
        total_accepted = 0
        total_acceptance = 0

        for color in L.checkerboarding:
            change_phi = L.zeros(0)
            change_phi[0, *color] = self.rng.uniform(
                -self.interval_phi, +self.interval_phi, len(color[0]),
            )

            change_S_local = (
                S.local(y['phi'] + change_phi, y['n'])
                - S.local(y['phi'], y['n'])
            ).face_sum()

            acceptance = np.clip(np.exp(-change_S_local[:, *color]), a_min=0, a_max=1)
            metropolis = self.rng.uniform(0, 1, size=acceptance.shape)
            accepted = metropolis < acceptance

            total_accepted += accepted.sum()
            total_acceptance += acceptance.sum()

            change_phi[0, *color] *= accepted[0]
            y['phi'] += change_phi

        sites = self.Action.Lattice2D.sites
        self.proposed += sites
        self.acceptance += total_acceptance / sites
        self.accepted += total_accepted

        return y

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'compact SiteUpdate: {self.accepted}/{self.proposed} accepted'
            f' ({self.accepted/self.proposed:.6f});'
            f' avg Metropolis {self.acceptance/self.sweeps:.6f}.'
        )


class LinkUpdate(Generator):
    r'''
    Parallel link :math:`n` updates (independent per link, fixed :math:`\phi`).
    '''

    def __init__(self, action, interval_n=1, rng=None):
        self.Action = action
        self.Lattice = action.Lattice
        self.interval_n = interval_n
        self.rng = np.random.default_rng(rng)
        self.n_changes = tuple(
            n for n in range(-interval_n, 0)
        ) + tuple(
            n for n in range(1, interval_n + 1)
        )

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def step(self, cfg):
        S = self.Action
        y = {key: value.copy() for key, value in cfg.items()}

        self.sweeps += 1

        change_n = self.Action.W * self.rng.choice(
            self.n_changes,
            size=y['n'].shape,
        )
        dphi = d(y['phi'])
        n = y['n']
        dS = (
            2 * np.pi * S.kappa * change_n
            * (dphi + 2 * np.pi * n + np.pi * change_n)
        )

        acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, size=acceptance.shape)
        accepted = metropolis < acceptance

        self.acceptance += acceptance.mean()
        self.accepted += accepted.sum()
        self.proposed += int(np.prod(y['n'].shape))

        y['n'] += np.where(accepted, change_n, 0)

        return y

    def inline_observables(self, steps):
        return {}

    def report(self):
        return (
            f'compact LinkUpdate: {self.accepted}/{self.proposed} accepted'
            f' ({self.accepted/self.proposed:.6f});'
            f' avg Metropolis {self.acceptance/self.sweeps:.6f}.'
        )


def MinimalHammer(action, *, rng=None):
    r'''Site + link updates — matches production ``SiteUpdate`` + ``LinkUpdate`` structure.'''
    if rng is None:
        return combining.Sequentially((
            SiteUpdate(action),
            LinkUpdate(action),
        ))
    seed = np.random.SeedSequence(rng)
    site_seed, link_seed = seed.spawn(2)
    return combining.Sequentially((
        SiteUpdate(action, rng=site_seed),
        LinkUpdate(action, rng=link_seed),
    ))
