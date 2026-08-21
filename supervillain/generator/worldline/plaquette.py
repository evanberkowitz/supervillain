#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import delta

class PlaquetteUpdate(ReadWriteable, Generator):
    r'''
    Ref. :cite:`Gattringer:2018dlw` suggests a simple update scheme where the links surrounding a single plaquette are updated in concert so that the :class:`~.Worldline` constraint is maintained.

    The links on a plaquette are updated in the same (oriented) way, which guarantees that $\delta m$ is invariant before and after the update, so that if it started 0 everywhere so it remains.
    The coordinated change is randomly chosen from ±1.

    .. warning::
        HOWEVER this algorithm is not ergodic on its own.
        The issue is that no proposal can change the worldline :class:`~.TorusWrapping`.
        Instead, if you start cold with $m=0$, which has global wrapping of (0,0) you stay in the (0,0) sector.

    '''

    def __init__(self, action):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The PlaquetteUpdate requires the Worldline action.')
        self.Action = action
        self.accepted = 0
        self.proposed = 0
        self.rng = np.random.default_rng()
        self.acceptance = 0.

    def __str__(self):
        return f'PlaquetteUpdate'

    def step(self, cfg):
        r'''
        Performs a sweep over all (site, direction-pair) plaquettes in a randomized order.

        For each plaquette at site x with directions (μ, ν), the 4 boundary links of the
        plaquette are updated in concert (±Δm with orientation) and v[comp, x] is also
        updated. This preserves δm=0 because the boundary change is exact: δ(∂A)=0.
        Direction pairs are processed sequentially to avoid shared-link conflicts.
        '''

        kappa = self.Action.kappa
        W     = self.Action._W
        L     = self.Action.Lattice

        m = cfg['m'].copy()
        v = cfg['v'].copy()

        # Precompute f = m − δv/W; updated incrementally for each accepted change.
        f = m - delta(v) / W

        n_per_site  = len(L.components[2])   # C(D, 2) direction pairs
        n_proposals = L.sites * n_per_site

        change_m_all   = self.rng.choice([-1, +1],   n_proposals)
        change_v_all   = self.rng.choice([-1, 0, +1], n_proposals)
        metropolis_all = self.rng.uniform(0, 1,        n_proposals)

        idx = 0
        for here in np.random.permutation(L.coordinates):
            here_t = tuple(here)
            for mu, nu in L.components[2]:
                change_m_val = change_m_all[idx]
                change_v_val = change_v_all[idx]
                met          = metropolis_all[idx]
                idx += 1

                comp_idx = L.comp_index[2][(mu, nu)]
                e_mu     = np.zeros(L.D, dtype=int); e_mu[mu] = 1
                e_nu     = np.zeros(L.D, dtype=int); e_nu[nu] = 1
                x_mu     = tuple(L.mod(here + e_mu))
                x_nu     = tuple(L.mod(here + e_nu))

                # f at the 4 boundary links (orientations +, +, −, −):
                # (μ, here), (ν, here+êμ), (μ, here+êν), (ν, here)
                f1 = f[mu][here_t]
                f2 = f[nu][x_mu]
                f3 = f[mu][x_nu]
                f4 = f[nu][here_t]

                delta_f = change_m_val - change_v_val / W
                dS = delta_f / kappa * (f1 + f2 - f3 - f4 + 2 * delta_f)

                acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
                self.acceptance += acceptance

                if met < acceptance:
                    m[mu][here_t] += +change_m_val
                    m[nu][x_mu]   += +change_m_val
                    m[mu][x_nu]   += -change_m_val
                    m[nu][here_t] += -change_m_val
                    v[comp_idx][here_t] += change_v_val
                    # Propagate the f change to keep f = m − δv/W current.
                    f[mu][here_t] += +delta_f
                    f[nu][x_mu]   += +delta_f
                    f[mu][x_nu]   -= +delta_f
                    f[nu][here_t] -= +delta_f
                    self.accepted += 1

        self.proposed += n_proposals
        return cfg | {'m': m, 'v': v}

    def report(self):
        return (
                f'There were {self.accepted} single-plaquette proposals accepted of {self.proposed} proposed updates.'
                +'\n'+
                f'    {self.accepted/self.proposed:.6f} acceptance rate'
                +'\n'+
                f'    {self.acceptance / self.proposed :.6f} average Metropolis acceptance probability.'
            )
