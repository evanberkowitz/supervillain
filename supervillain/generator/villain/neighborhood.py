#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.h5 import ReadWriteable
from supervillain.generator import Generator
from supervillain.lattice import d

import logging
logger = logging.getLogger(__name__)

class NeighborhoodUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update as :class:`NeighborhoodUpdateSlow <supervillain.generator.reference_implementation.villain.NeighborhoodUpdateSlow>` but is streamlined to eliminate calls, to calculate the change in action directly, and to avoid data movement.

    Proposals are drawn according to

    .. math ::

        \begin{aligned}
        \Delta\phi_x    &\sim \text{uniform}(-\texttt{interval\_phi}, +\texttt{interval\_phi})
        \\
        \Delta n_\ell   &\sim W \times [-\texttt{interval\_n}, +\texttt{interval\_n}]
        \end{aligned}

    We pick :math:`\Delta n_\ell` to be a multiple of the constraint integer $W$ so that if the adjacent plaquettes satisfy the :ref:`winding constraint <winding constraint>` $dn \equiv 0 \text{ mod }W$
    before the update they satisfy it after as well.

    Each site update changes $\phi$ at one site and $n$ on the $2D$ links touching that site (one forward
    and one backward link per direction).  A checkerboard sweep is used so that the $2D$ adjacent links
    of each same-color site are disjoint, enabling vectorized proposals and a single :meth:`~supervillain.lattice.Form.face_sum` call for $\Delta S$.

    .. seealso ::
       On a small 5×5 example this generator yields about three times as many updates per second than :class:`NeighborhoodUpdateSlow <supervillain.generator.reference_implementation.villain.NeighborhoodUpdateSlow>` on my machine.
       This ratio should *improve* for larger lattices because the change in action is computed directly and is of fixed cost, rather than scaling with the volume.
    '''

    def __init__(self, action, interval_phi=np.pi, interval_n=1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The Neighborhood Metropolis update requires the Villain action.')
        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_phi = interval_phi
        self.interval_n   = interval_n

        self.rng = np.random.default_rng()
        self.n_changes = np.arange(-interval_n, 1+interval_n)

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'NeighborhoodUpdate'

    def step(self, cfg):
        r'''
        Make a volume's worth of random single-site updates.

        Uses checkerboarding so that the 2D links adjacent to each same-color site are
        disjoint, allowing vectorized proposals and a single :func:`~supervillain.lattice.Form.face_sum`
        to aggregate $\Delta S$ per site.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n field variables.

        Returns
        -------
        dict
            Another configuration of fields.
        '''

        S = self.Action
        L = S.Lattice
        self.sweeps += 1
        total_accepted = 0
        total_acceptance = 0

        phi = cfg['phi'].copy()
        n   = cfg['n'].copy()

        metropolis = self.rng.uniform(0, 1, (L.N,) * L.D)

        # Precompute residuals r[mu, x] = d(phi)[mu,x] - 2π n[mu,x].
        # Updated incrementally after each accepted color sweep.
        r = d(phi) - 2 * np.pi * n

        for color in L.checkerboarding:
            n_sites = len(color[0])

            # --- Δφ proposal: non-zero only on this color ---
            change_phi = L.zeros(0)
            change_phi[0, *color] = self.rng.uniform(-self.interval_phi, +self.interval_phi, n_sites)

            # --- Δn proposals: one for each of the 2D links adjacent to each color-c site.
            # Forward link (mu, x) for x in color; backward link (mu, x-ê_mu) for x in color.
            # Same-color sites are ≥2 apart so their adjacent link sets are disjoint.
            change_n = L.zeros(1, dtype=int)
            for mu in range(L.D):
                change_n[mu, *color] = S.W * self.rng.choice(self.n_changes, n_sites)
                bwd = tuple(np.mod(color[i] - (1 if i == mu else 0), L.N) for i in range(L.D))
                change_n[mu, *bwd] = S.W * self.rng.choice(self.n_changes, n_sites)

            # --- ΔS: change in residual δr = d(Δφ) − 2π Δn on every link ---
            change_r = d(change_phi) - 2 * np.pi * change_n
            dS_link  = (S.kappa / 2) * change_r * (2 * r + change_r)
            dS       = dS_link.face_sum()  # 0-form: ΔS per site

            # --- Accept/reject on this color's sites only ---
            acceptance = np.clip(np.exp(-dS[0, *color]), a_min=0, a_max=1)
            accepted   = metropolis[color] < acceptance
            total_accepted     += accepted.sum()
            total_acceptance   += acceptance.sum()

            # --- Apply accepted updates: zero out rejected proposals ---
            change_phi[0, *color] *= accepted
            for mu in range(L.D):
                change_n[mu, *color] *= accepted
                bwd = tuple(np.mod(color[i] - (1 if i == mu else 0), L.N) for i in range(L.D))
                change_n[mu, *bwd] *= accepted

            phi = phi + change_phi
            n   = n   + change_n
            r   = r   + d(change_phi) - 2 * np.pi * change_n

        sites = L.cells_of_degree[0]
        self.proposed    += sites
        self.acceptance  += total_acceptance / sites
        self.accepted    += total_accepted
        logger.debug(f'Average proposal acceptance {total_acceptance/sites:.6f}; Actually {total_accepted}/{sites}')

        return cfg | {'phi': phi, 'n': n}

    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''
        return (
            f'There were {self.accepted} neighborhood proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate' 
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )

