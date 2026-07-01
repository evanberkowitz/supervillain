#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import delta, delta_sparse, coface_sum_at, Form

import logging
logger = logging.getLogger(__name__)

class CoexactUpdate(ReadWriteable, Generator):
    r'''
    One way to guarantee that $\delta m = 0$ is to change $m$ by $\delta t$ where $t$ is an integer-valued two-form.

    Proposals are drawn according to

    .. math ::

        \begin{aligned}
            t_p   &\sim [-\texttt{interval\_t}, +\texttt{interval\_t}] \setminus \{0\}
            &
            \Delta m_\ell &= (\delta t)_\ell
        \end{aligned}

    .. warning ::
        This algorithm is not ergodic on its own.  It does not change $v$ (see the :class:`~.worldline.VortexUpdate`)
        nor can it produce all coclosed changes---only coexact changes.
        For a coïnexact coclosed update consider the :class:`~.worldline.WrappingUpdate` or the :class:`worm <supervillain.generator.worldline.worm.Classic>`.
    '''

    def __init__(self, action, interval_t = 1):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('Need a Worldline action')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_t = interval_t
        self.ts = tuple(t for t in range(-interval_t, 0)) + tuple(t for t in range(1, interval_t+1))

        self.rng = np.random.default_rng()

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'SiteUpdate'

    def step(self, cfg):
        r'''
        Make a volume's worth of locally-exact updates to m.

        Each proposal changes m by $\delta t$ with $t$ supported on a single
        2-form component and checkerboard color, so both the proposal's $\delta t$
        and the accepted patch are evaluated with :func:`~.delta_sparse` instead
        of a full $\delta$.  $\delta v$ is already constant over the sweep (only m
        changes), and m is integer, so this is bit-identical to
        :meth:`step_reference` for every $W$ (checked by ``test/test_coexact_sparse.py``).

        Parameters
        ----------
        cfg: dict
            A dictionary with m and v field variables.

        Returns
        -------
        dict
            Another configuration of fields.
        '''

        self.sweeps += 1
        total_acceptance = 0
        total_accepted = 0

        v = cfg['v']
        delta_v_by_W = np.asarray(delta(v)) / self.Action._W   # frozen over the sweep

        m = cfg['m'].copy()
        m_raw = np.asarray(m)                                  # patched in place below

        L = self.Lattice
        n_comps = len(L.components[2])

        # One independent Metropolis draw per (2-form component, site).
        metropolis = self.rng.uniform(0, 1, (n_comps,) + L.dims)

        # The change in m is a coexact form δt (t a 2-form) so that δm stays 0 (δ²t = 0).  Two plaquettes
        # of the same component at same-color sites never share boundary links, so each (component, color)
        # is proposed and accepted independently; we process components sequentially to avoid conflicts
        # between different components at the same site.  Both δt's — the proposal and the accepted patch —
        # act on a single-component, single-color t, so delta_sparse touches only the affected links.
        for color in L.checkerboarding:
            for comp_idx in range(n_comps):

                vals = self.rng.choice(self.ts, len(color[0]))

                # change_m = δt, computed sparsely from the one nonzero component+color.
                cm = delta_sparse(L, 2, comp_idx, color, vals)
                dS_link = Form(
                    (0.5 / self.Action.kappa) * cm * (2 * (m_raw - delta_v_by_W) + cm),
                    degree=1, lattice=L,
                )

                # ΔS at this plaquette is the sum over its boundary links.  We only need it on the
                # current (component, color), so coface_sum_at gathers just those links instead of a
                # full coface_sum over the lattice.
                dS = coface_sum_at(dS_link, comp_idx, color)

                acceptance = np.clip(np.exp(-dS), a_min=0, a_max=1)
                accepted = (metropolis[comp_idx][color] < acceptance)

                total_accepted += accepted.sum()
                total_acceptance += acceptance.sum()

                # Apply: m += δ(t restricted to accepted plaquettes), patched in place.
                delta_sparse(L, 2, comp_idx, color, vals * accepted, out=m_raw)

        self.proposed += L.cells_of_degree[2]
        self.acceptance += total_acceptance / L.cells_of_degree[2]
        self.accepted += total_accepted

        logger.debug(f'Average proposal acceptance {total_acceptance / L.cells_of_degree[2]:.6f}; Actually accepted {total_accepted} / {L.cells_of_degree[2]} = {total_accepted / L.cells_of_degree[2]}')

        return cfg | {'m': m}

    def step_reference(self, cfg):
        r'''
        Reference (dense) implementation of :meth:`step`.

        Forms each proposal's $\delta t$ densely and re-derives $\delta t$ a
        second time for the accepted m update.  Kept as the simple,
        obviously-correct oracle that :meth:`step` is checked against
        (bit-identical for every $W$); not used in production.
        '''

        self.sweeps += 1
        total_acceptance = 0
        total_accepted = 0

        v = cfg['v']
        delta_v_by_W = delta(v) / self.Action._W

        m = cfg['m'].copy()

        L = self.Lattice
        n_comps = len(L.components[2])

        metropolis = self.rng.uniform(0, 1, (n_comps,) + L.dims)

        for color in L.checkerboarding:
            for comp_idx in range(n_comps):

                t = L.form(2, dtype=int)
                t[comp_idx][color] = self.rng.choice(self.ts, len(color[0]))

                change_m = delta(t)
                m_raw  = np.asarray(m)
                dvw    = np.asarray(delta_v_by_W)
                cm     = np.asarray(change_m)
                dS_link = Form(
                    (0.5 / self.Action.kappa) * cm * (2 * (m_raw - dvw) + cm),
                    degree=1, lattice=L,
                )

                dS = dS_link.coface_sum()

                acceptance = np.clip(np.exp(-np.asarray(dS[comp_idx])[color]), a_min=0, a_max=1)
                accepted = (metropolis[comp_idx][color] < acceptance)

                total_accepted += accepted.sum()
                total_acceptance += acceptance.sum()

                t[comp_idx][color] *= accepted
                m = m + delta(t)

        self.proposed += L.cells_of_degree[2]
        self.acceptance += total_acceptance / L.cells_of_degree[2]
        self.accepted += total_accepted

        return cfg | {'m': m}



    def report(self):
        return (
            f'There were {self.accepted} coexact proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )
