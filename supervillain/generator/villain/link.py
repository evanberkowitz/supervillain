#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

class LinkUpdate(ReadWriteable, Generator):
    r'''
    This performs the same update to $n$ as :class:`NeighborhoodUpdate <supervillain.generator.villain.NeighborhoodUpdate>` but leaves $\phi$ untouched.

    Proposals are drawn according to

    .. math ::

        \begin{align}
        \Delta n_\ell   &\sim W \times [-\texttt{interval_n}, +\texttt{interval_n}] \setminus \{0\}
        \end{align}

    We pick :math:`\Delta n_\ell` to be a multiple of the constraint integer $W$ so that if the adjacent plaquettes satisfy the :ref:`winding constraint <winding constraint>` $dn \equiv 0 \text{ mod }W$
    before the update they satisfy it after as well.

    .. note ::
        You can run ``python supervillain/generator/villain/link.py`` to compare a pure $W=1$ :class:`~.NeighborhoodUpdate` ensemble against an ensemble which also does :class:`LinkUpdates <LinkUpdate>`.
        Note that adding the :class:`LinkUpdate` costs essentially 0 time because all the links are done in parallel and there are no python-level for loops.
    '''

    def __init__(self, action, interval_n=1):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The LinkUpdate requires the Villain action.')

        self.Action       = action
        self.Lattice      = action.Lattice
        self.kappa        = action.kappa

        self.interval_n   = interval_n

        self.rng = np.random.default_rng()
        self.n_changes = tuple(n for n in range(-interval_n, 0)) + tuple(n for n in range(1, interval_n+1))

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def __str__(self):
        return 'LinkUpdate'

    def step(self, cfg):
        r'''
        Make volume's worth of random single-link updates.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n field variables.

        Returns
        -------
        dict
            Another configuration of fields.
        '''

        self.sweeps += 1
        total_acceptance = 0
        accepted = 0

        phi = cfg['phi'].copy()
        n   = cfg['n'].copy()
        one_form = n.shape

        # The n variables are all independent, in the sense that the action S doesn't couple them directly.
        # We can therefore offer updates independently, holding dphi a fixed background.
        dphi = self.Lattice.d(0, phi)
        # That lets us do a whole 1-form worth of updates simultaneously.
        change_n = self.Action.W * self.rng.choice(self.n_changes, one_form)

        # Use numpy's broadcasting to evaluate the change in S independently for each link.
        # What lets us do this so simply is that this generator does not update phi.
        # So the change in action from changing n just depends on a fixed background dphi,
        # and on n itself---no n from any other link is involved.
        #dS = 0.5 * self.kappa * (-2*np.pi*change_n) * (2*(dphi - 2*np.pi*n) - 2*np.pi*change_n)
        dS = -2*np.pi * self.kappa * change_n * (dphi - 2*np.pi*n - np.pi*change_n)
        # The point is, dS can really be evaluated link-by-link if we freeze phi;
        # we're not missing any pieces that come from changing n on two nearby links at once.

        # Because the links don't talk to one another, can accept or reject them simultaneously.
        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = self.rng.uniform(0, 1, one_form)
        accepted = metropolis < acceptance
        n += np.where(accepted, change_n, 0)

        self.acceptance += acceptance.mean()
        self.accepted += accepted.sum()
        self.proposed += n.size

        logger.debug(f'Average proposal acceptance {acceptance.mean():.6f}; Actually accepted {accepted.sum()} / {self.Action.Lattice.sites} = {accepted.sum() / self.Action.Lattice.sites / 2}')

        return {'phi': phi, 'n': n}

    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''
        return (
            f'There were {self.accepted} single-link proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )

if __name__ == '__main__':
    # Here we just make a simple comparison between Neighborhood-only and Neighborhood + Link updates.
    # Neighborhood alone is ergodic for W=1, all that adding the Link update can do is help decorrelate.

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import supervillain
    import supervillain.analysis.comparison_plot as comparison_plot
    supervillain.observable.progress=tqdm

    parser = supervillain.cli.ArgumentParser()
    parser.add_argument('--N', type=int, default=5, help='Sites on a side.  Defaults to 5.')
    parser.add_argument('--kappa', type=float, default=0.25, help='κ.  Defaults to 0.25.')
    parser.add_argument('--configurations', type=int, default=100000, help='Defaults to 100000.  You need a good deal of configurations with κ=0.5 because of autocorrelations in the Villain sampling.')

    args = parser.parse_args()

    L = supervillain.lattice.Lattice2D(args.N)

    # We compare pure Neighborhood updates
    S = supervillain.action.Villain(L, args.kappa, W=1)
    G = supervillain.generator.combining.KeepEvery(1,
        supervillain.generator.villain.NeighborhoodUpdate(S)
        )
    O = supervillain.Ensemble(S).generate(args.configurations, G, progress=tqdm)
    print('Neighborhood Only:')
    print(G.report())

    # to Neighborhood + Link.  Since the link updates should be accepted more frequently than the neighborhood updates
    # we expect to see the same expected values of observables (given the same action) but shorter autocorrelation times.
    G = supervillain.generator.combining.Sequentially((
        supervillain.generator.villain.NeighborhoodUpdate(S),
        supervillain.generator.villain.LinkUpdate(S),
    ))
    N = supervillain.Ensemble(S).generate(args.configurations, G, progress=tqdm)
    print('Neighborhood and Link Updates:')
    print(G.report())

    ensembles = (
        O,
        N,
    )
    labels = (
        'Neighborhood',
        '+ LinkUpdate',
    )

    taus = tuple(e.autocorrelation_time() for e in ensembles)

    thermalized = tuple(e.cut(10*tau) for e, tau in zip(ensembles, taus))
    taus = tuple(e.autocorrelation_time() for e in thermalized)

    for label, tau in zip(labels, taus):
        print(f'{label:12s}: τ={tau}')

    decorrelated = tuple(e.every(tau) for e, tau in zip(thermalized, taus))
    bootstraps   = tuple(supervillain.analysis.Bootstrap(e) for e in decorrelated)


    fig, ax = comparison_plot.setup()
    comparison_plot.bootstraps(ax,
            bootstraps,
            labels,
            )
    comparison_plot.histories(ax,
            ensembles,
            labels,
            )

    fig.suptitle(f'Villain N={args.N} κ={args.kappa} W=1')
    fig.tight_layout()

    plt.show()
