#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

class NeighborhoodUpdateSlow(ReadWriteable, Generator):
    r'''
    A neighborhood update changes only fields in some small area of the lattice.

    In particular, this updating scheme changes the $\phi$ and $n$ fields in the :class:`~.Villain` formulation.

    It works by picking a site $x$ at random, proposing a change 

    .. math ::
        
        \begin{align}
        \Delta\phi_x    &\sim \text{uniform}(-\texttt{interval_phi}, +\texttt{interval_phi})
        \\
        \Delta n_\ell   &\sim [-\texttt{interval_n}, +\texttt{interval_n}]
        \end{align}

    for the $\phi$ on $x$ and $n$ on links $\ell$ which touch $x$.

    .. warning ::
        Because we currently restrict to $W=1$ for the Villain formulation we do not update $v$.

    Parameters
    ----------
    action: Villain
        The action from which we sample.
    interval_phi: float
        A single float used to construct the uniform distribution for $\phi$.
    interval_n: int
        A single integer that gives the biggest allowed changes to $n$.
    '''

    def __init__(self, action, interval_phi=np.pi, interval_n=1):

        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The Neighborhood Metropolis update requires the Villain action.')
        self.Action       = action
        self.interval_phi = interval_phi
        self.interval_n   = interval_n

        self.rng = np.random.default_rng()
        self.n_changes = np.arange(-interval_n, 1+interval_n)

        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        self.sweeps = 0

    def proposal(self, cfg, dx):
        r'''
        Parameters
        ----------
        cfg: dict
            A dictionary with $\phi$ and $n$ to update.
        dx: Lattice coordinates
            Which site to move to the origin and update.

        Returns
        -------
        dict:
            A new configuration with updated $\phi$ and $n$.
        '''
        L = self.Action.Lattice

        # We move the lattice around (which is fine by translational symmetry)
        # so that we update a different site with each proposal.
        # The advantage of thinking this way is that we only have to reckon from the origin.
        phi = L.roll(cfg['phi'].copy(), dx)
        n   = L.roll(cfg['n'].copy(),   dx)

        phi[0,0] += self.rng.uniform(-self.interval_phi,+self.interval_phi,None)

        n_shift = self.rng.choice(self.n_changes,4)
        n[0][+0,+0] += n_shift[0]
        n[0][-1,+0] += n_shift[1]
        n[1][+0,+0] += n_shift[2]
        n[1][+0,-1] += n_shift[3]

        return {'phi': phi, 'n': n}

    def site(self, cfg, dx):
        r'''
        Rather than accepting every :func:`~proposal` we perform *importance sampling* by doing a Metropolis accept/reject step :cite:`Metropolis` on every single-site proposal.

        Parameters
        ----------
        cfg: dict
            A dictionary with $\phi$ and $n$ to update.
        dx: Lattice coordinates
            Which site to move to the origin and update.

        Returns
        -------
        dict:
            A configuration; either the provided one a new one changed by a proposal.
        float:
            The Metropolis-Hastings acceptance probability.
        int:
            1 if the proposal was accepted, 0 otherwise.
        '''

        S_start    = self.Action(cfg['phi'], cfg['n'])
        proposal   = self.proposal(cfg, dx)
        S_proposal = self.Action(proposal['phi'], proposal['n'])

        dS = S_proposal - S_start

        acceptance = np.clip( np.exp(-dS), a_min=0, a_max=1)
        metropolis = np.random.default_rng().uniform(0,1,None)
        if metropolis < acceptance:
            logger.debug(f'Proposal accepted; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
            return proposal, acceptance, 1
        else:
            logger.debug(f'Proposal rejected; ∆S = {dS:f}; acceptance probability = {acceptance:f}')
            return cfg, acceptance, 0

    def step(self, cfg):
        r'''
        Make volume's worth of random single-site updates.

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
        current = cfg
        acceptance = 0
        accepted = 0

        L = self.Action.Lattice

        shifts = np.stack((
            np.random.randint(L.dims[0], size=L.sites),
            np.random.randint(L.dims[1], size=L.sites)
        )).transpose()

        for dx in shifts:
            subsequent, probability, acc = self.site(current, dx)
            current = subsequent
            acceptance += probability
            accepted   += acc

        self.accepted += accepted
        self.proposed += len(shifts)

        acceptance /= len(shifts)
        self.acceptance += acceptance
        logger.info(f'Average proposal {acceptance=:.6f}; Actually {accepted = } / {self.Action.Lattice.sites} = {accepted / self.Action.Lattice.sites}')

        return current

    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''
        return (
            f'There were {self.accepted} single-site proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.proposed:.6f} acceptance rate' 
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )


class ClassicWorm(ReadWriteable, Generator):
    r'''
    This implements the classic worm of Prokof'ev and Svistunov :cite:`PhysRevLett.87.160601` for the Villain links $n\in\mathbb{Z}$ which satisfy $dn \equiv 0 $ (mod W) on every plaquette.

    On top of a constraint-satisfying configuration we put down a worm and let the head move, changing the crossed links.
    We uniformly propose a move in all 4 directions and Metropolize the change.

    Additionally, when the head and tail coincide, we allow a fifth possible move, where we remove the worm and emit the updated $z$ configuration into the Markov chain.
    
    As we evolve the worm we tally the histogram that yields the :class:`~.Vortex_Vortex` correlation function.

    .. warning ::
        
        This update algorithm is not ergodic on its own.  It doesn't change $\phi$ at all and even leaves $dn$ alone (while changing $n$ itself).
        It can be used, for example, :class:`~.Sequentially` with the :class:`~.SiteUpdate` and :class:`~.LinkUpdate` for an ergodic method.

    .. warning ::

        Because the algorithm is about moving a single defect around the lattice, when implemented in pure python
        the python-level loop can severely impact performance.  While this reference implementation was done in
        pure python, :class:`the production-ready generator <supervillain.generator.villain.worm.ClassicWorm>` uses numba for acceleration.

    .. note ::
        
        **Because** it doesn't change $dn$ at all, this algorithm can play an important role in sampling the $W=\infty$ sector, where all vortices are completely killed, though updates to $\phi$ would still be needed.
    '''

    def __init__(self, S):
        self.Action = S
        self.rng = np.random.default_rng()

        self.worm_lengths = deque()
        # The contributions to the plaquette tell you how an n contributes to dn.
        # Opposite directions contribute oppositely, which is exactly what you want.
        # That way, if the worm moves north, you increase n by 1, but if the worm then
        # immediately moves south it would cross the same link but decrease n by 1,
        # so that the constraint on this cul-de-sac would be restored.
        self.plaquette = np.array([+1, +1, -1, -1]) # east, north, west, south

    def __str__(self):
        return 'ClassicWorm'

    def _neighboring_plaquettes(self, here):
        # east, north, west, south
        return self.Action.Lattice.mod(here + np.array([[0,-1], [+1,0], [0,+1], [-1,0]]))

    def _surrounding_links(self, here):
        # These are the directions we'd like to move the head of the defect.
        east, north, west, south = self._neighboring_plaquettes(here)

        return ((0, here [0], here [1]), # t link to the east
                (1, north[0], north[1]), # x link to the north
                (0, west [0], west [1]), # t link to the west
                (1, here [0], here [1])) # x link to the south


    def inline_observables(self, steps):
        r'''
        The worm algorithm can measure the ``Vortex_Vortex`` correlator.
        We also store the ``Worm_Length`` for each step.
        '''

        return {
            'Vortex_Vortex': extendable.array(self.Action.Lattice.form(0, steps)),
            'Worm_Length':   extendable.array(np.zeros(steps)),
        }

    def step(self, configuration):
        r'''
        Given a constraint-satisfying configuration, returns another constraint-satisfying configuration udpated via worm as described above.
        '''

        S = self.Action
        L = S.Lattice

        displacements = L.form(0)

        # This algorithm will not update phi; but it is useful to precompute dphi
        # which is used in the evaluation of the changes in action.
        phi = configuration['phi'].copy()
        dphi = L.d(0, phi)

        # The documentation gives a definitive statement about moving the head only.
        # But we could equally well move the tail, making the opposite moves in the opposite worm evolution.
        # This can be accomplished simply by multiplying the offered changes to the links by -1.
        # We can randomly decide this orientation of the worm
        orientation = self.rng.choice([-1, +1])
        # and then simply multiply it into the constraint-restoring proposals.
        change_n = orientation * self.plaquette

        # We start with a constraint-satisfying configuration of n that is in the z sector.
        n = configuration['n'].copy()
        # and insert both the head and tail onto any random plaquette---because the head and the tail are
        # coincident, they don't change the action and so any choice should be equally weighted.
        tail = self.rng.choice(L.coordinates)
        #   The only exception is that when W=1 the unmodified Z constraint is (mod 1) which is always satisfied, even with an open worm.
        #   Therefore, we can put down an open worm from the start.  When W>1 the head and the tail have to be coincident
        #   in order to satisfy the unmodified constraint.
        head = (tail.copy() if S.W != 1 else self.rng.choice(L.coordinates))
        # by placing the head and tail down we have moved to the g sector!
        # Now we are ready to start evolving in z union g.

        while True:

            # In the general case we will uniformly choose between 4 moves,
            # but if the head and tail are together, we add the g--> z transition.
            # This has likelihood of 20%, conditioned on the worm being closed.
            # If it is proposed, however, the change in action is 0 and it is automatically accepted as a z configuration.
            if ((head == tail).all() or (S.W==1)) and (self.rng.uniform(0, 1) >= 0.8):
                # Just as we could put down an open worm when W=1, we can also remove an open worm in that case.
                # In other words, if W=1 we always have the possibility of return a Z configuration from the existing G configuration.
                wl = displacements.sum()
                self.worm_lengths.append(wl)
                return {'n': n, 'phi': phi, 'Vortex_Vortex': displacements, 'Worm_Length': wl}
            
            # Conditioned on not transitioning to z, we make a uniform choice of the 4 possible directions.
            choice = self.rng.choice([0,1,2,3])

            # We may move the head to 1 of 4 neighboring plaquettes
            the_next = self._neighboring_plaquettes(head)[choice]
            # in which case we will cross the corresponding links.
            link = self._surrounding_links(head)[choice]

            # Crossing the link changes n and therefore the action.
            change_link = dphi[link] - 2*np.pi*n[link]
            delta_n = change_n[choice]
            change_S = (
                (S.kappa / 2) *
                (-2*np.pi*delta_n) *
                (2*change_link - 2*np.pi*delta_n)
            )

            # Now we must compute the Metropolis amplitude
            #
            #   A = min(1, exp(-ΔS) )
            #
            A = np.clip(np.exp(-change_S), a_min=0, a_max=1)

            # and Metropolis-test the update.
            if self.rng.uniform(0, 1) < A:
                # If it accepted we move the head
                head = the_next
                # and cross the link.
                n[link] += delta_n

            # Finally, we tally the worm,
            x, y = L.mod(head-tail)
            displacements[x, y] += 1
            # and consider our next move.

    def report(self):
        l = np.array(self.worm_lengths)
        return f'There were {len(l)} worms.\nWorms lengths:\n    mean {l.mean()}\n    std  {l.std()}\n    max  {max(l)}'


