#!/usr/bin/env python

r"""The **wrapping-sheet move**: a non-diffusive proposal that changes the $H^2$ class
$[F]$ of the :class:`~.gas.SurfaceWormGas`'s state by $\pm1$ in one step.

Why a separate move at all
---------------------------

The SWG's own moves --- the plaquette toggle and the coboundary heatbath --- are
**local**.  Reaching a state with $[F] \neq 0$ through them means growing an open surface
whose boundary loop happens to wrap a non-contractible 1-cycle and then transporting that
loop all the way around a second direction before it recloses: an $\mathcal{O}(N^2)$-step
coherent accident in a random walk.  That is the kinetic form of the wall, and it is why
no chain on record has ever been observed at $[F] \neq 0$.

This module supplies the missing proposal.  It does **not** change the sampler's
stationary distribution: :class:`WrappingSheetGas` adds a move to
:class:`~.gas.SurfaceWormGas` and nothing else, so every tuned table, every emission
guarantee, and every measured observable carries over untouched.

The sheet
---------

:func:`wrapping_sheet` builds $W$: unit flux in one plaquette component $c = (ab)$,
localized at $(i, j)$ in $c$'s own two directions and **constant along the other two**.
Four properties, each load-bearing:

* $dW = 0$ **identically**.  Constancy along the transverse directions $\rho,\sigma$ kills
  $(dW)_{\mu ab}$ for $\mu \in \{\rho,\sigma\}$, and a repeated index kills the rest.  So
  the move changes $D$ by **exactly zero** --- it does not climb the open-surface wall, it
  goes around it.
* $W \wedge W = 0$ **identically**.  A single-component 2-form has no second pair of
  distinct indices to wedge with.  So $\Delta q$ is purely the cross term
  $F\wedge W + W\wedge F$: the move creates intersections only where it crosses the
  *existing* network.
* $\sum_x W_c = N^2$, so :attr:`~.state.FState.periods` moves by exactly $\pm N^2$ in one
  component and the class $[F]$ by $\pm1$.  The move **is** the wrapping; it is not a
  search for one.
* Its cost is a $2$-torus's self-energy plus a screened cross term, both already measured
  --- on thermal $N = 6$, $\kappa = 0.03$ backgrounds the best placement costs
  $\Delta S = -0.81 \pm 0.89$, i.e. **negative**.  What it costs instead is charge:
  $Q_{\min} \approx 16$, and no charge-free placement exists (the exact rational kernel of
  the charge response is $\{0\}$ in all six components).

Detailed balance
----------------

The proposal draws the component $c$ uniformly from 6, the placement $(i,j)$ uniformly
from $N^2$, and the sign $s$ uniformly from $\pm1$.  The reverse of "add $sW$ at $(c,i,j)$"
is "add $-sW$ at $(c,i,j)$", which is drawn with the *same* probability, so the proposal is
symmetric and plain Metropolis on $\Delta\log\pi_\text{ext}$ is exactly balanced --- there
is no Hastings ratio to get wrong.

.. note ::
    The acceptance is computed by differencing
    :meth:`~.gas.SurfaceWormGas._log_extended_weight`, the sampler's own from-scratch
    $\log\pi_\text{ext}$, rather than by an incremental formula.  That is
    $\mathcal{O}(V\log V)$ per proposal and deliberately so: the move is rare (once per
    ``sheetEvery`` local moves), and differencing the authoritative weight means the sheet
    move cannot drift away from what the rest of the sampler believes.
"""

import numpy as np

from .gas import SurfaceWormGas
from .kernel import pot

#: Plaquette components in the library's canonical order, $(01,02,03,12,13,23)$.
COMPONENTS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

#: For each component, the one whose two directions are the *complementary* pair --- the
#: only component whose sheets a given component's sheets actually intersect, since
#: $W\wedge W'$ needs all four indices distinct.  In the canonical order this is simply
#: index reversal: $(01)\leftrightarrow(23)$, $(02)\leftrightarrow(13)$,
#: $(03)\leftrightarrow(12)$.
COMPLEMENT = (5, 4, 3, 2, 1, 0)


def transverse_axes(c):
    r"""The two directions component ``c``'s wrapping sheet extends along.

    Parameters
    ----------
    c: int
        A plaquette component, indexing :data:`COMPONENTS`.

    Returns
    -------
    tuple of int
        The two axes *not* in ``COMPONENTS[c]``.
    """
    return tuple(mu for mu in range(4) if mu not in COMPONENTS[c])


def wrapping_sheet(N, c, i, j, sign=1):
    r"""The minimal $[F]$-changing closed 2-form $W$ --- a $2$-torus of unit flux.

    Unit flux in component $c = (ab)$, localized at ``i`` along $a$ and ``j`` along $b$,
    and constant along the remaining two directions.  Closed ($dW = 0$) and non-exact
    ($\sum_x W_c = N^2$), so adding it moves the class $[F]$ by ``sign`` in component
    ``c`` while leaving $D$ untouched.

    Parameters
    ----------
    N: int
        Lattice extent.
    c: int
        Plaquette component, indexing :data:`COMPONENTS`.
    i, j: int
        Position along ``COMPONENTS[c][0]`` and ``COMPONENTS[c][1]``.
    sign: int
        $+1$ or $-1$; which way the class moves.

    Returns
    -------
    numpy.ndarray
        ``int64`` of shape ``(6,) + (N,)*4``.
    """
    a, b = COMPONENTS[c]
    W = np.zeros((6,) + (N,) * 4, dtype=np.int64)
    index = [slice(None)] * 4
    index[a] = i
    index[b] = j
    W[c][tuple(index)] = sign
    return W


def sheet_dipole(N, c, i1, j1, i2, j2, sign=1):
    r"""A **transported** sheet: $W_c(i_1,j_1) - W_c(i_2,j_2)$.

    Closed (both terms are) and with **zero periods** (they cancel), hence *exact* --- so
    adding it leaves the $H^2$ class untouched.  A dipole alone has
    $\Delta\wedge\Delta = 0$ and deposits no charge and no $J$; it is only when two
    dipoles in **complementary** components are added together that the cross term
    $A\wedge B$ survives.  See :meth:`WrappingSheetGas.commutator_move`.

    Parameters
    ----------
    N: int
        Lattice extent.
    c: int
        Plaquette component, indexing :data:`COMPONENTS`.
    i1, j1, i2, j2: int
        The two placements, in ``COMPONENTS[c]``'s own two directions.
    sign: int
        $+1$ or $-1$.

    Returns
    -------
    numpy.ndarray
        ``int64`` of shape ``(6,) + (N,)*4``.
    """
    return (wrapping_sheet(N, c, i1, j1, sign)
            + wrapping_sheet(N, c, i2, j2, -sign))


def sheet_self_energy(N):
    r"""$C_W = \sum_x W\,\Delta^{-1}W$ for a unit wrapping sheet, $O(V\log V)$.

    Independent of component and placement by the lattice's translation and rotation
    symmetry, so one evaluation serves every proposal.

    Parameters
    ----------
    N: int
        Lattice extent.

    Returns
    -------
    float
        The sheet's bare coexact self-energy, in units where
        $\Delta S = 2\pi^2\kappa\,C_W$.
    """
    W = wrapping_sheet(N, 0, 0, 0)
    return float((W[0] * pot(W[0], N)).sum())


class WrappingSheetGas(SurfaceWormGas):
    r"""A :class:`~.gas.SurfaceWormGas` that also proposes whole wrapping sheets.

    .. note ::
        The stationary distribution is **identical** to the parent's.  This class adds a
        move; it changes no weight.  Consequently every tuned $w(D)$, $w(Q)$ or
        $w(D,Q)$ table, and every emission guarantee, transfers without re-tuning --- and
        an instance with ``sheetEvery = 0`` is the parent sampler exactly.

    .. warning ::
        The move only pays if the charge price lets it.  A wrapping sheet on a thermal
        background makes $Q_{\min}\approx16$ intersections at $N = 6$, so under the
        production fugacity $\eta_q = 0.012$ its acceptance is $e^{-70}$ and it will never
        fire.  It is meant to be run against a **capped** charge table (one whose cost
        saturates below the sheet's $Q$), where the same move costs almost nothing.
        :attr:`sheetAccepted` staying at 0 is the symptom of pairing it with the wrong
        table, not of a broken move.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action, as for the parent.
    sheetEvery: int
        Attempt one sheet move per this many local moves.  ``0`` disables the move
        entirely, recovering the parent sampler.
    sheetAttempts: int
        How many sheet proposals to make at each attempt point.
    kwargs:
        Everything else is passed to :class:`~.gas.SurfaceWormGas`.

    Attributes
    ----------
    sheetProposed, sheetAccepted: int
        Proposal and acceptance counts for the sheet move.
    sheetClassVisits: int
        How many sheet moves landed the chain at a nonzero class (as opposed to
        returning it to $[F] = 0$) --- the quantity the move exists to make nonzero.
    """

    def __init__(self, S, sheetEvery=10000, sheetAttempts=1, commutatorEvery=0, **kwargs):
        super().__init__(S, **kwargs)
        self.sheetEvery = int(sheetEvery)
        self.sheetAttempts = int(sheetAttempts)
        self.commutatorEvery = int(commutatorEvery)
        self.sheetProposed = 0
        self.sheetAccepted = 0
        self.sheetClassVisits = 0
        self.commutatorProposed = 0
        self.commutatorAccepted = 0

    def __str__(self):
        return f'WrappingSheetGas(sheetEvery={self.sheetEvery})'

    def sheet_log_acceptance(self, state, c, i, j, s):
        r"""The exact Metropolis log-acceptance for adding ``s`` times the sheet
        ``(c, i, j)``, by differencing the sampler's own from-scratch
        $\log\pi_\text{ext}$.

        Parameters
        ----------
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            Not mutated.
        c, i, j, s: int
            The proposal, as for :func:`wrapping_sheet`.

        Returns
        -------
        (float, numpy.ndarray)
            The log-acceptance and the proposed ``F``.
        """
        F = state.F + wrapping_sheet(self.N, c, i, j, s)
        return self._log_extended_weight(F) - self._log_extended_weight(state.F), F

    def sheet_move(self, state):
        r"""Propose one wrapping sheet and accept or reject it.

        Parameters
        ----------
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            Mutated in place on acceptance.

        Returns
        -------
        bool
            Whether the move was accepted.
        """
        N = self.N
        c = int(self.rng.integers(6))
        i, j = (int(v) for v in self.rng.integers(N, size=2))
        s = 1 if self.rng.random() < 0.5 else -1
        self.sheetProposed += 1
        lnA, F = self.sheet_log_acceptance(state, c, i, j, s)
        if lnA < 0 and self.rng.random() >= np.exp(lnA):
            return False
        state.F = F
        state.resync()
        self.sheetAccepted += 1
        if state.periods.any():
            self.sheetClassVisits += 1
        return True

    def commutator_log_acceptance(self, state, spec):
        r"""The exact Metropolis log-acceptance for the transport commutator ``spec``.

        Parameters
        ----------
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            Not mutated.
        spec: tuple
            ``(c, i1, j1, i2, j2, k1, l1, k2, l2, s)`` --- the component, its dipole's two
            placements, the complementary component's two placements, and the sign.

        Returns
        -------
        (float, numpy.ndarray)
            The log-acceptance and the proposed ``F``.
        """
        c, i1, j1, i2, j2, k1, l1, k2, l2, s = spec
        N = self.N
        F = (state.F + sheet_dipole(N, c, i1, j1, i2, j2, s)
                     + sheet_dipole(N, COMPLEMENT[c], k1, l1, k2, l2, s))
        return self._log_extended_weight(F) - self._log_extended_weight(state.F), F

    def commutator_move(self, state):
        r"""Propose the commutator of two sheet **transports** in complementary planes.

        .. note ::
            This is the move that keeps the class trivial.  Each dipole has zero periods,
            so $[F]$ is untouched and the chain stays on the physical shell --- it cannot
            be absorbed into the unpriced class direction the way a bare
            :meth:`sheet_move` chain is.  What survives is the cross term $A\wedge B$
            between complementary components, which deposits charge (exactly 4 quanta on
            an empty background, the minimum) where neither dipole alone deposits any.

        .. warning ::
            $J$ cannot be read at the proposed state itself: $dj = q$, so $J$ is
            slice-dependent wherever $Q \neq 0$, and this move always leaves $Q \neq 0$ on
            an empty background.  The sampler has to heal the charge first, which is the
            whole reason this is a *move* rather than a construction.

        Parameters
        ----------
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            Mutated in place on acceptance.

        Returns
        -------
        bool
            Whether the move was accepted.
        """
        N = self.N
        c = int(self.rng.integers(6))
        placements = [int(v) for v in self.rng.integers(N, size=8)]
        s = 1 if self.rng.random() < 0.5 else -1
        spec = (c, *placements, s)
        self.commutatorProposed += 1
        lnA, F = self.commutator_log_acceptance(state, spec)
        if lnA < 0 and self.rng.random() >= np.exp(lnA):
            return False
        state.F = F
        state.resync()
        self.commutatorAccepted += 1
        return True

    def sweep(self, state, nmoves, pCob=None):
        r"""The parent's sweep, with a sheet move attempted every
        :attr:`sheetEvery` local moves.

        Parameters
        ----------
        state: supervillain.generator.no_intersection.surface_worm.state.FState
            Mutated in place.
        nmoves: int
            Local moves to make; sheet attempts are *extra* and are not counted here.
        pCob: float, optional
            Coboundary-move probability, as for the parent.

        Returns
        -------
        FState
            ``state``, for chaining.
        """
        every = [e for e in (self.sheetEvery, self.commutatorEvery) if e > 0]
        if not every:
            return super().sweep(state, nmoves, pCob)
        step = min(every)
        done = 0
        while done < nmoves:
            chunk = min(step, nmoves - done)
            super().sweep(state, chunk, pCob)
            done += chunk
            for _ in range(self.sheetAttempts):
                if self.sheetEvery > 0 and done % self.sheetEvery == 0:
                    self.sheet_move(state)
                if self.commutatorEvery > 0 and done % self.commutatorEvery == 0:
                    self.commutator_move(state)
        return state

    def report(self):
        r"""The parent's acceptance summary, plus the sheet move's.

        Returns
        -------
        str
            Multi-line summary.
        """
        p = max(self.sheetProposed, 1)
        cp = max(self.commutatorProposed, 1)
        return (super().report()
                + f'\nwrapping sheets: {self.sheetAccepted}/{self.sheetProposed} '
                  f'({self.sheetAccepted / p:.4f}), '
                  f'{self.sheetClassVisits} landings at nonzero class'
                + f'\ntransport commutators: {self.commutatorAccepted}/'
                  f'{self.commutatorProposed} ({self.commutatorAccepted / cp:.4f})')
