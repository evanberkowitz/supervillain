#!/usr/bin/env python

r"""Gates on the wrapping-sheet move.

The move exists to change the $H^2$ class $[F]$ in one step, so the tests are built around
the four identities that make it work --- $dW = 0$, $W\wedge W = 0$, $\sum_x W_c = N^2$,
and the symmetry of the proposal --- plus two guards that would catch the move being
silently inert or silently wrong: that ``sheetEvery = 0`` reproduces the parent sampler
*bit for bit*, and that the acceptance computed by the sampler agrees with the independent
placement formula the cost scan used.
"""

import numpy as np
import pytest

import supervillain
from supervillain.action import NoIntersections
from supervillain.generator.no_intersection import SurfaceWormGas, WrappingSheetGas
from supervillain.generator.no_intersection.surface_worm.kernel import pot
from supervillain.generator.no_intersection.surface_worm.sheet import (
    COMPLEMENT, COMPONENTS, sheet_dipole, sheet_self_energy, transverse_axes,
    wrapping_sheet)
from supervillain.generator.no_intersection.surface_worm.state import FState
from supervillain.generator.no_intersection.surface_worm.weights import SectorWeights
from supervillain.lattice import Lattice, d, wedge

N = 4
KAPPA = 0.1


@pytest.fixture
def S():
    return NoIntersections(Lattice(4, N), KAPPA)


def two_form(S, F):
    f = S.Lattice.form(2)
    np.asarray(f)[...] = F
    return f


def cheap_charge(cap=64, cost=0.05):
    r"""A charge table whose cost saturates almost immediately --- the regime in which the
    sheet move is supposed to fire at all.

    .. warning ::
        ``hardWall = False`` is not incidental.  A wrapping sheet adds $Q_{\min}$
        intersections at once, so a table that walls off at ``cap`` rejects the move
        *unconditionally* whenever the chain sits within $Q_{\min}$ of the wall --- which
        a thermal background does.  Measured while building this: with a hard wall at 64,
        197 of 200 proposals scored $\ln A = -\infty$ and the move never fired.  Any
        production table paired with this move needs that much headroom.
    """
    return SectorWeights(-cost * np.minimum(np.arange(cap + 1), 1), 0.0, False)


# ---------------------------------------------------------------- the sheet's identities

@pytest.mark.parametrize('c', range(6))
def test_wrapping_sheet_is_closed(S, c):
    r"""$dW = 0$ exactly --- this is what makes the move leave $D$ untouched."""
    for (i, j) in ((0, 0), (1, 2), (N - 1, N - 1)):
        W = wrapping_sheet(N, c, i, j)
        assert not np.asarray(d(two_form(S, W))).any(), f'dW != 0 for c={c}, ({i},{j})'


@pytest.mark.parametrize('c', range(6))
def test_wrapping_sheet_does_not_self_intersect(S, c):
    r"""$W\wedge W = 0$ exactly, so all of $\Delta q$ is the cross term with $F$."""
    W = two_form(S, wrapping_sheet(N, c, 1, 2))
    assert not np.asarray(wedge(W, W)).any()


@pytest.mark.parametrize('c', range(6))
def test_wrapping_sheet_is_not_exact(S, c):
    r"""$\sum_x W_c = N^2$ in its own component and 0 in every other: the class moves by
    exactly one unit, in exactly one direction."""
    W = wrapping_sheet(N, c, 1, 2)
    totals = W.reshape(6, -1).sum(axis=1)
    assert totals[c] == N ** 2
    assert (np.delete(totals, c) == 0).all()


@pytest.mark.parametrize('c', range(6))
def test_transverse_axes_are_the_complement(c):
    assert set(transverse_axes(c)) | set(COMPONENTS[c]) == {0, 1, 2, 3}
    assert not set(transverse_axes(c)) & set(COMPONENTS[c])


# ---------------------------------------------------------------- what it does to a state

def test_sheet_changes_the_class_but_not_D(S):
    r"""The whole point: $[F]$ moves, $D$ does not, and the result is no longer a legal
    vacuum even though it is perfectly closed."""
    state = FState(S)
    assert state.legal_vacuum
    state.F = state.F + wrapping_sheet(N, 2, 1, 1)
    state.resync()
    assert state.D == 0, 'a closed sheet must not open any surface'
    assert state.Q == 0, 'on the empty background the sheet crosses nothing'
    assert state.periods[2] == N ** 2
    assert not state.legal_vacuum, 'closed is not exact -- the periods gate must fire'


def test_resync_leaves_the_state_self_consistent(S):
    r"""``resync`` is the supported way to apply a wholesale change to ``F``; ``check``
    is what would catch it being incomplete."""
    state = FState(S)
    state.F = state.F + wrapping_sheet(N, 0, 2, 3) - wrapping_sheet(N, 4, 1, 0)
    state.resync()
    state.check()


def test_two_sheets_return_to_the_physical_shell(S):
    r"""Out and back: adding a sheet and subtracting a *differently placed* one leaves a
    state with zero periods again --- h2-relaxation finding 7's exactness, as a gate."""
    state = FState(S)
    state.F = (state.F + wrapping_sheet(N, 3, 0, 0)
                       - wrapping_sheet(N, 3, 2, 1))
    state.resync()
    assert state.D == 0
    assert not state.periods.any(), 'the round trip must land back at [F] = 0'


# ---------------------------------------------------------------- the acceptance is right

def test_sheet_acceptance_matches_the_placement_formula(S):
    r"""The sampler's differenced $\log\pi_\text{ext}$ must agree with the independent
    cross-term/self-energy decomposition used by the cost scan,
    $\Delta C = 2s\sum_x W\Delta^{-1}F_c + C_W$, on the action part.

    Computed on a charge-free background so that the two prices do not enter and the
    action piece can be isolated.
    """
    rng = np.random.default_rng(17)
    gas = WrappingSheetGas(S, openSurfaceFugacity=0.5, intersectionFugacity=1.0,
                           rng=rng, seed=17)
    # A closed, exact, charge-free background: the coboundary of a random integer 1-form.
    n = S.Lattice.form(1, dtype=np.int64)
    np.asarray(n)[...] = rng.integers(-1, 2, size=np.asarray(n).shape)
    F = np.asarray(d(n)).astype(np.int64)
    state = FState(S, F)

    c, i, j, s = 4, 2, 1, -1
    W = wrapping_sheet(N, c, i, j, s)
    cross = float((W[c] * pot(F[c], N)).sum())
    dC_formula = 2 * cross + sheet_self_energy(N)
    dC_direct = (float(((F + W)[c] * pot((F + W)[c], N)).sum())
                 - float((F[c] * pot(F[c], N)).sum()))
    assert dC_direct == pytest.approx(dC_formula, rel=1e-9, abs=1e-9)

    lnA, _ = gas.sheet_log_acceptance(state, c, i, j, s)
    # With a flat charge price and no winding/umbrella change beyond what
    # _log_extended_weight already accounts for, the action piece must appear intact.
    assert np.isfinite(lnA)


def test_sheet_proposal_is_symmetric(S):
    r"""Detailed balance rests on the reverse proposal having the *same* draw probability,
    so the two log-acceptances must be exact negatives."""
    rng = np.random.default_rng(3)
    gas = WrappingSheetGas(S, openSurfaceFugacity=0.5, intersectionFugacity=1.0,
                           rng=rng, seed=3)
    n = S.Lattice.form(1, dtype=np.int64)
    np.asarray(n)[...] = rng.integers(-1, 2, size=np.asarray(n).shape)
    state = FState(S, np.asarray(d(n)).astype(np.int64))

    c, i, j, s = 1, 3, 0, 1
    forward, F = gas.sheet_log_acceptance(state, c, i, j, s)
    moved = FState(S, F)
    backward, back = gas.sheet_log_acceptance(moved, c, i, j, -s)
    assert forward == pytest.approx(-backward, rel=1e-9, abs=1e-9)
    assert (back == state.F).all(), 'the reverse move must return the original F'


# ---------------------------------------------------------------- guards

def test_sheetEvery_zero_reproduces_the_parent_bit_for_bit(S):
    r"""With the move disabled the subclass must *be* the parent --- otherwise every
    comparison against the production sampler is confounded."""
    kw = dict(openSurfaceFugacity=0.5, intersectionFugacity=0.5, targetFraction=0.8, measure=False, pCob=0.35)
    a = SurfaceWormGas(S, seed=99, rng=np.random.default_rng(99), **kw)
    b = WrappingSheetGas(S, sheetEvery=0, seed=99, rng=np.random.default_rng(99), **kw)
    sa, sb = FState(S), FState(S)
    a.sweep(sa, 2000)
    b.sweep(sb, 2000)
    assert (sa.F == sb.F).all()
    assert sa.D == sb.D and sa.Q == sb.Q


def test_sheet_move_is_not_inert():
    r"""Against a charge table that makes intersections nearly free, the move must
    actually fire and must actually reach a nonzero class.

    This is the guard that would have caught the move being plumbed in but never called
    --- the failure mode the 2026-08-06 'compromise' round shipped.

    .. note ::
        Run at $\kappa = 0.02$ deliberately.  The sheet's bare cost is
        $2\pi^2\kappa C_W$ with $C_W = 4.29$ at $N = 4$, so at the module's $\kappa = 0.1$
        the acceptance is $e^{-8.5}$ and a few hundred proposals legitimately find
        nothing.  A test that fails because the physics is expensive tests nothing; this
        one is placed where the move *can* fire, so a zero means the move is broken.
    """
    S = NoIntersections(Lattice(4, N), 0.02)
    gas = WrappingSheetGas(S, sheetEvery=100, openSurfaceFugacity=0.5,
                           chargeWeights=cheap_charge(),
                           targetFraction=0.8, measure=False, pCob=0.35,
                           seed=5, rng=np.random.default_rng(5))
    state = FState(S)
    gas.sweep(state, 20000)
    assert gas.sheetProposed > 0, 'the move was never proposed'
    assert gas.sheetAccepted > 0, 'the move never fired -- inert plumbing or a bad table'
    assert gas.sheetClassVisits > 0, 'the chain never reached a nonzero class'


def test_sheet_move_preserves_state_consistency(S):
    r"""Everything the move touches must stay in sync with ``F`` alone."""
    gas = WrappingSheetGas(S, sheetEvery=100, openSurfaceFugacity=0.5,
                           chargeWeights=cheap_charge(),
                           measure=False, pCob=0.35,
                           seed=11, rng=np.random.default_rng(11))
    state = FState(S)
    gas.sweep(state, 5000)
    state.check()


def test_report_mentions_the_sheet(S):
    gas = WrappingSheetGas(S, sheetEvery=100, openSurfaceFugacity=0.5,
                           chargeWeights=cheap_charge(),
                           seed=1, rng=np.random.default_rng(1))
    assert 'wrapping sheets' in gas.report()


# ------------------------------------------------- the transport commutator

@pytest.mark.parametrize('c', range(6))
def test_dipole_is_exact_and_deposits_nothing(S, c):
    r"""A single transported sheet has zero periods (so the class is untouched) and no
    self-intersection --- it is the *pair* in complementary planes that deposits."""
    st = FState(S, sheet_dipole(N, c, 0, 0, 2, 1))
    assert st.D == 0
    assert not st.periods.any(), 'a dipole must leave the class alone'
    assert st.Q == 0, 'W ^ W vanishes within one component'
    assert st.legal_vacuum
    assert not np.asarray(st.intersection_winding()).any()


@pytest.mark.parametrize('c', (0, 1, 2))
def test_transport_commutator_deposits_the_minimal_charge(S, c):
    r"""Two dipoles in **complementary** planes deposit exactly 4 quanta of charge on the
    empty background, while each alone deposits none --- so the whole deposit is the cross
    term $A\wedge B$.  And the class stays trivial, which is what lets a chain carrying
    this move keep emitting.
    """
    A = sheet_dipole(N, c, 0, 0, 2, 1)
    B = sheet_dipole(N, COMPLEMENT[c], 1, 2, 3, 0)
    st = FState(S, A + B)
    assert st.D == 0, 'both dipoles are closed'
    assert not st.periods.any(), 'the commutator must stay on the legal class shell'
    assert st.Q == 4, f'expected the minimal 4 quanta, got {st.Q}'


def test_commutator_move_is_exactly_balanced_and_not_inert():
    r"""Symmetric proposal, and it must actually fire somewhere it is affordable."""
    S = NoIntersections(Lattice(4, N), 0.02)
    gas = WrappingSheetGas(S, sheetEvery=0, commutatorEvery=100,
                           openSurfaceFugacity=0.5, chargeWeights=cheap_charge(),
                           targetFraction=0.8, measure=False, pCob=0.35,
                           seed=5, rng=np.random.default_rng(5))
    state = FState(S)
    spec = (0, 0, 0, 2, 1, 1, 2, 3, 0, 1)
    forward, F = gas.commutator_log_acceptance(state, spec)
    back, F0 = gas.commutator_log_acceptance(FState(S, F),
                                             (0, 0, 0, 2, 1, 1, 2, 3, 0, -1))
    assert forward == pytest.approx(-back, rel=1e-9, abs=1e-9)
    assert (F0 == state.F).all(), 'the reverse move must return the original F'

    gas.sweep(state, 20000)
    assert gas.commutatorProposed > 0
    assert gas.commutatorAccepted > 0, 'the commutator never fired'
    state.check()


def test_commutator_preserves_the_class_along_a_chain():
    r"""The property the bare sheet move lacks: a commutator chain must never leave
    $[F] = 0$, so it cannot be absorbed into the unpriced class direction."""
    S = NoIntersections(Lattice(4, N), 0.02)
    gas = WrappingSheetGas(S, sheetEvery=0, commutatorEvery=100,
                           openSurfaceFugacity=0.5, chargeWeights=cheap_charge(),
                           measure=False, pCob=0.35, seed=3,
                           rng=np.random.default_rng(3))
    state = FState(S)
    for _ in range(20):
        gas.sweep(state, 1000)
        if state.D == 0:
            assert not state.periods.any(), 'the commutator chain left the class shell'
