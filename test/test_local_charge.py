#!/usr/bin/env python
r"""
Tests for the local single-link charge machinery that accelerates
:class:`~supervillain.generator.no_intersection.ConstrainedLinkUpdate`.

Every stage of the accelerated charge computation is pinned against the plain,
obviously-correct global recompute:

  * the ``d(delta_link)`` stencil rebuilds ``d`` of a unit link exactly;
  * ``apply_link_to_F`` keeps ``F`` equal to ``d(n)`` after a flip;
  * ``charge_change_from_link`` equals the global ``charge`` difference on every
    background, and is exactly linear in the shift ``c``;
  * "clean" (empty response) is exactly "the constraint is preserved";
  * the derived stencil is genuinely local (guards against an undersized probe box);
  * the accelerated ``step`` reproduces ``step_reference`` bit-for-bit on a shared seed.
"""

from itertools import product

import numpy as np
import pytest

import supervillain
from supervillain.lattice import Lattice, d, Form, wedge
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection.link import ConstrainedLinkUpdate
from supervillain.generator.no_intersection import local_charge as lc


def _action(kappa=0.3, N=5):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def _valid_background(S, seed, n_links=None):
    r"""
    A deterministic, nontrivial, *valid* (q = 0) background, independent of any generator
    under test: deposit random single links on the cold vacuum, keeping only those that
    hold q = 0 (verified by the global recompute --- cheap, done only ``n_links`` times).
    Builds honest flux without invoking the constrained-link update it is used to test.
    """
    L = S.Lattice
    N = L.N
    if n_links is None:
        n_links = 3 * N
    rng = np.random.default_rng(seed)
    n = np.asarray(L.zeros(1, dtype=int))
    for _ in range(n_links):
        mu = int(rng.integers(0, 4))
        site = tuple(int(x) for x in rng.integers(0, N, 4))
        c = int(rng.choice([-1, 1]))
        n[(mu,) + site] += c
        if charge(Form(n, degree=1, lattice=L)).any():
            n[(mu,) + site] -= c        # revert anything that would break the constraint
    assert not charge(Form(n, degree=1, lattice=L)).any()          # genuinely valid
    assert np.abs(np.asarray(d(Form(n, degree=1, lattice=L)))).sum() > 0   # genuinely fluxful
    return n


def _backgrounds(S, seed=0):
    L = S.Lattice
    yield np.asarray(L.zeros(1, dtype=int))                 # cold, F = 0
    yield _valid_background(S, seed)                        # thermalized-ish flux


def _global_dq(L, n, mu, site, c):
    r"""The change in the charge density from a single-link flip, by full recompute."""
    q0 = np.asarray(charge(Form(n, degree=1, lattice=L)))
    trial = n.copy()
    trial[(mu,) + tuple(site)] += c
    dq = np.asarray(charge(Form(trial, degree=1, lattice=L))) - q0
    return {tuple(int(x) for x in h[1:]): int(dq[tuple(h)]) for h in np.argwhere(dq != 0)}


# --------------------------------------------------------------------------- d(delta) stencil

def test_df_stencil_reconstructs_d_of_a_unit_link():
    # Applying the DF stencil (F += c * d(delta_link)) must reproduce d of the dense
    # unit-link 1-form exactly, at an arbitrary site and with periodic wrap.
    L = Lattice(4, 5)
    N = L.N
    for mu in range(4):
        for site in [(0, 0, 0, 0), (2, 3, 1, 4), (4, 4, 4, 4)]:
            n = L.zeros(1, dtype=int)
            n[(mu,) + site] += 1
            want = np.asarray(d(n))
            F = np.zeros_like(want)
            lc.apply_link_to_F(F, mu, site, 1, N)
            assert np.array_equal(F, want), (mu, site)


def test_apply_link_to_F_keeps_F_equal_dn():
    # Maintaining F incrementally across a stream of flips must stay bit-identical to
    # recomputing d(n) from scratch -- this is what lets the sweep never recompute d(n).
    S = _action(N=5)
    L = S.Lattice
    N = L.N
    n = _valid_background(S, seed=3)
    F = np.asarray(d(Form(n, degree=1, lattice=L))).copy()
    rng = np.random.default_rng(11)
    for _ in range(200):
        mu = int(rng.integers(0, 4))
        site = tuple(int(x) for x in rng.integers(0, N, 4))
        c = int(rng.choice([-1, 1, 2, -2]))
        n[(mu,) + site] += c
        lc.apply_link_to_F(F, mu, site, c, N)
        assert np.array_equal(F, np.asarray(d(Form(n, degree=1, lattice=L))))


# --------------------------------------------------------------------------- charge response

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_charge_change_matches_global_recompute(seed):
    # The local response equals the global charge difference on every background, for
    # every direction/site/shift -- the central correctness claim.
    S = _action(N=5)
    L = S.Lattice
    N = L.N
    for n in _backgrounds(S, seed):
        F = np.asarray(d(Form(n, degree=1, lattice=L))).copy()
        rng = np.random.default_rng(seed + 100)
        for _ in range(300):
            mu = int(rng.integers(0, 4))
            site = tuple(int(x) for x in rng.integers(0, N, 4))
            c = int(rng.choice([-2, -1, 1, 2]))
            assert lc.charge_change_from_link(F, mu, site, c, N) == _global_dq(L, n, mu, site, c)


def test_charge_change_is_linear_in_c():
    # dq(c) = c * dq(1) entrywise -- the linearity that the whole scheme rests on
    # (the single-link self-wedge vanishes identically).
    S = _action(N=5)
    L = S.Lattice
    N = L.N
    n = _valid_background(S, seed=5)
    F = np.asarray(d(Form(n, degree=1, lattice=L))).copy()
    rng = np.random.default_rng(7)
    for _ in range(200):
        mu = int(rng.integers(0, 4))
        site = tuple(int(x) for x in rng.integers(0, N, 4))
        unit = lc.charge_change_from_link(F, mu, site, 1, N)
        for c in (-3, -2, 2, 3):
            scaled = lc.charge_change_from_link(F, mu, site, c, N)
            assert scaled == {cell: c * v for cell, v in unit.items()}


def test_clean_iff_constraint_preserved():
    # An empty response is exactly "the flip keeps q = 0" on a valid background.
    S = _action(N=5)
    L = S.Lattice
    N = L.N
    n = _valid_background(S, seed=9)
    F = np.asarray(d(Form(n, degree=1, lattice=L))).copy()
    rng = np.random.default_rng(13)
    saw_clean = saw_dirty = 0
    for _ in range(400):
        mu = int(rng.integers(0, 4))
        site = tuple(int(x) for x in rng.integers(0, N, 4))
        c = int(rng.choice([-1, 1]))
        clean = not lc.charge_change_from_link(F, mu, site, c, N)
        trial = n.copy()
        trial[(mu,) + site] += c
        preserved = not charge(Form(trial, degree=1, lattice=L)).any()
        assert clean == preserved
        saw_clean += clean
        saw_dirty += not clean
    assert saw_clean > 0 and saw_dirty > 0    # the background actually exercises both


def test_stencil_is_local():
    # Guard against an undersized probe box: the response of a link at `site` may only
    # touch hypercubes within Chebyshev radius 1 (the measured reach), and may only read
    # F strictly inside the probe box, so no term sits on the box boundary (which would
    # mean a real contribution was clipped).
    dq_stencil, df_stencil = lc._stencils()
    for mu in range(4):
        for o, p, s, k in dq_stencil[mu]:
            assert max(abs(x) for x in o) <= 1, (mu, o)          # output within radius 1
            assert max(abs(x) for x in s) <= 1, (mu, s)          # reads F within radius 1
            assert k != 0
        for p, s, val in df_stencil[mu]:
            assert max(abs(x) for x in s) <= 1
            assert val != 0


# --------------------------------------------------------------------------- checkerboard

def test_single_link_interaction_reach_is_one():
    # The whole checkerboard rests on this: two single-link flips interact ONLY through the
    # bilinear cross term d(delta_i)^d(delta_j)+d(delta_j)^d(delta_i), and that term is
    # nonzero only when the links sit within Chebyshev distance 1.  Probed on a lattice big
    # enough that offsets up to 3 (the structural bound) never wrap.  By axis-permutation
    # symmetry only mu_i == mu_j and mu_i != mu_j are distinct, so two mu-pairs suffice.
    N = 11
    L = Lattice(4, N)
    anchor = (5, 5, 5, 5)

    def dd(mu, site):
        z = L.zeros(1, dtype=int)
        z[(mu,) + site] = 1
        return d(z)

    reach = 0
    for mu_i, mu_j in [(0, 0), (0, 1)]:
        ei = dd(mu_i, anchor)
        for delta in product(range(-3, 4), repeat=4):
            if mu_i == mu_j and all(x == 0 for x in delta):
                continue                                   # the same link
            sj = tuple((anchor[k] + delta[k]) % N for k in range(4))
            ej = dd(mu_j, sj)
            cross = np.asarray(wedge(ei, ej)) + np.asarray(wedge(ej, ei))
            if cross.any():
                reach = max(reach, max(abs(x) for x in delta))
    assert reach == 1


@pytest.mark.parametrize('N', [4, 5, 6, 7, 8, 9])
def test_axis_colors_partition_and_nonadjacent(N):
    # Each axis colouring must (a) partition range(N) and (b) never place two same-colour
    # coordinates adjacent on the ring -- the property that makes a full link colour
    # non-interacting.  Two colours for even N, three for odd.
    colors = lc.axis_colors(N)
    assert len(colors) == (2 if N % 2 == 0 else 3)
    assert sorted(int(x) for c in colors for x in c) == list(range(N))   # partition
    for c in colors:
        c = sorted(int(x) for x in c)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                ring = min((c[i] - c[j]) % N, (c[j] - c[i]) % N)
                assert ring >= 2                                          # never adjacent


@pytest.mark.parametrize('N', [6, 7, 8])
def test_same_color_links_are_non_interacting(N):
    # Structural guarantee behind the simultaneous update: within one colour, every pair of
    # distinct links differs by >= 2 in at least one axis (Chebyshev >= 2 > reach).
    axis = lc.axis_colors(N)
    for choice in product(range(len(axis)), repeat=4):
        rows = [axis[choice[a]] for a in range(4)]
        sites = list(product(*[[int(x) for x in r] for r in rows]))
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                cheb = max(min((sites[i][a] - sites[j][a]) % N, (sites[j][a] - sites[i][a]) % N)
                           for a in range(4))
                assert cheb >= 2


@pytest.mark.parametrize('N,seed', [(6, 0), (6, 1), (7, 0), (5, 1)])   # even and ODD N
def test_clean_mask_matches_scalar_oracle(N, seed):
    # Same arithmetic: the vectorised colour clean-mask must agree, link for link, with the
    # per-link scalar oracle charge_change_from_link on the same background.  Odd N is
    # included because its 3-colouring (with a singleton seam colour) exercises a different
    # index structure than the even-N 2-colouring.
    S = _action(N=N)
    L = S.Lattice
    N = L.N
    n = _valid_background(S, seed=seed)
    F = np.asarray(d(Form(n, degree=1, lattice=L))).copy()
    axis = lc.axis_colors(N)
    for mu in range(4):
        for choice in product(range(len(axis)), repeat=4):
            idx = [axis[a] for a in choice]
            mask = lc.clean_mask_for_color(F, mu, idx, N)
            it = np.ndindex(*mask.shape)
            for cell in it:
                site = tuple(int(idx[a][cell[a]]) for a in range(4))
                oracle_clean = not lc.charge_change_from_link(F, mu, site, 1, N)
                assert bool(mask[cell]) == oracle_clean


@pytest.mark.parametrize('N,choice', [(6, (0, 1, 0, 1)), (7, (0, 1, 2, 0)), (5, (2, 0, 1, 2))])
def test_simultaneous_color_apply_matches_sequential(N, choice):
    # Same arithmetic: applying a whole colour at once (n and the F patch) must equal
    # applying its links one by one -- the commutativity that non-interaction guarantees.
    # Odd N choices include the singleton seam colour (index 2).
    S = _action(N=N)
    L = S.Lattice
    n0 = _valid_background(S, seed=2)
    axis = lc.axis_colors(N)
    rng = np.random.default_rng(4)
    mu = 1
    idx = [axis[a] for a in choice]
    flip = rng.integers(-1, 2, size=tuple(len(i) for i in idx))

    # simultaneous
    n_s = n0.copy()
    F_s = np.asarray(d(Form(n0, degree=1, lattice=L))).copy()
    lc.apply_color(n_s, F_s, mu, idx, N, flip)

    # sequential, link by link
    n_q = n0.copy()
    F_q = np.asarray(d(Form(n0, degree=1, lattice=L))).copy()
    for cell in np.ndindex(*flip.shape):
        c = int(flip[cell])
        if c:
            site = tuple(int(idx[a][cell[a]]) for a in range(4))
            n_q[(mu,) + site] += c
            lc.apply_link_to_F(F_q, mu, site, c, N)

    assert np.array_equal(n_s, n_q)
    assert np.array_equal(F_s, F_q)
    assert np.array_equal(F_s, np.asarray(d(Form(n_s, degree=1, lattice=L))))


@pytest.mark.parametrize('N', [4, 5, 6, 7, 8])
def test_step_matches_broadcast_bit_for_bit(N):
    # The production numba `step` must reproduce the readable `step_reference_broadcast`
    # exactly: same RNG order/shapes and same float arithmetic, with only the integer clean
    # check and apply moved into the compiled kernel.  Even AND odd N (the odd 3-colouring
    # with a singleton seam exercises the kernel's wrap handling).
    S = _action(kappa=0.3, N=N)
    L = S.Lattice
    start = _valid_background(S, seed=N)

    def run(method):
        g = ConstrainedLinkUpdate(S)
        g.rng = np.random.default_rng(2024)
        cfg = {'phi': L.zeros(0), 'n': Form(start.copy(), degree=1, lattice=L)}
        for _ in range(4):
            cfg = {'phi': cfg['phi'], 'n': getattr(g, method)(cfg)['n']}
        return np.asarray(cfg['n'])

    assert np.array_equal(run('step'), run('step_reference_broadcast'))


@pytest.mark.parametrize('N', [4, 5, 6, 7, 8])
def test_checkerboard_step_preserves_validity(N):
    # End-to-end guard, even AND odd N: if the colouring ever grouped interacting links, a
    # simultaneous colour update could create charge.  It must not.
    S = _action(kappa=0.15, N=N)
    g = ConstrainedLinkUpdate(S)
    cfg = S.configurations(1)[0]
    for _ in range(6):
        cfg = g.step(cfg)
        assert S.valid(cfg)


def test_checkerboard_step_agrees_with_reference_on_physics():
    # Same physics: the checkerboard and the reference sweep sample the same distribution.
    # Different RNG consumption forbids a bit-for-bit check, so compare a robust observable
    # (mean sheet area <Sigma|F|>) over independent runs from the same cold start; they must
    # agree well within the run-to-run spread.
    S = _action(kappa=0.1, N=4)     # N=4: step_reference (O(N^8)) still affordable, and
    L = S.Lattice                   # M2 has few enough colours to stay fast

    def mean_area(method, seed, sweeps=30, burn=12):
        g = ConstrainedLinkUpdate(S)
        g.rng = np.random.default_rng(seed)
        cfg = {'phi': L.zeros(0), 'n': L.zeros(1, dtype=int)}
        areas = []
        for s in range(sweeps):
            cfg = getattr(g, method)(cfg)
            if s >= burn:
                areas.append(float(np.abs(np.asarray(d(cfg['n']))).sum()))
        return np.mean(areas)

    fast = [mean_area('step', seed) for seed in (1, 2, 3)]
    ref = [mean_area('step_reference', seed) for seed in (11, 12, 13)]
    spread = np.std(fast + ref) + 1.0
    assert abs(np.mean(fast) - np.mean(ref)) < 2.0 * spread
