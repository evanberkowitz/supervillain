#!/usr/bin/env python

import numpy as np
import supervillain
import supervillain.generator.no_intersection as gen
from supervillain.lattice import Lattice, Form, d
from supervillain.generator.no_intersection.charge import charge


def _worm(N=6, kappa=0.3):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    return gen.FreeTargetWorm(S)


def _key(shape):
    return frozenset(shape)


def _valid_configs(kappa=0.3, N=6, steps=140, burn=100, stride=8):
    L = Lattice(4, N)
    S = supervillain.action.NoIntersections(L, kappa=kappa)
    H = gen.Hammer(S, zeta=0.025)
    e = supervillain.Ensemble(S).generate(steps, H, start='cold')
    return S, [e.configuration[i] for i in range(burn, steps, stride)]


def _apply(n_arr, change):
    trial = n_arr.copy()
    for link, c in change.items():
        trial[link] += c
    return trial


def _defects(L, trial, q0):
    dq = charge(Form(trial, degree=1, lattice=L)) - q0
    return {tuple(int(x) for x in z[1:]): int(dq[tuple(z)])
            for z in np.argwhere(dq != 0)}


def test_family_is_deduped_and_negation_closed():
    w = _worm()
    keys = [_key(s) for s in w._candidate_family]
    assert len(keys) == len(set(keys))                     # no duplicate placements
    keyset = set(keys)
    for shape in w._candidate_family:
        neg = _key(tuple((mu, r, -c) for mu, r, c in shape))
        assert neg in keyset                               # negation-closed
        assert shape in w._self_charge                     # self-charge registered


def test_family_touches_the_head():
    # Support-anchoring: every family member's charge-reach support contains the
    # head (the origin, in relative coordinates) through at least one link.
    w = _worm()
    origin = (0, 0, 0, 0)
    for shape in w._candidate_family:
        assert any(origin in w._slot_support(mu, r) for mu, r, _c in shape)


def test_family_contains_singles_pairs_and_library():
    w = _worm()
    sizes = {len(shape) for shape in w._candidate_family}
    assert 1 in sizes and 2 in sizes                       # singles and pairs
    assert max(sizes) >= 3                                 # library 3-/4-link templates


def test_classified_set_reference_classifies_exactly():
    # The family is ~2e5 shapes and the reference does a global charge() recompute per
    # shape, so full-family reference enumerations are hand-run only.  Here the oracle's
    # semantic contract is checked on random SUBSAMPLES of the family (the ``shapes``
    # parameter): everything it returns is exactly an idle or a head-rooted dipole,
    # deduped.  (Full-set claims are covered by the local/kernel equalities of Tasks
    # 5 and 7 plus the pointwise subsample equality of Task 5.)
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    L, N = S.Lattice, S.Lattice.N
    rng = np.random.default_rng(0)
    movers = idles = 0
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        q0 = charge(cfg['n'])
        for _ in range(2):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            sub = [w._candidate_family[i] for i in
                   rng.choice(len(w._candidate_family), size=1500, replace=False)]
            C = w.classified_set_reference(n_arr, q0, head, shapes=sub)
            keys = [frozenset((l, c) for l, c in ch.items() if c != 0) for ch, _ in C]
            assert len(keys) == len(set(keys))              # deduped
            for change, target in C:
                dq_map = _defects(L, _apply(n_arr, change), q0)
                if target == head:
                    assert dq_map == {}                     # idle: Δq ≡ 0
                    idles += 1
                else:
                    assert dq_map == {head: -1, target: 1}  # exact dipole, any y
                    movers += 1
    assert movers + idles > 0


def test_local_py_matches_reference_on_subsamples():
    # Pointwise global-vs-local equality on random family subsamples (full-family
    # reference calls are infeasible; the subsample makes the comparison exact on the
    # shapes it covers, movers, idles, and discards alike).
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    N = S.Lattice.N
    rng = np.random.default_rng(3)
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        q0 = charge(cfg['n'])
        for _ in range(2):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            sub = [w._candidate_family[i] for i in
                   rng.choice(len(w._candidate_family), size=2000, replace=False)]
            assert (w.classified_set_local_py(F, head, shapes=sub)
                    == w.classified_set_reference(n_arr, q0, head, shapes=sub))


def test_full_local_enumeration_is_clean_and_reaches_beyond_orthogonal():
    # Full-family local enumeration (tractable); every CLEAN element it returns is
    # verified by a global recompute (clean sets are small, so this is cheap), and the
    # merged worm's raison d'etre -- transport beyond the orthogonal menu -- appears.
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    L, N = S.Lattice, S.Lattice.N
    rng = np.random.default_rng(4)
    targets = set()
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        q0 = charge(cfg['n'])
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        C = w.classified_set_local_py(F, head)
        keys = [frozenset((l, c) for l, c in ch.items() if c != 0) for ch, _ in C]
        assert len(keys) == len(set(keys))                  # deduped
        for change, target in C:
            dq_map = _defects(L, _apply(n_arr, change), q0)
            if target == head:
                assert dq_map == {}
            else:
                assert dq_map == {head: -1, target: 1}
                targets.add(tuple((target[k] - head[k]) % N for k in range(4)))
    assert any(sum(min(x, N - x) for x in t) > 1 for t in targets)


def test_step_reference_emits_valid_configs_with_populated_origin():
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    w.rng = np.random.default_rng(7)
    L = S.Lattice
    cfg = configs[0]
    for _ in range(10):
        cfg = w.step_reference(cfg)
        assert np.abs(charge(cfg['n'])).max() == 0          # constraint preserved
        theta = cfg['IntersectionTwoPoint']
        assert theta[L.origin] >= 1                          # pre-populated pivot dwell
        assert cfg['Worm_Length'] == theta.sum()
        assert cfg['Worm_Length'] >= 1


# This checks the |C'| >= 1 guarantee (no zero division) and formula
# self-consistency (the two arithmetic expressions for the closing balance agree);
# it does NOT exercise the actual accept/reject branch in _run_free_worm -- that
# acceptance-logic coverage lives in test_acceptance_boundary_matches_hastings_ratio
# below, which drives the walk with a scripted rng.
def test_elementary_detailed_balance_no_menu_factor():
    # A full-family classified_set_reference (no `shapes`) is a global charge()
    # recompute per shape over ~3e5 shapes -- infeasible here (~hours).  Task 5 proved
    # classified_set_local_py == classified_set_reference pointwise, so C/Cp are drawn
    # from the local (F-driven) enumeration instead; every assertion is unchanged.
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    L, N = S.Lattice, S.Lattice.N
    rng = np.random.default_rng(4)
    tested = 0
    for cfg in configs:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        for _ in range(3):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            C = w.classified_set_local_py(F, head)
            if not C:
                continue
            change, target = C[int(rng.integers(0, len(C)))]
            trial = _apply(n_arr, change)
            Fp = np.asarray(d(Form(trial, degree=1, lattice=L))).astype(np.int64)
            Cp = w.classified_set_local_py(Fp, target)
            dS = w._delta_S(dphi, n_arr, change)
            A_fwd = min(1.0, (len(C) / len(Cp)) * np.exp(-dS))
            A_rev = min(1.0, (len(Cp) / len(C)) * np.exp(+dS))
            # w(s) q(s->s') A(s->s') = w(s') q(s'->s) A(s'->s), with q = 1/|C|: NO menu factor.
            assert abs(A_fwd / len(C) - np.exp(-dS) * A_rev / len(Cp)) < 1e-12
            tested += 1
    assert tested > 0


def test_compiled_classified_set_matches_python():
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    N = S.Lattice.N
    rng = np.random.default_rng(5)
    for cfg in configs:
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        for _ in range(4):
            head = tuple(int(x) for x in rng.integers(0, N, size=4))
            assert (w.classified_set_local(F, head)
                    == w.classified_set_local_py(F, head))


def test_step_matches_reference_bit_for_bit():
    # Also the cache validator: step reuses the reverse enumeration across
    # iterations, step_reference recomputes fresh; caching consumes no RNG, so any
    # stale cache shows up as a trajectory divergence.
    S, configs = _valid_configs()
    cfg = configs[0]
    a = gen.FreeTargetWorm(S)
    b = gen.FreeTargetWorm(S)
    for seed in (2024, 2025, 2026):
        a.rng = np.random.default_rng(seed)
        b.rng = np.random.default_rng(seed)
        ra = a.step(cfg)
        rb = b.step_reference(cfg)
        assert np.array_equal(np.asarray(ra['n']), np.asarray(rb['n']))
        assert np.array_equal(ra['IntersectionTwoPoint'],
                              rb['IntersectionTwoPoint'])
        assert ra['Worm_Length'] == rb['Worm_Length']
        cfg = rb


def test_closure_involution_over_all_realized_targets():
    # For each clean mover, the negated change must be enumerated from its target
    # (facts (1)+(2)); checked with the full local enumeration on both ends.  Movers
    # per head are capped to keep the runtime sane; idles are checked at fixed head.
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    N = S.Lattice.N
    rng = np.random.default_rng(5)
    checked = 0
    for cfg in configs[:3]:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        C = w.classified_set_local_py(F, head)
        movers = [(ch, t) for ch, t in C if t != head]
        idles = [(ch, t) for ch, t in C if t == head]
        picks = ([movers[i] for i in rng.choice(len(movers),
                                                size=min(6, len(movers)),
                                                replace=False)] if movers else []) \
            + ([idles[i] for i in rng.choice(len(idles),
                                             size=min(3, len(idles)),
                                             replace=False)] if idles else [])
        for change, target in picks:
            trial = _apply(n_arr, change)
            Fp = np.asarray(d(Form(trial, degree=1, lattice=S.Lattice))).astype(np.int64)
            Cp = w.classified_set_local_py(Fp, target)
            inv = frozenset((l, -c) for l, c in change.items() if c != 0)
            assert any(frozenset((l, c) for l, c in ch.items() if c != 0) == inv
                       and tgt == head
                       for ch, tgt in Cp)                   # closure: reverse enumerated
            checked += 1
    assert checked > 0


def test_runs_in_ensemble_and_normalizes():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    w = gen.FreeTargetWorm(S)
    e = supervillain.Ensemble(S).generate(60, w, start='cold')
    q2 = np.asarray(e.TopologicalChargeDensitySquared)
    assert np.abs(q2).max() == 0                            # every emitted config valid
    theta = np.asarray(e.IntersectionTwoPoint)
    origins = theta[(slice(None),) + L.origin]
    assert (origins >= 1).all()                             # origin >= 1 per worm, by construction
    mean = theta.mean(axis=0)
    normalized = mean / mean[L.origin]
    assert normalized[L.origin] == 1
    assert 'free-target worms' in w.report()


class _ScriptedRNG:
    r"""Feeds _run_free_worm a predetermined draw sequence."""
    def __init__(self, integers_values, uniform_values):
        self._ints = list(integers_values)
        self._unis = list(uniform_values)

    def integers(self, low, high=None, size=None):
        v = self._ints.pop(0)
        if size is not None:
            return np.asarray(v)
        return v

    def uniform(self, low, high):
        return self._unis.pop(0)


def test_acceptance_boundary_matches_hastings_ratio():
    # Drives the walk's accept line directly: with the uniform draw just BELOW the
    # independently computed min(1, (|C|/|C'|) e^{-dS}) the mover must be applied,
    # just ABOVE it must be rejected.  A sign flip in dS or an inverted count ratio
    # moves the boundary and fails this test -- the discriminator the shared
    # _run_free_worm bit-for-bit test cannot provide.
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    w = gen.FreeTargetWorm(S)
    cold = S.configurations(1)[0]
    n_arr = np.asarray(cold['n']).astype(np.int64)
    dphi = np.asarray(d(cold['phi']))
    F = np.asarray(d(cold['n'])).astype(np.int64)
    tail = (2, 2, 2, 2)
    C = w.classified_set_local_py(F, tail)
    k, (change, target) = next((i, ct) for i, ct in enumerate(C) if ct[1] != tail)
    trial = n_arr.copy()
    for lnk, c in change.items():
        trial[lnk] += c
    Fp = np.asarray(d(Form(trial, degree=1, lattice=L))).astype(np.int64)
    Cp = w.classified_set_local_py(Fp, target)
    dS = w._delta_S(dphi, n_arr, change)
    A = min(1.0, (len(C) / len(Cp)) * np.exp(-dS))
    assert 0 < A < 1                                   # boundary is nontrivial
    inv = frozenset((l, -c) for l, c in change.items() if c != 0)
    j = next(i for i, (ch, tgt) in enumerate(Cp)
             if frozenset((l, c) for l, c in ch.items() if c != 0) == inv
             and tgt == tail)
    A_rev = min(1.0, (len(Cp) / len(C)) * np.exp(+dS))

    def run(first_uniform):
        w.rng = _ScriptedRNG(
            integers_values=[list(tail), k, j],
            uniform_values=[first_uniform, min(1.0, A_rev) - 1e-12],
        )
        return w.step_reference(cold)

    accepted = run(A - 1e-9)                            # just below: mover applies,
    theta = accepted['IntersectionTwoPoint']       # reverse brings it home
    disp = tuple((target[m] - tail[m]) % L.N for m in range(4))
    assert theta[disp] == 1 and theta[L.origin] == 1 and theta.sum() == 2
    assert np.array_equal(np.asarray(accepted['n']), n_arr)   # net change zero

    w.rng = _ScriptedRNG(integers_values=[list(tail), k], uniform_values=[A + 1e-9])
    rejected = w.step_reference(cold)                   # just above: zero-length worm
    assert np.array_equal(np.asarray(rejected['n']), n_arr)
    assert rejected['IntersectionTwoPoint'].sum() == 1
    assert rejected['Worm_Length'] == 1


def test_classes_filter_matches_manual_split():
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S)
    N = S.Lattice.N
    rng = np.random.default_rng(11)
    for cfg in configs[:2]:
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        both = w.classified_set_local(F, head)
        movers = w.classified_set_local(F, head, classes='movers')
        idles = w.classified_set_local(F, head, classes='idles')
        assert movers == [(ch, t) for ch, t in both if t != head]
        assert idles == [(ch, t) for ch, t in both if t == head]
        assert len(movers) > 0 and len(idles) > 0          # hot backgrounds have both
        # the pure-Python twin filters identically (bit-for-bit contract per class)
        assert movers == w.classified_set_local_py(F, head, classes='movers')
        assert idles == w.classified_set_local_py(F, head, classes='idles')
        # the oracle agrees on a subsample
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        q0 = charge(cfg['n'])
        sub = [w._candidate_family[i] for i in
               rng.choice(len(w._candidate_family), size=1500, replace=False)]
        assert (w.classified_set_reference(n_arr, q0, head, shapes=sub, classes='movers')
                == w.classified_set_local_py(F, head, shapes=sub, classes='movers'))


def test_idle_probability_validation():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    for bad in (1.0, 1.5, -0.1):
        try:
            gen.FreeTargetWorm(S, idle_probability=bad)
            assert False, f'idle_probability={bad} should raise'
        except ValueError:
            pass
    for ok in (None, 0.0, 0.5):
        gen.FreeTargetWorm(S, idle_probability=ok)


def test_modes_bit_for_bit():
    # step (compiled + reuse where applicable) vs step_reference (pure Python, fresh)
    # in the movers-only and two-slot modes; the flat default is already covered.
    S, configs = _valid_configs()
    cfg = configs[0]
    for p in (0.0, 0.3):
        a = gen.FreeTargetWorm(S, idle_probability=p)
        b = gen.FreeTargetWorm(S, idle_probability=p)
        for seed in (5150, 5151):
            a.rng = np.random.default_rng(seed)
            b.rng = np.random.default_rng(seed)
            ra = a.step(cfg)
            rb = b.step_reference(cfg)
            assert np.array_equal(np.asarray(ra['n']), np.asarray(rb['n']))
            assert np.array_equal(ra['IntersectionTwoPoint'],
                                  rb['IntersectionTwoPoint'])
            assert ra['Worm_Length'] == rb['Worm_Length']
            cfg = rb


def test_movers_only_elementary_detailed_balance():
    # Per-slot balance: q = 1/|M| forward, 1/|M'| reverse, no slot factor (it cancels).
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S, idle_probability=0.0)
    L, N = S.Lattice, S.Lattice.N
    rng = np.random.default_rng(6)
    tested = 0
    for cfg in configs[:3]:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        M = w.classified_set_local_py(F, head, classes='movers')
        if not M:
            continue
        for _ in range(4):
            change, target = M[int(rng.integers(0, len(M)))]
            trial = _apply(n_arr, change)
            Fp = np.asarray(d(Form(trial, degree=1, lattice=L))).astype(np.int64)
            Mp = w.classified_set_local_py(Fp, target, classes='movers')
            inv = frozenset((l, -c) for l, c in change.items() if c != 0)
            assert any(frozenset((l, c) for l, c in ch.items() if c != 0) == inv
                       and tgt == head for ch, tgt in Mp)   # reverse mover enumerated
            dS = w._delta_S(dphi, n_arr, change)
            A_fwd = min(1.0, (len(M) / len(Mp)) * np.exp(-dS))
            A_rev = min(1.0, (len(Mp) / len(M)) * np.exp(+dS))
            assert abs(A_fwd / len(M) - np.exp(-dS) * A_rev / len(Mp)) < 1e-12
            tested += 1
    assert tested > 0


def test_movers_only_acceptance_boundary():
    # Scripted-rng straddle of min(1, (|M|/|M'|) e^{-dS}) in movers-only mode --
    # the discriminator for the mode's accept line (the shared-walk bit-for-bit
    # test cannot catch a sign or ratio error).
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.3)
    w = gen.FreeTargetWorm(S, idle_probability=0.0)
    cold = S.configurations(1)[0]
    n_arr = np.asarray(cold['n']).astype(np.int64)
    dphi = np.asarray(d(cold['phi']))
    F = np.asarray(d(cold['n'])).astype(np.int64)
    tail = (2, 2, 2, 2)
    M = w.classified_set_local_py(F, tail, classes='movers')
    k, (change, target) = next(
        (i, ct) for i, ct in enumerate(M)
        if 0 < min(1.0, (len(M) / len(w.classified_set_local_py(
            np.asarray(d(Form(_apply(n_arr, ct[0]), degree=1, lattice=L))).astype(np.int64),
            ct[1], classes='movers')))
            * np.exp(-w._delta_S(dphi, n_arr, ct[0]))) < 1)
    trial = _apply(n_arr, change)
    Fp = np.asarray(d(Form(trial, degree=1, lattice=L))).astype(np.int64)
    Mp = w.classified_set_local_py(Fp, target, classes='movers')
    dS = w._delta_S(dphi, n_arr, change)
    A = min(1.0, (len(M) / len(Mp)) * np.exp(-dS))
    inv = frozenset((l, -c) for l, c in change.items() if c != 0)
    j = next(i for i, (ch, tgt) in enumerate(Mp)
             if frozenset((l, c) for l, c in ch.items() if c != 0) == inv
             and tgt == tail)
    A_rev = min(1.0, (len(Mp) / len(M)) * np.exp(+dS))

    w.rng = _ScriptedRNG(integers_values=[list(tail), k, j],
                         uniform_values=[A - 1e-9, min(1.0, A_rev) - 1e-12])
    accepted = w.step_reference(cold)
    theta = accepted['IntersectionTwoPoint']
    disp = tuple((target[m] - tail[m]) % L.N for m in range(4))
    assert theta[disp] == 1 and theta[L.origin] == 1 and theta.sum() == 2
    assert np.array_equal(np.asarray(accepted['n']), n_arr)

    w.rng = _ScriptedRNG(integers_values=[list(tail), k], uniform_values=[A + 1e-9])
    rejected = w.step_reference(cold)
    assert np.array_equal(np.asarray(rejected['n']), n_arr)
    assert rejected['IntersectionTwoPoint'].sum() == 1
    assert rejected['Worm_Length'] == 1


def test_two_slot_idle_elementary_detailed_balance():
    # The two-slot mode's idle branch: q = p_idle/|I| forward, p_idle/|I'| reverse at the
    # SAME head -- the slot probability cancels, leaving |I|/|I'|.  The reverse of an
    # accepted idle must be enumerated among the idles at the same head on the applied
    # background (reversal identity + negation closure), which is also what the walk's
    # reverse-class keying (target == head -> 'idles') relies on.
    S, configs = _valid_configs()
    w = gen.FreeTargetWorm(S, idle_probability=0.5)
    L, N = S.Lattice, S.Lattice.N
    rng = np.random.default_rng(8)
    tested = 0
    for cfg in configs[:3]:
        n_arr = np.asarray(cfg['n']).astype(np.int64)
        dphi = np.asarray(d(cfg['phi']))
        F = np.asarray(d(cfg['n'])).astype(np.int64)
        head = tuple(int(x) for x in rng.integers(0, N, size=4))
        I = w.classified_set_local_py(F, head, classes='idles')
        if not I:
            continue
        for _ in range(4):
            change, target = I[int(rng.integers(0, len(I)))]
            assert target == head                           # idles never move the head
            trial = _apply(n_arr, change)
            Fp = np.asarray(d(Form(trial, degree=1, lattice=L))).astype(np.int64)
            Ip = w.classified_set_local_py(Fp, head, classes='idles')
            inv = frozenset((l, -c) for l, c in change.items() if c != 0)
            assert any(frozenset((l, c) for l, c in ch.items() if c != 0) == inv
                       for ch, _ in Ip)                     # reverse idle enumerated
            dS = w._delta_S(dphi, n_arr, change)
            A_fwd = min(1.0, (len(I) / len(Ip)) * np.exp(-dS))
            A_rev = min(1.0, (len(Ip) / len(I)) * np.exp(+dS))
            assert abs(A_fwd / len(I) - np.exp(-dS) * A_rev / len(Ip)) < 1e-12
            tested += 1
    assert tested > 0
