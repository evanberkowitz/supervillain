r"""The intersection count $Q$ is priced by an object, not a scalar.

The gate is the one the open-surface axis used when it migrated: a
:class:`~.pricing.Fugacity` **is** the affine special case, so a gas priced by one must
reproduce the bare-scalar sampler exactly.  These tests pin that, pin the plumbing (a
non-affine table has to actually reach the acceptance), and pin the measurement
normalizations --- where a silent misnormalization would hide, as it did once already
for ``FourDefectDistribution``.
"""

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice
from supervillain.generator.no_intersection.surface_worm.gas import SurfaceWormGas
from supervillain.generator.no_intersection.surface_worm.accumulator import CorrelatorAccumulator
from supervillain.generator.no_intersection.surface_worm.state import FState
from supervillain.generator.no_intersection.surface_worm.pricing import Fugacity, WeightTable

ETA = 0.05


def _gas(**kw):
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)
    kw.setdefault('openSurfaceFugacity', 0.2)
    kw.setdefault('sectorWeightCap', 512)
    return S, SurfaceWormGas(S, seed=5, **kw)


def _equivalent_table(eta, cap=64):
    r"""A hand-built table carrying exactly a fugacity's numbers, as a *general*
    ``WeightTable`` --- so the comparison exercises the table code path rather than
    recognizing a ``Fugacity``."""
    lg = np.log(eta)
    return WeightTable(lg * np.arange(cap + 1), lg, hardWall=False)


def test_fugacity_price_is_a_power_not_an_exponential():
    r"""$\eta^n$ exactly.  Going through $e^{n\log\eta}$ moves the last bit, and every
    stored ``Theta_Theta`` would then differ in its final digits from the pre-pricing
    runs."""
    f = Fugacity(ETA)
    for n in (0, 1, 2, 3, 4, 7):
        assert f.price(n) == ETA ** n
    assert np.array_equal(f.price(np.array([4, 3, 3, 2])), ETA ** np.array([4, 3, 3, 2]))


def test_fugacity_is_affine_and_uncapped():
    f = Fugacity(ETA, cap=8)
    assert f.hardWall is False
    for a, b in ((0, 1), (3, 5), (7, 11), (11, 7)):
        assert f.change(a, b) == pytest.approx((b - a) * np.log(ETA), abs=1e-13)


def test_fugacity_rejects_nonpositive():
    with pytest.raises(ValueError):
        Fugacity(0.0)
    with pytest.raises(ValueError):
        Fugacity(-1.0)


def test_at_most_one_intersection_price():
    S = supervillain.action.NoIntersections(Lattice(4, 4), kappa=0.2)
    with pytest.raises(ValueError):
        SurfaceWormGas(S, openSurfaceFugacity=0.2,
                       intersectionFugacity=ETA, chargeWeights=Fugacity(ETA))


def test_default_is_the_historical_fugacity():
    r"""Neither spelling given keeps $\eta_q = 0.3$, so no existing caller is affected by
    the introduction of the object."""
    _, g = _gas()
    assert isinstance(g.chargeWeights, Fugacity)
    assert g.intersectionFugacity == 0.3


def test_tuned_table_reports_no_fugacity():
    r"""Once the price stops being affine there is no $\eta_q$, and nothing may read one
    off it --- the alternative is a stale scalar quietly normalizing a measurement."""
    _, g = _gas(chargeWeights=WeightTable(np.array([0.0, -1.0, -1.5, -1.6]), -0.1))
    assert g.intersectionFugacity is None


def test_fugacity_and_equivalent_table_agree_on_the_extended_weight():
    r"""The two code paths must price identical configurations identically."""
    _, g = _gas(intersectionFugacity=ETA)
    _, h = _gas(chargeWeights=_equivalent_table(ETA))
    rng = np.random.default_rng(11)
    for _ in range(5):
        F = rng.integers(-1, 2, size=(6,) + (4,) * 4)
        assert g._log_extended_weight(F) == pytest.approx(h._log_extended_weight(F), abs=1e-9)


def _chain(steps=3, **kw):
    r"""Run one gas to completion and hand back its last configuration.

    Deliberately NOT interleaved with a second gas: ``seed=`` seeds *numba's global* RNG
    in ``_build_nb``, so two gases stepped alternately draw from one shared stream and
    diverge for reasons having nothing to do with what is under test.  Each gas gets the
    stream to itself, from its own seeding.
    """
    S, g = _gas(ticksPerStep=300, stride=50, **kw)
    cfg = S.configurations(1)[0]
    for _ in range(steps):
        cfg = g.step(cfg)
    return cfg


def test_fugacity_and_equivalent_table_generate_the_same_chain():
    r"""End to end, through the compiled kernel: same seed, same trajectory.  This is the
    gate --- the acceptance must not be able to tell a ``Fugacity`` from a table carrying
    its numbers."""
    a = _chain(intersectionFugacity=ETA)
    b = _chain(chargeWeights=_equivalent_table(ETA))
    assert np.array_equal(np.asarray(a['n']), np.asarray(b['n']))
    assert np.array_equal(np.asarray(a['phi']), np.asarray(b['phi']))


def test_a_nonaffine_table_actually_changes_the_acceptance():
    r"""Guards against inert plumbing: if the table never reached the weight, this would
    agree with the fugacity and every tuning run would be a no-op that looked converged.

    The state must actually *have* intersections, or both tables are read at $Q = 0$ and
    agree trivially.  A sparse random $F$ gives $Q = 0$ almost surely ($q = F\wedge F$
    pairs a component only with its complement, so scattered plaquettes never meet), and
    a dense random $F$ blows past the open-surface cap, making both weights $-\infty$ and
    their difference ``nan`` --- which compares False and would pass this test for
    entirely the wrong reason.  So: place complementary-component plaquettes together and
    keep only the finite states with $Q > 0$.

    Where the partner goes is not free: the cup product shifts its second factor, so
    $F_c(x)$ meets $F_{\bar c}(x + \hat e_a + \hat e_b)$ with $(a,b)$ the directions of
    $c$ --- verified by exhaustive scan over component pairs and offsets, which finds
    exactly the three complementary pairs at exactly that displacement.  Placing the
    partner at the *same* site (the obvious guess) gives $Q = 0$ every time.
    """
    S, g = _gas(intersectionFugacity=ETA)
    bowl = np.array([0.0, -3.0, 1.5, -2.0, 4.0] + [0.0] * 60)
    _, h = _gas(chargeWeights=WeightTable(bowl, -1.0, hardWall=False))
    components = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rng = np.random.default_rng(7)
    tested = differ = 0
    for _ in range(40):
        F = np.zeros((6,) + (4,) * 4, dtype=int)
        for _k in range(3):
            x = np.array(rng.integers(0, 4, size=4))
            c = int(rng.integers(0, 6))
            mu, nu = components[c]
            shift = np.zeros(4, dtype=int)
            shift[mu] = shift[nu] = 1
            F[(c,) + tuple(x)] = 1
            F[(5 - c,) + tuple((x + shift) % 4)] = 1
        a, b = g._log_extended_weight(F), h._log_extended_weight(F)
        if FState(S, F).Q == 0 or not (np.isfinite(a) and np.isfinite(b)):
            continue
        tested += 1
        differ += abs(a - b) > 1e-6
    assert tested > 0, 'fixture never produced a finite state with Q > 0'
    assert differ == tested, f'the table failed to reach the weight on {tested - differ} states'


def test_accumulator_normalizations_are_unchanged_for_a_fugacity():
    r"""``Theta_Theta`` divided by $V\eta_q^2$ and ``FourDefectDistribution`` by
    $\eta_q^{[4,3,3,2]}$ --- the two places a pricing change silently misnormalizes the
    physics, and where the four-defect bug lived."""
    old = CorrelatorAccumulator(4, intersectionFugacity=ETA)
    new = CorrelatorAccumulator(4, chargeWeights=Fugacity(ETA))
    for acc in (old, new):
        acc.pair = 137.0
        acc.pairWeightSquared = 19.0
        acc.fourDefect = np.array([2.0, 3.0, 5.0, 7.0])
    a, b = old.harvest(), new.harvest()
    V = 4 ** 4
    assert a['Theta_Theta'] == 137.0 / (V * ETA ** 2)
    assert b['Theta_Theta'] == a['Theta_Theta']
    assert np.array_equal(b['FourDefectDistribution'], a['FourDefectDistribution'])
    assert np.array_equal(a['FourDefectDistribution'],
                          np.array([2.0, 3.0, 5.0, 7.0]) / ETA ** np.array([4, 3, 3, 2]))


def test_pricing_round_trips_through_h5(tmp_path):
    import h5py
    f = Fugacity(ETA, cap=12)
    t = WeightTable(np.array([0.0, -1.0, -1.5]), -0.25, hardWall=True)
    with h5py.File(tmp_path / 'pricing.h5', 'w') as h:
        f.to_h5(h.create_group('fugacity'))
        t.to_h5(h.create_group('table'))
    with h5py.File(tmp_path / 'pricing.h5', 'r') as h:
        f2 = Fugacity.from_h5(h['fugacity'])
        t2 = WeightTable.from_h5(h['table'])
    assert f2.eta == f.eta and f2.cap == f.cap
    assert f2.price(3) == f.price(3)
    assert np.array_equal(t2.logWeight, t.logWeight) and t2.hardWall == t.hardWall
