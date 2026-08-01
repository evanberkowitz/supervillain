"""Tests for surface worm F -> (n, phi) reconstruction.

Ported (in miniature, at reduced statistics) from AUDIT/reconstruct.py's
gate functions in the no-intersections lab notebook's
swg-audit-2026-07-31 snapshot: gate_roundtrip (exactness raises,
d(n)==F/winding/q==0 roundtrip) and the action-density check embedded in
gate_gauge (used here as the exact-conditional-phi sanity check, since
that is what test_phi_conditional_action targets).
"""

import numpy as np
import pytest

import supervillain
from supervillain.lattice import Lattice, d, wedge
from supervillain.generator.no_intersection.surface_worm.reconstruct import (
    reconstruct_n,
    draw_phi,
)
from supervillain.generator.no_intersection.surface_worm.staircase import d0


def _S(N=4, kappa=0.2):
    return supervillain.action.NoIntersections(Lattice(4, N), kappa=kappa)


def test_raises_on_class():
    """A closed but non-exact F (nonzero H^2 period, the 'class-1 sheet')
    must raise -- no n has d(n) == F."""
    S = _S()
    N = S.Lattice.N
    F = np.zeros((6,) + (N,) * 4, dtype=np.int64)
    F[0, 0, 0, :, :] = 1
    with pytest.raises(ValueError):
        reconstruct_n(S, F, np.zeros(4, dtype=np.int64))


def test_raises_on_open():
    """A single open plaquette (dF != 0) must raise -- F is not even closed."""
    S = _S()
    N = S.Lattice.N
    F = np.zeros((6,) + (N,) * 4, dtype=np.int64)
    F[0, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError):
        reconstruct_n(S, F, np.zeros(4, dtype=np.int64))


def test_roundtrip():
    """d(reconstruct_n(S, F, M)) == F, winding kept, q == 0, on 10 valid
    (F, M) samples built from a deterministic constraint-safe n (only a
    spatial component, no x0-dependence, so q = dn^dn = 0 identically)
    plus a random exact shift n + d0(z) for variety.  The reconstructed n
    need not equal the source n -- only its gauge class (F, M) does."""
    S = _S()
    N = S.Lattice.N
    rng = np.random.default_rng(1)
    n_base = np.zeros((4,) + (N,) * 4, dtype=np.int64)
    n_base[1][:, :, 0, :] = 1
    for _ in range(10):
        z = rng.integers(-2, 3, size=(N,) * 4).astype(np.int64)
        n_src = n_base + d0(z)
        nf = S.Lattice.form(1)
        np.asarray(nf)[...] = n_src
        F = np.asarray(d(nf)).astype(np.int64)
        M = n_src.reshape(4, -1).sum(axis=1)

        n = reconstruct_n(S, F, M)

        nrf = S.Lattice.form(1)
        np.asarray(nrf)[...] = n
        assert np.array_equal(np.asarray(d(nrf)).astype(np.int64), F)
        assert np.array_equal(n.reshape(4, -1).sum(axis=1), M)
        ff = S.Lattice.form(2)
        np.asarray(ff)[...] = F
        assert not np.any(np.asarray(wedge(ff, ff)) != 0)


def test_winding_quantum_mismatch_raises():
    """M off the true winding by a non-multiple of N**3 is an inconsistent
    (F, M) pair and must raise."""
    S = _S()
    N = S.Lattice.N
    n_base = np.zeros((4,) + (N,) * 4, dtype=np.int64)
    n_base[1][:, :, 0, :] = 1
    nf = S.Lattice.form(1)
    np.asarray(nf)[...] = n_base
    F = np.asarray(d(nf)).astype(np.int64)
    M0 = n_base.reshape(4, -1).sum(axis=1)
    M_bad = M0.copy()
    M_bad[0] += 1  # not a multiple of N**3
    with pytest.raises(ValueError):
        reconstruct_n(S, F, M_bad)


def test_phi_conditional_action():
    """draw_phi's exact Gaussian conditional phi ~ N(2*pi*Delta^-1 d^+ n,
    (kappa*Delta)^-1) has the equipartition-scale mean action density.

    With n == 0 the mean (a fixed longitudinal projection) vanishes, so
    the action density S/V = (kappa/2) * sum((dphi)**2) / V is pure
    fluctuation: a Gaussian free field with covariance (kappa*Delta)^-1
    on the (N**4 - 1) non-zero Fourier modes (the zero mode carries no
    fluctuation), giving <S/V> = (N**4 - 1) / (2 * N**4) by equipartition
    -- independent of kappa, since kappa cancels between the action's
    prefactor and the fluctuation's variance.  Ported (in miniature) from
    the action_density check embedded in AUDIT/reconstruct.py's
    gate_gauge.
    """
    S = _S(N=4, kappa=0.2)
    N = S.Lattice.N
    rng = np.random.default_rng(3)
    n = np.zeros((4,) + (N,) * 4, dtype=np.int64)

    def action_density(phi):
        dphi = np.stack(
            [np.roll(phi, -1, axis=mu) - phi for mu in range(4)]
        )
        return 0.5 * S.kappa * (dphi ** 2).sum() / (N ** 4)

    draws = 50
    vals = np.array([action_density(draw_phi(S, n, rng)) for _ in range(draws)])
    target = (N ** 4 - 1) / (2 * N ** 4)
    sem = vals.std() / np.sqrt(draws)
    assert abs(vals.mean() - target) < 5 * sem
