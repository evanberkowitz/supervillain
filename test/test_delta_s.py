#!/usr/bin/env python
r"""
Tests that each generator's fast ΔS formula exactly matches ΔS_obvious = S(new) - S(old).

Each test makes the minimal field change that preserves any relevant constraint,
then checks that the direct (per-link) ΔS formula agrees with the brute-force
action difference to floating-point precision.
"""

import numpy as np
import pytest
import supervillain
from supervillain.lattice import d, delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def villain_cfg(L, seed=0):
    rng = np.random.default_rng(seed)
    phi = L.form(0)
    phi[0] = rng.uniform(-np.pi, np.pi, L.dims)
    n = L.form(1, dtype=int)
    for mu in range(L.D):
        n[mu] = rng.integers(-2, 3, L.dims)
    return phi, n


def worldline_cfg(L, seed=0):
    rng = np.random.default_rng(seed)
    # Build m as δt so δm = δ²t = 0 is guaranteed.
    t = L.form(2, dtype=int)
    for c in range(len(L.components[2])):
        t[c] = rng.integers(-2, 3, L.dims)
    m = delta(t)
    v = L.form(2, dtype=int)
    for c in range(len(L.components[2])):
        v[c] = rng.integers(-2, 3, L.dims)
    return m, v


GEOMETRIES = [(2, 4), (3, 3)]


# ---------------------------------------------------------------------------
# Villain action: φ change at a single site
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('D,N', GEOMETRIES)
def test_villain_phi_single_site(D, N):
    r"""
    ΔS_direct = (κ/2) Σ_l Δr_l (2r_l + Δr_l) for Δφ at one site
    equals S(φ+Δφ, n) − S(φ, n).
    """
    L = supervillain.lattice.Lattice(D=D, N=N)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    phi, n = villain_cfg(L)

    site = tuple([1] * D)
    change_phi = L.zeros(0)
    change_phi[0][site] = 0.3

    r = d(phi) - 2 * np.pi * n
    change_r = d(change_phi)                    # n unchanged
    dS_direct  = float((S.kappa / 2) * (change_r * (2 * r + change_r)).sum())
    dS_obvious = float(S(phi + change_phi, n) - S(phi, n))

    assert abs(dS_direct - dS_obvious) < 1e-10


# ---------------------------------------------------------------------------
# Villain action: n change on a plaquette boundary (maintains dn = 0 mod W)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('D,N', GEOMETRIES)
def test_villain_n_plaquette_boundary(D, N):
    r"""
    ΔS_direct for a plaquette boundary change to n (dn preserved)
    equals S(φ, n+Δn) − S(φ, n).
    """
    L = supervillain.lattice.Lattice(D=D, N=N)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    phi, n = villain_cfg(L)

    # Change n on the boundary of the (0,1) plaquette at site (1,1,...,1).
    # Boundary: +n[0,x], −n[1,x], +n[1,x+ê₀], −n[0,x+ê₁]
    mu, nu = 0, 1
    x = np.array([1] * D)
    e_mu = np.zeros(D, dtype=int); e_mu[mu] = 1
    e_nu = np.zeros(D, dtype=int); e_nu[nu] = 1
    x_t  = tuple(x)
    xmu  = tuple(L.mod(x + e_mu))
    xnu  = tuple(L.mod(x + e_nu))

    change_n = L.zeros(1, dtype=int)
    change_n[mu][x_t]  += +1
    change_n[nu][x_t]  += -1
    change_n[nu][xmu]  += +1
    change_n[mu][xnu]  += -1

    r = d(phi) - 2 * np.pi * n
    change_r   = -2 * np.pi * change_n         # phi unchanged
    dS_direct  = float((S.kappa / 2) * (change_r * (2 * r + change_r)).sum())
    dS_obvious = float(S(phi, n + change_n) - S(phi, n))

    assert abs(dS_direct - dS_obvious) < 1e-10


# ---------------------------------------------------------------------------
# Villain action: combined φ + n change (NeighborhoodUpdate style)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('D,N', GEOMETRIES)
def test_villain_neighborhood_proposal(D, N):
    r"""
    ΔS_direct via d(Δφ) − 2π Δn residual formula
    equals S(φ+Δφ, n+Δn) − S(φ, n) for a neighborhood-style proposal.
    """
    L = supervillain.lattice.Lattice(D=D, N=N)
    S = supervillain.action.Villain(L, kappa=0.7, W=1)
    phi, n = villain_cfg(L)

    # Propose a change at site x: Δφ at x, and Δn on all 2D adjacent links.
    x    = np.array([1] * D)
    x_t  = tuple(x)
    rng  = np.random.default_rng(99)

    change_phi = L.zeros(0)
    change_phi[0][x_t] = rng.uniform(-np.pi, np.pi)

    change_n = L.zeros(1, dtype=int)
    for mu in range(D):
        e_mu = np.zeros(D, dtype=int); e_mu[mu] = 1
        bwd  = tuple(L.mod(x - e_mu))
        change_n[mu][x_t] = rng.integers(-1, 2)
        change_n[mu][bwd] = rng.integers(-1, 2)

    r          = d(phi) - 2 * np.pi * n
    change_r   = d(change_phi) - 2 * np.pi * change_n
    dS_direct  = float((S.kappa / 2) * (change_r * (2 * r + change_r)).sum())
    dS_obvious = float(S(phi + change_phi, n + change_n) - S(phi, n))

    assert abs(dS_direct - dS_obvious) < 1e-10


# ---------------------------------------------------------------------------
# Worldline action: v change at a single plaquette (VortexUpdate style)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('D,N', GEOMETRIES)
def test_worldline_v_single_plaquette(D, N):
    r"""
    ΔS_direct = (1/2κ) Σ_l Δf_l (2f_l + Δf_l) for Δv at one plaquette
    equals S(m, v+Δv) − S(m, v), where f = m − δv/W.
    """
    L   = supervillain.lattice.Lattice(D=D, N=N)
    S   = supervillain.action.Worldline(L, kappa=0.7, W=1)
    m, v = worldline_cfg(L)

    mu, nu    = 0, 1
    comp_idx  = L.comp_index[2][(mu, nu)]
    site      = tuple([1] * D)

    change_v              = L.zeros(2, dtype=int)
    change_v[comp_idx][site] = 1

    f             = m - delta(v) / S._W
    change_f      = -delta(change_v) / S._W   # m unchanged
    dS_direct     = float((1 / (2 * S.kappa)) * (change_f * (2 * f + change_f)).sum())
    dS_obvious    = float(S(m, v + change_v) - S(m, v))

    assert abs(dS_direct - dS_obvious) < 1e-10


# ---------------------------------------------------------------------------
# Worldline action: coexact m change δt at a single plaquette (CoexactUpdate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('D,N', GEOMETRIES)
def test_worldline_m_coexact(D, N):
    r"""
    ΔS_direct for m → m + δt (t supported on one plaquette)
    equals S(m+δt, v) − S(m, v), preserving δm = 0.
    """
    L   = supervillain.lattice.Lattice(D=D, N=N)
    S   = supervillain.action.Worldline(L, kappa=0.7, W=1)
    m, v = worldline_cfg(L)

    mu, nu   = 0, 1
    comp_idx = L.comp_index[2][(mu, nu)]
    site     = tuple([1] * D)

    t               = L.zeros(2, dtype=int)
    t[comp_idx][site] = 1
    change_m        = delta(t)

    f             = m - delta(v) / S._W
    change_f      = change_m                   # v unchanged, Δf = Δm
    dS_direct     = float((1 / (2 * S.kappa)) * (change_f * (2 * f + change_f)).sum())
    dS_obvious    = float(S(m + change_m, v) - S(m, v))

    assert abs(dS_direct - dS_obvious) < 1e-10


# ---------------------------------------------------------------------------
# Worldline action: combined m + v change (PlaquetteUpdate style)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('D,N', GEOMETRIES)
def test_worldline_plaquette_proposal(D, N):
    r"""
    ΔS_direct for simultaneous m boundary change and v change at one plaquette
    equals S(m+Δm, v+Δv) − S(m, v).
    """
    L   = supervillain.lattice.Lattice(D=D, N=N)
    S   = supervillain.action.Worldline(L, kappa=0.7, W=1)
    m, v = worldline_cfg(L)

    mu, nu   = 0, 1
    comp_idx = L.comp_index[2][(mu, nu)]
    x        = np.array([1] * D)
    e_mu     = np.zeros(D, dtype=int); e_mu[mu] = 1
    e_nu     = np.zeros(D, dtype=int); e_nu[nu] = 1
    x_t      = tuple(x)
    xmu      = tuple(L.mod(x + e_mu))
    xnu      = tuple(L.mod(x + e_nu))

    change_m_val = 1
    change_v_val = -1
    delta_f_val  = change_m_val - change_v_val / S._W

    change_m = L.zeros(1, dtype=int)
    change_m[mu][x_t]  += +change_m_val
    change_m[nu][xmu]  += +change_m_val
    change_m[mu][xnu]  += -change_m_val
    change_m[nu][x_t]  += -change_m_val

    change_v              = L.zeros(2, dtype=int)
    change_v[comp_idx][x_t] = change_v_val

    f         = m - delta(v) / S._W
    change_f  = change_m - delta(change_v) / S._W
    dS_direct  = float((1 / (2 * S.kappa)) * (change_f * (2 * f + change_f)).sum())
    dS_obvious = float(S(m + change_m, v + change_v) - S(m, v))

    assert abs(dS_direct - dS_obvious) < 1e-10
