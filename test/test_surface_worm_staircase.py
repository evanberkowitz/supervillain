"""Tests for surface worm staircase integer primitives."""

import numpy as np
import pytest


def test_primitive2_2d():
    """Verify primitive2 for random exact 2-forms on T^2.

    Ported from AUDIT/staircase.py _gate2 (lines 156-163).
    For D=2, generate random exact 2-forms and verify d(a)==b exactly.
    """
    from supervillain.generator.no_intersection.surface_worm.staircase import (
        d2,
        primitive2,
    )

    for N in (3, 4, 6, 8):
        rng = np.random.default_rng(N)
        c0 = rng.integers(-3, 4, (N, N))
        c1 = rng.integers(-3, 4, (N, N))
        b = d2(c0, c1)
        a0, a1 = primitive2(b)
        assert np.array_equal(d2(a0, a1), b), f'primitive2 FAIL N={N}'


def test_primitive_general_D():
    """Verify primitive_1form and primitive_2form for D=2,3,4.

    Ported from AUDIT/staircase.py _gate_generic (lines 166-182).
    Generate random exact forms and verify d(a)==b exactly.
    """
    from supervillain.generator.no_intersection.surface_worm.staircase import (
        d0,
        d1,
        primitive_1form,
        primitive_2form,
    )

    for D in (2, 3, 4):
        for N in (3, 4, 5):
            rng = np.random.default_rng(1000 * D + N)
            # exact 1-form m = dλ₀
            lam0 = rng.integers(-3, 4, (N,) * D)
            m = d0(lam0)
            lam = primitive_1form(m)
            assert np.array_equal(
                d0(lam), m
            ), f'primitive_1form FAIL D={D} N={N}'
            # exact 2-form b = da₀
            a0 = rng.integers(-3, 4, (D,) + (N,) * D)
            b = d1(a0)
            a = primitive_2form(b)
            assert np.array_equal(
                d1(a), b
            ), f'primitive_2form FAIL D={D} N={N}'


def test_primitive2_raises_on_flux():
    """Verify that primitive2 raises ValueError on non-zero total flux."""
    from supervillain.generator.no_intersection.surface_worm.staircase import (
        primitive2,
    )

    with pytest.raises(ValueError):
        primitive2(np.ones((4, 4), dtype=np.int64))


def test_primitive_2form_is_permissive_and_linear():
    """Verify that primitive_2form is permissive on non-exact input and linear.

    By design, primitive_2form does not raise on closed-non-exact 2-forms,
    accepting them as a linear map (for use in the sampler). The reconstruct
    function is the gate for exactness.
    """
    from supervillain.generator.no_intersection.surface_worm.staircase import (
        primitive_2form,
    )

    # Create a non-exact 2-form: F has non-zero period, but primitive_2form
    # does not raise. This is the permissive behavior by design.
    F = np.zeros((6, 4, 4, 4, 4), dtype=np.int64)
    F[0, 0, 0, :, :] = 1
    a = primitive_2form(F)  # returns without raising
    b = primitive_2form(2 * F)
    assert np.array_equal(b, 2 * a), "primitive_2form must be linear"
