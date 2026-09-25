'''
The reference Whitney mass matrix, on its own.

These tests do not involve the lattice at all -- only the geometry of one
fine simplex -- so a failure here localizes to the simplex, not the routing.
'''

from itertools import combinations
from math import factorial

import numpy as np
import pytest

import supervillain.lattice.barycentric.whitney as w


def test_gradient_gram_is_the_bordered_tridiagonal():
    '''tridiag(-4, 8, -4) with 4s in the corners -- pins the vertex convention.'''
    for D in (2, 3, 4):
        G = w.gradient_gram(D)
        assert G.shape == (D + 1, D + 1), D
        expected = np.zeros((D + 1, D + 1))
        for k in range(D + 1):
            expected[k, k] = 4 if k in (0, D) else 8
        for k in range(D):
            expected[k, k + 1] = expected[k + 1, k] = -4
        assert np.allclose(G, expected), D


def test_gradients_sum_to_zero():
    '''The barycentric coordinates sum to 1, so their gradients cancel.'''
    for D in (2, 3, 4):
        assert np.allclose(w.reference_gradients(D).sum(axis=0), 0), D


def test_monomial_integral_of_one_is_the_simplex_volume():
    for D in (2, 3, 4):
        vol = 1.0 / (factorial(D) * 2 ** D)
        assert np.isclose(w.monomial_integral(D, (0,) * (D + 1)), vol), D


def test_face_counts_are_binomial():
    from math import comb
    for D in (2, 3, 4):
        for p in range(D + 1):
            assert len(w.faces(D, p)) == comb(D + 1, p + 1), (D, p)


def test_local_mass_is_symmetric_positive_definite():
    for D in (2, 3, 4):
        for p in range(D + 1):
            M = w.local_mass(D, p)
            assert np.allclose(M, M.T), (D, p)
            evals = np.linalg.eigvalsh(M)
            assert evals.min() > 1e-12, (D, p, evals.min())


def test_local_mass_at_degree_zero_sums_to_the_simplex_volume():
    '''sum_ij M_0[i][j] = integral of (sum_i lambda_i)^2 = integral of 1 = |T|.'''
    for D in (2, 3, 4):
        vol = 1.0 / (factorial(D) * 2 ** D)
        assert np.isclose(w.local_mass(D, 0).sum(), vol), D


def test_local_mass_at_the_top_degree_is_the_reciprocal_volume():
    """The top Whitney form gives a 1x1 matrix equal to 1/|T| = D! 2^D:
    8, 48, 384 for D = 2, 3, 4.  Exact, and it pins the p! normalization."""
    for D in (2, 3, 4):
        M = w.local_mass(D, D)
        assert M.shape == (1, 1), D
        assert np.isclose(M[0, 0], factorial(D) * 2 ** D), (D, M[0, 0])


def test_local_mass_at_degree_zero_matches_the_closed_form():
    for D in (2, 3, 4):
        M = w.local_mass(D, 0)
        for i in range(D + 1):
            for j in range(D + 1):
                want = 2.0 ** -D * (1 + (i == j)) / factorial(D + 2)
                assert np.isclose(M[i, j], want), (D, i, j)


def test_local_mass_at_degree_one_matches_the_hand_expanded_whitney_form():
    '''Independent anchor at a middle degree, built by a route that shares no
    code with local_mass's general minor/determinant path -- so it catches a
    (-1)**(a+b) position-vs-label mixup that the p=0 and p=D tests cannot,
    since only the top face has every position equal to its label.

    For an edge (i, j), i < j, the Whitney 1-form is
    w_ij = lambda_i d(lambda_j) - lambda_j d(lambda_i), so

        <w_ij, w_kl> = l_i l_k G[j][l] - l_i l_l G[j][k]
                       - l_j l_k G[i][l] + l_j l_l G[i][k]

    with l_m l_n meaning integral of lambda_m * lambda_n over the simplex.
    Edges are enumerated directly here (not via faces) and only
    gradient_gram and monomial_integral -- each separately pinned by an
    exact closed-form test above -- are reused.
    '''
    for D in (2, 3, 4):
        edges = list(combinations(range(D + 1), 2))
        G = w.gradient_gram(D)

        def integral_of_lambda_product(m, n):
            exponents = [0] * (D + 1)
            exponents[m] += 1
            exponents[n] += 1
            return w.monomial_integral(D, tuple(exponents))

        F = len(edges)
        expected = np.zeros((F, F))
        for s, (i, j) in enumerate(edges):
            for t, (k, l) in enumerate(edges):
                expected[s, t] = (
                    integral_of_lambda_product(i, k) * G[j, l]
                    - integral_of_lambda_product(i, l) * G[j, k]
                    - integral_of_lambda_product(j, k) * G[i, l]
                    + integral_of_lambda_product(j, l) * G[i, k]
                )

        M = w.local_mass(D, 1)
        assert np.allclose(M, expected), D
