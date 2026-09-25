r'''
Whitney forms on the reference fine simplex.

This module is pure simplex geometry: it knows about barycentric coordinates
and Gram matrices and nothing about parities, sites or routing.  It exists so
that ``supervillain.lattice.barycentric.compact`` can build its metric star
without growing a second vocabulary.

The reference fine $D$-simplex is the orthoscheme whose steps release the
directions $0, 1, \ldots, D-1$ in order.  In original lattice units its
vertices are $v_k = \frac{1}{2}\sum_{i<k}\hat e_i$, so consecutive edge
vectors are $\frac{1}{2}\hat e_{k-1}$ --- mutually orthogonal and each of
length $\frac{1}{2}$, giving $|T| = 1/(D!\,2^D)$.

Definitions follow :cite:`Arnold:2006zzb`.
'''

from functools import cache
from itertools import combinations
from math import factorial

import numpy as np

__all__ = ['reference_gradients', 'gradient_gram', 'monomial_integral',
           'faces', 'local_mass']


@cache
def reference_gradients(D):
    r'''
    The barycentric gradients $\nabla\lambda_k$ of the reference simplex.

    Parameters
    ----------
    D : int
        The dimension.

    Returns
    -------
    np.ndarray
        Shape ``(D+1, D)``; row $k$ is $\nabla\lambda_k$.
    '''
    g = np.zeros((D + 1, D))
    g[0, 0] = -2.0
    for i in range(1, D):
        g[i, i - 1] = 2.0
        g[i, i] = -2.0
    g[D, D - 1] = 2.0
    g.flags.writeable = False
    return g


@cache
def gradient_gram(D):
    r'''
    $G_{jk} = \nabla\lambda_j \cdot \nabla\lambda_k$, tridiagonal with
    diagonal $(4, 8, \ldots, 8, 4)$ and off-diagonal $-4$.

    Parameters
    ----------
    D : int
        The dimension.

    Returns
    -------
    np.ndarray
        Shape ``(D+1, D+1)``.
    '''
    g = reference_gradients(D)
    G = g @ g.T
    G.flags.writeable = False
    return G


def monomial_integral(D, exponents):
    r'''
    $\int_T \prod_k \lambda_k^{a_k}\,dV$ over the reference simplex.

    .. note ::
        $\int_T \prod_k \lambda_k^{a_k} = D!\,|T| \prod_k a_k! /
        (D + \sum_k a_k)!$, and $D!\,|T| = 2^{-D}$ here.

    Parameters
    ----------
    D : int
        The dimension.
    exponents : sequence of int
        One exponent $a_k$ per vertex, so ``D+1`` of them.

    Returns
    -------
    float
        The integral.
    '''
    total = sum(exponents)
    num = 1
    for a in exponents:
        num *= factorial(a)
    return 2.0 ** -D * num / factorial(D + total)


@cache
def faces(D, p):
    r'''
    The $p$-faces of the reference simplex, as sorted vertex tuples.

    Parameters
    ----------
    D : int
        The dimension.
    p : int
        The face degree.

    Returns
    -------
    tuple
        $\binom{D+1}{p+1}$ tuples, each of $p+1$ vertex indices.
    '''
    return tuple(combinations(range(D + 1), p + 1))


@cache
def local_mass(D, p):
    r'''
    The Whitney mass matrix of the reference simplex at degree $p$.

    Entry $[\sigma, \tau]$ is $\int_T \langle w_\sigma, w_\tau \rangle$, where
    $w_\sigma = p! \sum_a (-1)^a \lambda_{i_a} d\lambda_{i_0} \wedge \cdots
    \widehat{d\lambda_{i_a}} \cdots \wedge d\lambda_{i_p}$ and the pointwise
    inner product of two $p$-covectors is the Gram determinant
    $\det G[I][J]$.

    .. note ::
        At $p = 0$ this is $\int_T \lambda_i \lambda_j$, whose total is
        $\int_T 1 = |T|$ because the barycentric coordinates sum to one.  That
        is the cheapest check that the normalization is right.

    Parameters
    ----------
    D : int
        The dimension.
    p : int
        The degree.

    Returns
    -------
    np.ndarray
        Shape ``(F, F)`` with ``F = len(faces(D, p))``, symmetric positive
        definite.
    '''
    G = gradient_gram(D)
    F = faces(D, p)
    fac = factorial(p) ** 2
    M = np.zeros((len(F), len(F)))
    for s, I in enumerate(F):
        for t, J in enumerate(F):
            total = 0.0
            for a, ia in enumerate(I):
                Ia = [k for k in I if k != ia]
                for b, jb in enumerate(J):
                    Jb = [k for k in J if k != jb]
                    minor = G[np.ix_(Ia, Jb)]
                    det = 1.0 if p == 0 else float(np.linalg.det(minor))
                    e = [0] * (D + 1)
                    e[ia] += 1
                    e[jb] += 1
                    total += (-1) ** (a + b) * monomial_integral(D, e) * det
            M[s, t] = fac * total
    M.flags.writeable = False
    return M
