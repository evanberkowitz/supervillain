#!/usr/bin/env python

r"""
Reconstruct a full Villain configuration $(n, \phi)$ from an $F$-space state.

This is a *map*, not a generator.  The :class:`~.SurfaceWormGas` samples $F
\propto \pi(n)$ (the exact $\phi$-marginal, since $\int d\phi\, e^{-S} =
e^{-2\pi^2\kappa\|n_\perp\|^2} \times$ an $n$-independent Gaussian
determinant); this module turns each $F$-sample into an $(n, \phi)$ pair so
it can be fed to *all* the usual library observables:

1. $n \leftarrow$ integer primitive of $F$
   (:func:`~.staircase.primitive_2form`) plus winding.  $F = dn$ fixes $n$
   only up to a closed 1-form; the harmonic part (winding) is the extra
   datum $M = \sum_{\text{links}} n_\mu$ carried alongside $F$.  Because any
   two integer primitives of $F$ differ by $d\lambda + \text{const}$, $M$ is
   fixed *mod* $N^3$ by $F$ alone, so the winding shift $c = (M - M_0)/N^3$
   is an exact integer.
2. $\phi \leftarrow$ exact Gaussian conditional $\pi(\phi \mid n) =
   \mathcal N(2\pi\Delta^{-1}d^\dagger n, (\kappa\Delta)^{-1})$, one draw per
   config (zero mode uniform on the gauge circle).  Since $F \propto$ the
   marginal and $\phi\mid n$ is the exact conditional, the reconstructed
   $(n, \phi)$ is a valid joint sample --- every observable, $\phi$-dependent
   or not, is unbiased.

Gauge note: reconstructed $n$ differs from any particular source $n$ by a
gauge transform ($n \to n + dk$, $\phi \to \phi + 2\pi k$); all library
observables are gauge invariant, and $M$ itself is gauge invariant ($\sum dk$
telescopes to $0$), so the map is well defined on gauge classes.

.. note ::

    This is a straight port of the audited reference implementation
    (``reconstruct.py``'s ``reconstruct_n``/``draw_phi``, and
    ``surgery.py``'s ``fft_symbols``, in the no-intersections lab
    notebook's ``swg-audit-2026-07-31`` snapshot).  ``reconstruct`` and
    ``build_ensemble`` (thin wrappers around these two that build a single
    configuration / a whole :class:`~supervillain.ensemble.Ensemble`) were
    dropped: the gas emits configurations directly and nothing downstream
    consumes them.  See ``test_surface_worm_reconstruct.py`` for the gates.
"""

from itertools import combinations

import numpy as np

from supervillain.lattice import d

from .staircase import primitive_2form


def fft_symbols(N):
    r"""Fourier symbols of the lattice exterior derivative $d$ on a 0-form.

    Parameters
    ----------
    N: int
        Linear lattice extent (the same $N$ in all four directions).

    Returns
    -------
    omega: numpy.ndarray
        Complex array, shape ``(4,) + (N,)*4``: $\omega_\mu(k) = e^{ik_\mu}
        - 1$, the symbol of $(d\lambda)_\mu(x) = \lambda(x+\hat\mu) -
        \lambda(x)$.
    k2: numpy.ndarray
        Real array, shape ``(N,)*4``: $\hat k^2 = \sum_\mu |\omega_\mu|^2$,
        the symbol of the scalar Laplacian $\Delta = d^\dagger d$.
    k2safe: numpy.ndarray
        ``k2`` with the zero mode replaced by ``1.0``, safe to divide by
        (the caller masks the zero mode out separately).
    """
    k = 2 * np.pi * np.fft.fftfreq(N)
    K = np.meshgrid(k, k, k, k, indexing='ij')
    omega = np.array([np.exp(1j * Ki) - 1 for Ki in K])   # symbol of d
    k2 = (np.abs(omega) ** 2).sum(axis=0)
    k2safe = np.where(k2 == 0, 1.0, k2)
    return omega, k2, k2safe


def reconstruct_n(S, F, M):
    r"""Integer 1-form $n$ (array ``(4,N,N,N,N)``) with $d(n) = F$ and
    $\sum_\mu n_\mu = M$.

    The winding is restored with the harmonic *sheet* form $h^{(\mu)}_\mu =
    1$ on the $x_\mu = 0$ slice (holonomy 1, $\sum = N^3$, closed).  Two
    primitives of the same $F$ differ by a closed 1-form whose total is a
    multiple of $N^3$, so $(M - M_0)$ is an exact multiple of $N^3$ for any
    consistent $(F, M)$ pair.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action, used for its four-dimensional
        :class:`~supervillain.lattice.Lattice` (``S.Lattice``).
    F: numpy.ndarray
        Integer 2-form, shape ``(6,) + (N,)*4``, that must be exact.
    M: numpy.ndarray
        Target per-direction winding, shape ``(4,)`` (integer).

    Returns
    -------
    numpy.ndarray
        Integer 1-form $n$, shape ``(4,) + (N,)*4``, with $d(n) = F$ and
        $\sum_\text{links} n_\mu = M_\mu$.

    Raises
    ------
    ValueError
        If $F$ is not closed ($dF \neq 0$), if $F$ is closed but not exact
        (a nonzero $H^2$ period --- a nontrivial 2-cycle class), or if $M$
        is not consistent with $F$ (the winding mismatch $M - M_0$ is not
        $\equiv 0 \pmod{N^3}$).

        :func:`~.staircase.primitive_2form` requires exactness but does not
        check it, and on an illegal $F$ silently returns a changeling with
        $d(n) \neq F$ --- every downstream observable would then be
        measured on a configuration the chain never sampled.  (The sampler
        legitimately uses ``primitive_2form`` as a *linear map* on open $F$
        for its winding bookkeeping; it is the n-ification that must be
        gated, so the gate lives here.)
    """
    N = S.Lattice.N
    quantum = N ** 3
    F = np.asarray(F, dtype=np.int64)
    ff = S.Lattice.form(2)
    np.asarray(ff)[...] = F
    dF = np.asarray(d(ff))
    if np.any(dF != 0):
        raise ValueError(
            f'F is not closed ({int((dF != 0).sum())} cells with dF != 0): '
            'no n with d(n) == F exists')
    w = [np.asarray(F[c].sum(axis=(mu, nu)))
         for c, (mu, nu) in enumerate(combinations(range(4), 2))]
    if any(np.any(P != 0) for P in w):
        cls = tuple(int(P.flat[0]) for P in w)
        raise ValueError(
            f'F is closed but NOT exact: nontrivial H^2 class {cls} '
            '(nonzero 2-cycle periods); no n with d(n) == F exists')
    n0 = primitive_2form(F).astype(np.int64)          # d(n0) == F, some winding
    M0 = n0.reshape(4, -1).sum(axis=1)
    dM = np.asarray(M, dtype=np.int64) - M0
    if np.any(dM % quantum != 0):
        raise ValueError(
            f'winding mismatch not ≡0 mod N³={quantum}: (M−M0)={dM} — '
            'F and M are not a consistent (dn, winding) pair')
    c = dM // quantum
    n = n0.copy()
    for mu in range(4):                               # add c[mu] sheets on x_mu=0
        sl = [slice(None)] * 4
        sl[mu] = 0
        n[mu][tuple(sl)] += c[mu]
    return n


def draw_phi(S, n, rng):
    r"""Exact Gaussian conditional $\phi \sim \mathcal N(2\pi\Delta^{-1}
    d^\dagger n, (\kappa\Delta)^{-1})$; zero mode uniform on the gauge
    circle.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action; ``S.Lattice.N`` sets the lattice extent and ``S.kappa``
        the coupling.
    n: numpy.ndarray
        Integer 1-form, shape ``(4,) + (N,)*4``.
    rng: numpy.random.Generator
        Source of randomness for both the Gaussian fluctuation and the flat
        zero mode.

    Returns
    -------
    numpy.ndarray
        Real 0-form $\phi$, shape ``(N,)*4``.
    """
    N = S.Lattice.N
    kappa = float(S.kappa)
    omega, k2, k2safe = fft_symbols(N)
    nt = np.fft.fftn(np.asarray(n), axes=(1, 2, 3, 4))
    rhs = (omega.conj() * nt).sum(axis=0)             # (d^dagger n)~
    mean_t = np.where(k2 == 0, 0.0, 2 * np.pi * rhs / k2safe)
    xi = rng.normal(size=(N,) * 4)
    xit = np.fft.fftn(xi, axes=(0, 1, 2, 3))
    fluct_t = np.where(k2 == 0, 0.0, xit / np.sqrt(kappa * k2safe))
    phi = np.real(np.fft.ifftn(mean_t + fluct_t, axes=(0, 1, 2, 3)))
    phi += rng.uniform(0, 2 * np.pi)                  # flat zero mode (gauge circle)
    return phi
