#!/usr/bin/env python
# coding: utf-8

r"""
Spectral differential forms on a D-dimensional hypercubic lattice.

Uses the same compact data layout as :mod:`supervillain.lattice.compact`
(p-form stored as shape ``(C(D,p), N,...,N)``) but replaces finite-difference
operators with their spectral (Fourier-space) counterparts:

- Exterior derivative d: multiplication by $ik_\mu$ in Fourier space.
- Codifferential δ: multiplication by $-ik_\mu$ (formal adjoint of spectral d).
- Hodge star ★: pointwise algebraic sign, **no** spatial shift.
- Wedge product ∧: pointwise product, **no** shift on either factor.

The following continuum identities hold exactly at every finite $N$:

- $d^2 = 0$
- $\delta^2 = 0$
- $\langle da, b\rangle = \langle a, \delta b\rangle$ (exact adjointness)
- $(a \wedge b) = (-1)^{nm}(b \wedge a)$ (exact anti-commutativity)
- $\sum_x (a \wedge \star b)_\mathrm{top}[x] = \langle a, b \rangle$
- $\delta = (-1)^{D(k+1)+1}\,\star\,d\,\star$ with **no** translational shift

**Why the Leibniz rule fails — and why it cannot be fixed by a momentum
convention.**

The Leibniz rule $d(a \wedge b) = da \wedge b + (-1)^n a \wedge db$ does
**not** hold at finite $N$.

*Umklapp processes.*  In Fourier space the pointwise product of $a$ and $b$
becomes a circular convolution $\hat{a} \star \hat{b}$.  For each pair of
modes $(m,\, k{-}m)$ contributing to output mode $k$, the spectral
derivative multiplies by $iq_k$ on the left-hand side but by
$iq_m + iq_{k-m}$ on the right-hand side.  When the sum of the two input
momenta wraps around the Brillouin zone (Umklapp),

.. math::

   q_m + q_{k-m} = q_k \pm 2\pi,

so $i(q_m + q_{k-m}) \neq iq_k$, and Leibniz fails by $O(1)$.

*Why "mod Brillouin zone" does not help.*  One might hope to repair this by
replacing $q_m + q_{k-m}$ with $(q_m + q_{k-m}) \bmod 2\pi = q_k$.  The
problem is that this reduction must be applied *after* summing over modes and
*knowing* the output index $k$ — it cannot be distributed back to the
individual factors $da$ and $db$.  Carrying it out explicitly on the
right-hand side just reproduces $iq_k \,(\hat{a} \star \hat{b})[k] = d(a
\wedge b)[k]$, i.e.\ the left-hand side.  The "fix" is a tautology.

*Why finite-difference derivatives are immune.*  The interlaced and compact
lattice derivatives use eigenvalues of the form $e^{iq} - 1$ rather than
$iq$.  Because $e^{i(q+2\pi)} = e^{iq}$, Umklapp processes drop out
automatically:

.. math::

   e^{i(q_m + q_{k-m})} = e^{i(q_k \pm 2\pi)} = e^{iq_k}.

Any derivative eigenvalue of the form $f(e^{iq})$ inherits this
Brillouin-zone periodicity and satisfies Leibniz exactly on a finite lattice.
The spectral eigenvalue $iq$ is the unique continuum limit of these
expressions — and the only one that is not $2\pi$-periodic — which is
precisely why it fails.

The failure is $O(1)$ for generic random fields and vanishes as
$N\to\infty$ for smooth, band-limited forms.  Because this obstruction is
structural (not a convention choice), the Leibniz rule is simply not listed
among the properties of this module and no test for it is included.

*No-go: larger stencils cannot fix this.*  One might ask whether a longer-range
local derivative — second-nearest neighbours, or an arbitrary finite stencil
— could recover Leibniz while keeping exact anti-commutativity.  The answer is
no; the obstruction is algebraic, independent of stencil size.

Any local, translation-equivariant derivative has a Fourier-space eigenvalue
that is a trigonometric polynomial $\hat{g}(q) = \sum_n c_n e^{inq}$.
Leibniz imposes a functional equation on $\hat{g}$ that depends on the wedge:

- *Pointwise wedge* (anti-commutative, no shift): Leibniz requires
  $g(q{+}p) = g(q) + g(p)$ for all $q,p$.  Continuous $2\pi$-periodic
  solutions of this Cauchy equation satisfy $g(q) = cq$, but periodicity
  forces $c = 0$.  The only solution is $g \equiv 0$.

- *Shifted wedge* (one factor displaced by $s$ lattice steps): Leibniz
  requires $g(q{+}p) = g(q)e^{isp} + g(p)$.  Comparing $q \leftrightarrow p$
  forces $g(z) = C\bigl(w(z)-1\bigr)$ where $w : S^1 \to S^1$ is a group
  homomorphism, hence $w(e^{iq}) = e^{inq}$ for some integer $n$.  The only
  solutions are $g(q) = C(e^{inq}-1)$ — the $n$-step forward difference
  $f[x{+}n]-f[x]$, with the wedge shifting its second factor by $n$ steps.

Multi-point stencils such as $c_1(e^{iq}-1) + c_2(e^{2iq}-1)$ satisfy
neither equation for any fixed shift $s$, and therefore cannot satisfy
Leibniz for any choice of wedge.  The trade-off is exact:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Property
     - Compact / interlaced
     - This module
   * - $d^2 = 0$
     - ✓
     - ✓
   * - Adjointness $\langle da,b\rangle = \langle a,\delta b\rangle$
     - ✓
     - ✓
   * - Leibniz $d(a\wedge b) = da\wedge b + (-1)^n a\wedge db$
     - ✓
     - ✗ (Umklapp)
   * - Anti-commutativity $a\wedge b = (-1)^{nm}b\wedge a$
     - ✗ (shifted wedge)
     - ✓
   * - $\delta = (-1)^{\cdots}\star d\star$ (up to translation)
     - ✓
     - ✓ (no shift)

One column or the other — no local formulation achieves both.

The price for the other exact identities is that d (and δ) are non-local:
each application requires a pair of FFTs at cost $O(N^D \log N)$ instead of
the $O(N^D)$ stencil of :mod:`supervillain.lattice.compact`.

The :class:`~supervillain.lattice.compact.Lattice` and
:class:`~supervillain.lattice.compact.Form` classes are reused directly; this
module provides only the operator functions and re-exports
:func:`~supervillain.lattice.compact.push` and
:func:`~supervillain.lattice.compact.pull` unchanged.
"""

from itertools import combinations
import numpy as np

from supervillain.lattice.compact import Lattice, Form, _perm_sign, push, pull  # noqa: F401


# ---------------------------------------------------------------------------
# Momentum helper
# ---------------------------------------------------------------------------

def _momentum(N, D, direction):
    r"""
    Physical momenta $2\pi \times \texttt{fftfreq}(N)$ broadcast to a shape
    broadcastable with an $(N,)^D$ spatial array, with the non-trivial axis at
    position ``direction``.

    Parameters
    ----------
    N : int
        Lattice size per direction.
    D : int
        Number of spatial dimensions.
    direction : int
        Spatial direction (0-based) for this momentum component.

    Returns
    -------
    np.ndarray
        Shape ``(1,...,N,...,1)`` with ``N`` at axis ``direction``.
    """
    freqs = 2 * np.pi * np.fft.fftfreq(N)
    shape = [1] * D
    shape[direction] = N
    return freqs.reshape(shape)


# ---------------------------------------------------------------------------
# Exterior derivative  d : Ω^p → Ω^{p+1}
# ---------------------------------------------------------------------------

def d(f):
    r"""
    Spectral exterior derivative of a p-form, returning a (p+1)-form.

    For each output component $O = (o_0, \ldots, o_p)$:

    .. math::

        (df)_O[x] = \sum_{j=0}^{p} (-1)^j \, \mathcal{F}^{-1}\!\left[
            2\pi i\, \tilde{n}_{o_j} \cdot \hat{f}_{O \setminus \{o_j\}}
        \right][x]

    where $\hat{f}$ is the ortho-normalised DFT of $f$ along the spatial axes,
    $\tilde{n}_\mu = \texttt{fftfreq}(N)$ is the signed frequency index, and
    $\mathcal{F}^{-1}$ is the inverse DFT.  The factor $2\pi i\tilde{n}_\mu$
    is the exact eigenvalue of $\partial_\mu$ on the Fourier modes
    $e^{2\pi i\tilde{n}\cdot x/N}$.

    Unlike :func:`supervillain.lattice.d`, which uses a one-step forward
    finite difference, this operator has infinite range in position space and
    satisfies $d^2 = 0$ exactly.  The Leibniz rule fails at finite $N$; see
    the module docstring.

    Parameters
    ----------
    f : Form
        A p-form on a :class:`~supervillain.lattice.compact.Lattice`.

    Returns
    -------
    Form
        The (p+1)-form $df$, or the scalar ``0`` if $f$ is a D-form.
    """
    lat  = f.lattice
    p    = f.degree
    D    = lat.D
    if p == D:
        return 0

    result = lat.zeros(p + 1)
    axes   = tuple(range(D))

    for out_comp in lat.components[p + 1]:
        out_idx = lat.comp_index[p + 1][out_comp]

        for j, k_j in enumerate(out_comp):
            in_comp = tuple(k for k in out_comp if k != k_j)
            in_idx  = lat.comp_index[p][in_comp]
            sign    = (-1) ** j

            f_hat = np.fft.fftn(f[in_idx], axes=axes, norm='ortho')
            mom   = _momentum(lat.N, D, k_j)
            result[out_idx] += sign * np.fft.ifftn(
                1j * mom * f_hat, axes=axes, norm='ortho'
            ).real

    return result


# ---------------------------------------------------------------------------
# Codifferential  δ : Ω^p → Ω^{p-1}
# ---------------------------------------------------------------------------

def delta(f):
    r"""
    Spectral codifferential (formal adjoint of spectral :func:`d`), returning
    a (p-1)-form.

    For each output component $M = (m_0, \ldots, m_{p-1})$ and each
    direction $e \notin M$, let $j = \#\{m \in M : m < e\}$:

    .. math::

        (\delta f)_M[x] = -\sum_{e \notin M} (-1)^j \, \mathcal{F}^{-1}\!\left[
            2\pi i\, \tilde{n}_e \cdot \hat{f}_{M \cup \{e\}}
        \right][x]

    Derived from $\langle da, b\rangle = \langle a, \delta b\rangle$: the
    Fourier-space adjoint of the spectral derivative $ik_e$ is $-ik_e$
    (since $(ik_e)^\dagger = -ik_e$ under the real $L^2$ inner product).
    This replaces the backward finite difference of
    :func:`supervillain.lattice.delta` with the same spectral factor as in
    :func:`d`, so $\delta^2 = 0$ and the adjoint identity hold exactly.

    Parameters
    ----------
    f : Form
        A p-form on a :class:`~supervillain.lattice.compact.Lattice`.

    Returns
    -------
    Form
        The (p-1)-form $\delta f$, or the scalar ``0`` if $f$ is a 0-form.
    """
    lat      = f.lattice
    p        = f.degree
    D        = lat.D
    if p == 0:
        return 0

    result   = lat.zeros(p - 1)
    all_dirs = set(range(D))
    axes     = tuple(range(D))

    for out_comp in lat.components[p - 1]:
        out_idx = lat.comp_index[p - 1][out_comp]
        M_set   = set(out_comp)

        for e in sorted(all_dirs - M_set):
            j    = sum(1 for m in out_comp if m < e)
            sign = (-1) ** j

            in_comp = tuple(sorted(M_set | {e}))
            in_idx  = lat.comp_index[p][in_comp]

            f_hat = np.fft.fftn(f[in_idx], axes=axes, norm='ortho')
            mom   = _momentum(lat.N, D, e)
            result[out_idx] -= sign * np.fft.ifftn(
                1j * mom * f_hat, axes=axes, norm='ortho'
            ).real

    return result


δ = delta


# ---------------------------------------------------------------------------
# Hodge star  ★ : Ω^p → Ω^{D-p}
# ---------------------------------------------------------------------------

def star(f):
    r"""
    Pointwise Hodge star of a p-form, returning a (D-p)-form.

    For each output component $J$ (a sorted $(D-p)$-tuple), let $I$ be its
    complement in $\{0, \ldots, D-1\}$:

    .. math::

        (\star f)_J[x] = \sigma(I \frown J) \; f_I[x]

    Unlike :func:`supervillain.lattice.star`, there is **no** spatial shift
    $-\hat{e}_I$.  The shift in the standard compact star exists to align
    dual cells in the interlaced geometry; because the pointwise wedge here
    evaluates both factors at the same site, it is not needed and its absence
    is what makes $\delta = (-1)^{D(k+1)+1}\,\star\,d\,\star$ hold without a
    translational correction.

    Parameters
    ----------
    f : Form
        A p-form on a :class:`~supervillain.lattice.compact.Lattice`.

    Returns
    -------
    Form
        The (D-p)-form $\star f$.
    """
    lat    = f.lattice
    p      = f.degree
    D      = lat.D
    result = lat.zeros(D - p)

    for J_comp in lat.components[D - p]:
        J_set  = set(J_comp)
        I_comp = tuple(k for k in range(D) if k not in J_set)
        sign   = _perm_sign(I_comp + J_comp)
        result[lat.comp_index[D - p][J_comp]] = sign * f[lat.comp_index[p][I_comp]]

    return result


# ---------------------------------------------------------------------------
# Wedge product  ∧ : Ω^n × Ω^m → Ω^{n+m}
# ---------------------------------------------------------------------------

def wedge(a, b):
    r"""
    Pointwise wedge product of an n-form ``a`` and an m-form ``b``, returning
    an (n+m)-form.

    For each output component $O$, summing over all shuffles $O = A \sqcup B$
    ($A$ the $n$ a-directions, $B$ the $m$ b-directions):

    .. math::

        (a \wedge b)_O[x] = \sum_{O = A \sqcup B} \sigma(A \frown B) \;
            a_A[x] \; b_B[x]

    Both factors are evaluated at the **same site** $x$ — unlike
    :func:`supervillain.lattice.wedge`, which shifts $b$ to $x + \hat{e}_A$.
    This makes the product exactly anti-commutative at every site:

    .. math::

        a \wedge b = (-1)^{nm} \, b \wedge a

    Note that the Leibniz rule $d(a \wedge b) = da \wedge b + (-1)^n a \wedge
    db$ does **not** hold at finite $N$ — see the module docstring for the
    Umklapp analysis.

    Parameters
    ----------
    a : Form
        An n-form on a :class:`~supervillain.lattice.compact.Lattice`.
    b : Form
        An m-form on the same :class:`~supervillain.lattice.compact.Lattice`.

    Returns
    -------
    Form
        The (n+m)-form $a \wedge b$.

    Raises
    ------
    ValueError
        If $n + m > D$.
    """
    lat = a.lattice
    n, m = a.degree, b.degree
    if n + m > lat.D:
        raise ValueError(
            f"Cannot wedge a {n}-form and a {m}-form in D={lat.D}: {n}+{m} > {lat.D}"
        )

    result = lat.zeros(n + m)

    for out_comp in lat.components[n + m]:
        out_idx = lat.comp_index[n + m][out_comp]

        for A_dirs in combinations(out_comp, n):
            B_dirs = tuple(k for k in out_comp if k not in A_dirs)

            inversions = sum(1 for k in A_dirs for j in B_dirs if j < k)
            sign       = (-1) ** inversions

            a_idx = lat.comp_index[n][A_dirs]
            b_idx = lat.comp_index[m][B_dirs]

            result[out_idx] += sign * a[a_idx] * b[b_idx]

    return result
