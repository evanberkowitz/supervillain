#!/usr/bin/env python
r"""
The extended-ensemble state $F$ for the :class:`~.SurfaceWormGas`.

The surface worm samples on the enlarged manifold of *all* integer 2-forms $F$
--- not just the closed, exact ones ($dF = 0$, $[F] = 0$) that correspond to a
genuine defect-free $n$ --- so that it can tunnel between constraint-surface
sectors by passing through open, non-topological intermediate states.
:class:`FState` is the bookkeeping this needs: $F$ itself plus every quantity
derived from it that the acceptance ratio or a measurement wants, split into

* incrementally-maintained fields (:attr:`F`, :attr:`dF`, :attr:`q`,
  :attr:`G`, :attr:`counts`, :attr:`winding`, :attr:`periods`) that a
  compiled batch of local moves updates in $O(1)$ per move without ever
  recomputing a global sum, and
* python-only charge bookkeeping (:attr:`absoluteCharge`,
  :attr:`squaredCharge`, :attr:`chargeSites`) that cannot cross the ``njit``
  boundary (a python ``dict`` is not a numba type) and so is rebuilt in
  $O(V)$ by :meth:`refresh_charge` after a batch, rather than touched by
  every move.

.. note ::

    All of this is a straight port of the audited reference implementation
    (``gas.py``'s ``_init`` and ``_refresh_charge_state`` in the
    no-intersections lab notebook's ``swg-audit-2026-07-31`` snapshot),
    re-expressed as a standalone state object rather than methods glued to
    the sampler.  See ``test_surface_worm_state.py`` for the gates.

The state is deliberately *not* self-mutating: nothing in this module
performs a worm move.  It only constructs $F$'s derived quantities from
scratch (:meth:`__init__`, :meth:`from_configuration`), offers an
independent from-scratch recompute to check incremental maintenance against
(:meth:`recomputed`, :meth:`check`), and rebuilds the charge bookkeeping a
compiled kernel cannot carry (:meth:`refresh_charge`).  The compiled mover
this state is meant to sit next to (a future task) is expected to mutate
:attr:`F`, :attr:`dF`, :attr:`q`, :attr:`G`, :attr:`counts`, :attr:`winding`,
and :attr:`periods` in place.
"""

import numpy as np

from supervillain.lattice import d, delta, wedge

from .kernel import pot
from .staircase import primitive_2form


class FState:
    r"""
    The surface worm's extended-ensemble state: an integer 2-form $F$ on a
    four-dimensional :class:`~supervillain.lattice.Lattice`, plus everything
    derived from it.

    Parameters
    ----------
    S: supervillain.action.NoIntersections
        The action, used for its four-dimensional
        :class:`~supervillain.lattice.Lattice` (``S.Lattice``) --- nothing
        else about the action (not even $\kappa$) is read here; the
        acceptance weight built from $\kappa$ lives in the sampler, not the
        state.
    F: numpy.ndarray or None
        The integer 2-form, shape ``(6,) + (N,)*4`` (six plaquette
        components, ordered $(01,02,03,12,13,23)$, on a $D=4$ lattice).
        ``None`` gives the cold start $F \equiv 0$ --- the trivial vacuum,
        already :attr:`legal_vacuum`.  Otherwise cast to ``int64`` and
        copied (the state owns its array; the caller's is untouched).

    Attributes
    ----------
    F: numpy.ndarray
        The 2-form itself, ``int64``, shape ``(6,) + (N,)*4``.
    dF: numpy.ndarray
        The closure defect $dF$, a 3-form, ``int64``.  Zero exactly where
        $F$ is closed.
    q: numpy.ndarray
        The self-intersection density $F \wedge F$, a top-form (one integer
        per site), ``int64``, shape ``(N,)*4``.
    G: numpy.ndarray
        $\Delta^{-1}F$, component-wise, shape ``(6,) + (N,)*4``, real.  The
        coexact norm $C(F) = \sum_c F_c \cdot G_c$ that prices $F$ in the
        extended-ensemble weight, and (via :meth:`intersection_winding`) a
        real primitive of $F$ for the direct-$J$ construction.
    counts: dict
        ``{'D': int, 'Q': int}`` --- the open-surface count
        $D = \#\{dF \neq 0\}$ and the intersection count
        $Q = \#\{q \neq 0\}$.  Kept as a dict (rather than two plain
        attributes) so a compiled batch can hand back both numbers through
        one mutable container without the caller needing to know their
        names in advance; :attr:`D` and :attr:`Q` are read-only views onto
        it.
    winding: numpy.ndarray
        The winding 4-vector $M_0(F)$, ``int64``, shape ``(4,)`` --- the
        component sums of a staircase primitive of $F$
        (:func:`~.staircase.primitive_2form`).  Linear in $F$; this is the
        from-scratch route, kept here as the incrementally-maintained
        reference a compiled kernel's sensitivity-based update is checked
        against.
    periods: numpy.ndarray
        The six component totals $\Sigma_x F_c(x)$, ``int64``, shape
        ``(6,)``.  Defined on *every* state (unlike the $H^2$ class, which
        needs $dF = 0$), and once $D = 0$ these test $[F] = 0$ exactly ---
        the zero-period condition a legal vacuum needs on top of $D = Q = 0$
        (see :attr:`legal_vacuum`): a worm that recloses around a
        non-contractible 2-cycle emits a "vacuum" no integer $n$ can
        represent, and this is what catches it.
    absoluteCharge: int
        $\sum_x \lvert q(x) \rvert$.
    squaredCharge: int
        $\sum_x q(x)^2$.
    chargeSites: dict
        ``{site: charge}`` for every site with $q \neq 0$, ``site`` a tuple
        of four ints.
    """

    def __init__(self, S, F=None):
        self.S = S
        self.N = int(S.Lattice.N)
        expected_shape = (6,) + (self.N,) * 4
        if F is None:
            F = np.zeros(expected_shape, dtype=np.int64)
        else:
            F = np.array(F, dtype=np.int64, copy=True)
            if F.shape != expected_shape:
                raise ValueError(f'FState needs F of shape {expected_shape}, got {F.shape}')
        self.F = F
        self._rebuild_all()

    def _rebuild_all(self):
        r"""Populate every derived attribute from :attr:`F` alone, $O(V\log V)$
        (the Green's function FFTs) --- the constructor's and
        :meth:`from_configuration`'s shared implementation."""
        N = self.N
        F = self.F
        f = self.S.Lattice.form(2)
        np.asarray(f)[...] = F
        self.dF = np.asarray(d(f)).astype(np.int64).copy()
        self.q = np.asarray(wedge(f, f)).astype(np.int64).reshape((N,) * 4).copy()
        self.G = np.array([pot(F[c], N) for c in range(6)])
        self.counts = {
            'D': int((self.dF != 0).sum()),
            'Q': int((self.q != 0).sum()),
        }
        self.winding = primitive_2form(F).reshape(4, -1).sum(axis=1).astype(np.int64)
        self.periods = np.array([int(F[c].sum()) for c in range(6)], dtype=np.int64)
        self.refresh_charge()

    @classmethod
    def from_configuration(cls, S, configuration):
        r"""Build the state from a Villain configuration's integer 1-form $n$,
        $F = dn$.

        Parameters
        ----------
        S: supervillain.action.NoIntersections
            The action, as for :meth:`__init__`.
        configuration: dict
            A configuration as produced by ``S.configurations(...)`` ---
            only its ``'n'`` entry is read.

        Returns
        -------
        FState
        """
        L = S.Lattice
        n = np.asarray(configuration['n'], dtype=np.int64)
        nf = L.form(1, dtype=np.int64)
        np.asarray(nf)[...] = n
        F = np.asarray(d(nf)).astype(np.int64)
        return cls(S, F)

    @property
    def D(self):
        r"""The open-surface count $D = \#\{dF \neq 0\}$, read from :attr:`counts`."""
        return self.counts['D']

    @property
    def Q(self):
        r"""The intersection count $Q = \#\{q \neq 0\}$, read from :attr:`counts`."""
        return self.counts['Q']

    @property
    def legal_vacuum(self):
        r"""Whether $F$ represents a genuine defect-free configuration: closed
        ($D = 0$), self-intersection-free ($Q = 0$), and with every period
        zero ($[F] = 0$) so some integer $n$ with $dn = F$ actually exists.

        $D = Q = 0$ alone is not enough --- $F$ could be closed but wrap a
        non-contractible 2-cycle (nonzero :attr:`periods`), a state no
        emitted $n$ can represent (see :attr:`periods`)."""
        return self.D == 0 and self.Q == 0 and not self.periods.any()

    def refresh_charge(self):
        r"""Rebuild :attr:`absoluteCharge`, :attr:`squaredCharge`, and
        :attr:`chargeSites` from :attr:`q`, $O(V)$ plus one pass over the
        occupied cells.

        A compiled batch of moves maintains :attr:`F`, :attr:`dF`, :attr:`q`,
        :attr:`G`, :attr:`counts`, and :attr:`winding` incrementally but
        cannot touch this python-only bookkeeping (a ``dict`` does not cross
        the ``njit`` boundary), so it is refreshed here instead --- once per
        batch rather than once per move, which is what makes strided
        measurement affordable.

        Returns
        -------
        FState
            ``self``, for chaining.
        """
        q = self.q
        self.absoluteCharge = int(np.abs(q).sum())
        self.squaredCharge = int((q ** 2).sum())
        self.chargeSites = {tuple(int(v) for v in h): int(q[tuple(h)])
                             for h in np.argwhere(q != 0)}
        return self

    def recomputed(self):
        r"""Every incrementally-maintained quantity, recomputed from :attr:`F`
        alone rather than trusted, $O(V\log V)$.

        For gates: the independent computation :meth:`check` compares
        incremental state against.

        Returns
        -------
        dict
            ``{'dF', 'q', 'D', 'Q', 'winding', 'periods', 'C'}`` --- the
            closure defect, self-intersection density, open-surface and
            intersection counts, winding 4-vector, component-total periods,
            and coexact norm $C(F) = \sum_c F_c \cdot (\Delta^{-1}F)_c$, all
            freshly computed.
        """
        N = self.N
        F = self.F
        f = self.S.Lattice.form(2)
        np.asarray(f)[...] = F
        dF = np.asarray(d(f)).astype(np.int64)
        q = np.asarray(wedge(f, f)).astype(np.int64).reshape((N,) * 4)
        C = sum(float((F[c] * pot(F[c], N)).sum()) for c in range(6))
        winding = primitive_2form(F).reshape(4, -1).sum(axis=1).astype(np.int64)
        periods = np.array([int(F[c].sum()) for c in range(6)], dtype=np.int64)
        return {
            'dF': dF,
            'q': q,
            'D': int((dF != 0).sum()),
            'Q': int((q != 0).sum()),
            'winding': winding,
            'periods': periods,
            'C': C,
        }

    def check(self, atol=1e-9):
        r"""Verify the incrementally-maintained state against a from-scratch
        :meth:`recomputed` build.

        Parameters
        ----------
        atol: float
            Absolute tolerance for the one real-valued comparison, the
            coexact norm $C(F)$ built from :attr:`G` versus its fresh FFT
            recompute.

        Returns
        -------
        bool
            ``True`` if everything matches.

        Raises
        ------
        AssertionError
            On the first mismatch found, naming which quantity disagreed.
        """
        r = self.recomputed()
        assert np.array_equal(self.dF, r['dF']), 'FState.check: dF disagrees with recompute'
        assert np.array_equal(self.q, r['q']), 'FState.check: q disagrees with recompute'
        assert self.D == r['D'], f"FState.check: D={self.D} != recomputed D={r['D']}"
        assert self.Q == r['Q'], f"FState.check: Q={self.Q} != recomputed Q={r['Q']}"
        assert np.array_equal(self.winding, r['winding']), \
            'FState.check: winding disagrees with recompute'
        assert np.array_equal(self.periods, r['periods']), \
            'FState.check: periods disagrees with recompute'
        C = sum(float((self.F[c] * self.G[c]).sum()) for c in range(6))
        assert abs(C - r['C']) <= atol, \
            f"FState.check: C={C} != recomputed C={r['C']} (atol={atol})"
        return True

    def intersection_winding(self, spread_tol=1e-8):
        r"""$J_\mu$, the :class:`~supervillain.observable.IntersectionWinding`,
        computed directly from $F$ with no integer primitive and no $n$.

        Since gauge and harmonic winding drop out of the periods of
        $n \wedge dn$ (see :class:`~supervillain.observable.IntersectionWinding`),
        *any* primitive of $F$ gives the same $J$ --- including the
        real-valued Green's-function one, $a = \delta\Delta^{-1}F$ (the
        lattice Biot--Savart / helicity construction):

        .. math ::

            d(\delta\Delta^{-1}F) = (\Delta - \delta d)\Delta^{-1}F
            = F - \delta\Delta^{-1}(dF) = F,

        using $dF = 0$ and the vanishing zero mode of $F$ (it has no
        harmonic/constant part).  So

        .. math ::

            J_\mu = \sum_{x:\,x_\mu = c} (a \wedge F)_{\bar\mu}(x)

        must equal the library :class:`~supervillain.observable.IntersectionWinding`
        computed from an emitted integer $n$ --- as an exact integer reached
        by floats.  This reuses :attr:`G`, which already *is*
        $\Delta^{-1}F$ per component, instead of a fresh FFT.

        Only valid when $F$ is closed with vanishing periods (i.e.
        :attr:`legal_vacuum`, or at least $dF = 0$ and $[F] = 0$) --- $j$ is
        closed only on that surface, which is exactly what makes the slice
        sum independent of the slice position $c$; :func:`spread_tol` is the
        gate that catches a call off that surface rather than silently
        returning a slice-dependent, non-topological number.

        Parameters
        ----------
        spread_tol: float
            Tolerance for two independent gates: how much the slice sum may
            vary across the $N$ choices of $c$, and how far the (real, before
            rounding) result may sit from an integer.  Either exceeded raises
            ``ValueError`` rather than silently rounding a meaningless float.

        Returns
        -------
        numpy.ndarray
            $J$, ``int64``, shape ``(4,)``.

        Raises
        ------
        ValueError
            If the slice sum is not independent of the slice, or the result
            is not close to integers, to tolerance ``spread_tol`` --- both
            symptoms of $F$ not being closed with vanishing periods.
        """
        L = self.S.Lattice
        N = self.N
        g = L.form(2)
        np.asarray(g)[...] = self.G
        a = delta(g)
        f = L.form(2)
        np.asarray(f)[...] = self.F
        j = wedge(a, f)
        ja = np.asarray(j)
        J = np.zeros(4)
        for mu in range(4):
            comp = L.comp_index[3][tuple(k for k in range(4) if k != mu)]
            slices = np.array([ja[comp].take(c, axis=mu).sum() for c in range(N)])
            spread = float(slices.max() - slices.min())
            if spread > spread_tol:
                raise ValueError(
                    f'intersection_winding: slice sum varies by {spread:.3e} across '
                    f'direction {mu} (> spread_tol={spread_tol}); F is not closed with '
                    'vanishing periods, so J is not slice-independent -- check legal_vacuum.')
            J[mu] = slices[0]
        distance = float(np.abs(J - np.round(J)).max())
        if distance > spread_tol:
            raise ValueError(
                f'intersection_winding: result {J} sits {distance:.3e} from the nearest '
                f'integer (> spread_tol={spread_tol}); F is not closed with vanishing '
                'periods, so J is not topological -- check legal_vacuum.')
        return np.round(J).astype(np.int64)
