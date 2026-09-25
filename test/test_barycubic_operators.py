#!/usr/bin/env python
r'''
Exterior-calculus operators, coarsening, dual forms, I/O and dispatch on the barycubic lattice
(SPEC §7 items 7, 8, 13, 14, 15, and the dispatch decision of SPEC §11 Q2).

* **Coboundary** (item 7): ``d∘d = 0`` exactly on integers, ``d`` nonzero below top degree, linear,
  and its incidence table (SPEC §9) is rank-ordered vertex deletion with face offsets in ``{0,1}^D``.
* **Metric operators** (item 8): ``δ = M^{-1} dᵀ M`` is the adjoint of ``d`` in the Hermitian inner
  product ``Σ M conj(a) b``; ``δ∘δ = 0``; the Laplacian ``dδ + δd`` is self-adjoint and positive
  semidefinite.  These are the ports of the hypercubic tests in ``test/test_lattice.py``.
* **Bridge to the hypercubic lattice** (item 13): ``coarsen = Sdᵀ`` is a cochain map into
  ``compact.Form`` and preserves the integral of top forms.
* **I/O** (item 14): HDF5 round trips of the lattice, of standalone forms, and of forms inside a
  :class:`~supervillain.batch.Batch`, for every degree and for int/float/complex dtypes, keeping the
  class, degree, dual flag and lattice.
* **Dual forms** (item 15): ``star`` returns a :class:`BarycubicDualForm` of degree ``D − p``;
  ``inverse_star`` undoes it; the dual form has no ``d``.
* **Dispatch** (SPEC §11): ``supervillain.lattice.{d, delta, laplacian, star, wedge, cup}`` hand
  barycubic forms to ``barycubic.py`` and hypercubic ``Form``s to ``compact.py`` unchanged;
  hypercubic ``cup`` is the wedge at ``i = 0`` and raises ``NotImplementedError`` for ``i > 0``.
'''


import h5py as h5
import numpy as np
import pytest

import supervillain.lattice as lattice_pkg
import supervillain.h5.strategy  # noqa: F401  (registers every HDF5 strategy)
from supervillain.batch import Batch, resolve_batch_cls
from supervillain.h5 import Data
from supervillain.lattice import Lattice, Form, compact
from supervillain.lattice import barycubic as bc

import barycubic_reference as ref
from barycubic_reference import DN_ALL, lattice, int_form, float_form, is_zero


def _laplacian_via_d_delta(f):
    r'''$\Delta = d\delta + \delta d$ straight from ``d`` and ``δ``, dropping the scalar 0 at the ends of the complex.'''
    L = f.lattice
    out = np.zeros(f.shape, dtype=complex if np.iscomplexobj(f) else float)
    if f.degree > 0:
        out = out + np.asarray(bc.d(bc.delta(f)))
    if f.degree < L.D:
        out = out + np.asarray(bc.delta(bc.d(f)))
    return out


def _inner_or_zero(a, b):
    r'''``inner`` that treats the scalar 0 returned at the ends of the complex as the zero form.'''
    if np.ndim(a) == 0 or np.ndim(b) == 0:
        return 0.0
    return bc.inner(a, b)


# =============================================================================================
# Item 7: the coboundary
# =============================================================================================

@pytest.mark.parametrize("D,N", DN_ALL + [(5, 2)])
def test_d_nilpotent(D, N):
    r'''$d \circ d = 0$ exactly on integer cochains of every degree (including the D = 5, N = 2 smoke case).'''
    # SPEC §4, §7 item 7.
    L = lattice(D, N)
    for p in range(D - 1):
        assert is_zero(bc.d(bc.d(int_form(L, p, 70))))


@pytest.mark.parametrize("D,N", DN_ALL)
def test_d_nonzero_typed_and_integer(D, N):
    r'''Below top degree, $d$ of a random integer form is a nonzero integer :class:`BarycubicForm` of degree p+1.'''
    # SPEC §2.3 "Dtypes", §7 item 7.
    L = lattice(D, N)
    for p in range(D):
        df = bc.d(int_form(L, p, 71))
        assert isinstance(df, bc.BarycubicForm)
        assert df.degree == p + 1 and df.lattice is L
        assert df.shape == (ref.simplices_per_site(D, p + 1),) + (N,) * D
        assert np.issubdtype(df.dtype, np.integer)
        assert np.asarray(df).any()


@pytest.mark.parametrize("D,N", DN_ALL)
def test_d_top_degree_is_scalar_zero(D, N):
    r'''$d$ of a D-form is the scalar 0, as ``compact.d`` returns; $\delta$ of a 0-form is the scalar 0.'''
    # SPEC §4, §5.2.
    L = lattice(D, N)
    top = bc.d(int_form(L, D, 72))
    assert np.ndim(top) == 0 and top == 0
    bottom = bc.delta(float_form(L, 0, 72))
    assert np.ndim(bottom) == 0 and bottom == 0


@pytest.mark.parametrize("D,N", DN_ALL)
def test_d_linear_and_complex(D, N):
    r'''$d(a + 3b) = da + 3\,db$ exactly, and on complex forms $d(a + ib) = da + i\,db$.'''
    # SPEC §2.3, §7 item 7.
    L = lattice(D, N)
    for p in range(D):
        a, b = int_form(L, p, 73), int_form(L, p, 74)
        assert np.array_equal(np.asarray(bc.d(a + 3 * b)), np.asarray(bc.d(a) + 3 * bc.d(b)))
        z = bc.d(bc.BarycubicForm(np.asarray(a) + 1j * np.asarray(b), degree=p, lattice=L))
        assert np.iscomplexobj(z)
        assert np.array_equal(np.asarray(z), np.asarray(bc.d(a)) + 1j * np.asarray(bc.d(b)))


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_d_table_is_rank_ordered_deletion(D):
    r'''``operator_table('d', p)`` rows are exactly the faces $(v_0…\hat v_j…v_{p+1})$ with sign $(-1)^j$, anchored at $x + o$, $o \in \{0,1\}^D$.'''
    # SPEC §2.4, §4, §9.
    L = lattice(D, 2)
    for p in range(D):
        table = np.asarray(L.operator_table('d', p))
        assert table.dtype == np.int64 and table.shape[1] == D + 3
        assert set(np.unique(table[:, 2:2 + D])) <= {0, 1}
        got = {}
        for row in table:
            key = (int(row[0]), int(row[1]), tuple(int(t) for t in row[2:2 + D]))
            got[key] = got.get(key, 0) + int(row[-1])
        got = {k: v for k, v in got.items() if v}
        assert got == ref.expected_d_table(L.local_vertices, p)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_d_table_reproduces_d(D, N):
    r'''Applying the ``'d'`` table, $(d\omega)(o, x) = \sum {\rm sign}\,\omega(i, x + {\rm offset})$, reproduces ``d`` exactly.'''
    # SPEC §9 "Incidence application".
    L = lattice(D, N)
    axes = tuple(range(D))
    for p in range(D):
        f = int_form(L, p, 75)
        out = np.zeros((ref.simplices_per_site(D, p + 1),) + (N,) * D, dtype=np.int64)
        for row in np.asarray(L.operator_table('d', p)):
            o, i, off, s = row[0], row[1], row[2:2 + D], row[-1]
            out[o] += s * np.roll(np.asarray(f)[i], tuple(-off), axis=axes)
        assert np.array_equal(out, np.asarray(bc.d(f)))


@pytest.mark.parametrize("D,N", DN_ALL)
def test_dT_table_is_transpose(D, N):
    r'''``operator_table('dT', p+1)`` is the transpose of ``operator_table('d', p)`` (offsets negated, in $\{0,-1\}^D$): $\sum (d f)\, g = \sum f\, (d^\top g)$.'''
    # SPEC §9.  Interpretation: 'dT' at input degree p+1 maps (p+1)-cochains to p-cochains.
    L = lattice(D, N)
    axes = tuple(range(D))
    for p in range(D):
        dT = np.asarray(L.operator_table('dT', p + 1))
        assert set(np.unique(dT[:, 2:2 + D])) <= {0, -1}
        f, g = int_form(L, p, 76), int_form(L, p + 1, 77)
        out = np.zeros(f.shape, dtype=np.int64)
        for row in dT:
            o, i, off, s = row[0], row[1], row[2:2 + D], row[-1]
            out[o] += s * np.roll(np.asarray(g)[i], tuple(-off), axis=axes)
        assert (np.asarray(bc.d(f)) * np.asarray(g)).sum() == (np.asarray(f) * out).sum()


# =============================================================================================
# Item 8: inner product, δ, Laplacian
# =============================================================================================

@pytest.mark.parametrize("D,N", DN_ALL)
def test_inner_is_weighted_hermitian(D, N):
    r'''$\langle a, b\rangle = \sum_\sigma M_p\,\overline{a_\sigma} b_\sigma$; Hermitian, and positive on nonzero complex forms.'''
    # SPEC §5.2.
    L = lattice(D, N)
    for p in range(D + 1):
        a, b = float_form(L, p, 80, dtype=complex), float_form(L, p, 81, dtype=complex)
        M = np.asarray(L.hodge_weights(p)).reshape((-1,) + (1,) * D)
        assert bc.inner(a, b) == pytest.approx((M * np.conj(np.asarray(a)) * np.asarray(b)).sum(), rel=1e-12)
        assert bc.inner(a, b) == pytest.approx(np.conj(bc.inner(b, a)), rel=1e-12)
        aa = bc.inner(a, a)
        assert abs(np.imag(aa)) < 1e-12 * abs(aa) and np.real(aa) > 0


@pytest.mark.parametrize("D,N", DN_ALL)
def test_delta_is_adjoint_of_d(D, N):
    r'''$\langle da, b\rangle = \langle a, \delta b\rangle$ for real and complex forms at every degree.'''
    # SPEC §5.2, §7 item 8.
    L = lattice(D, N)
    for p in range(D):
        for dtype in (float, complex):
            a, b = float_form(L, p, 82, dtype=dtype), float_form(L, p + 1, 83, dtype=dtype)
            assert bc.inner(bc.d(a), b) == pytest.approx(bc.inner(a, bc.delta(b)), rel=1e-10)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_delta_nilpotent_and_typed(D, N):
    r'''$\delta \circ \delta = 0$; $\delta$ of a p-form is a floating (p−1)-form, even on integer input.'''
    # SPEC §2.3 "Dtypes", §7 item 8.
    L = lattice(D, N)
    for p in range(1, D + 1):
        g = bc.delta(int_form(L, p, 84))
        assert isinstance(g, bc.BarycubicForm) and g.degree == p - 1
        assert np.issubdtype(g.dtype, np.inexact)
        if p >= 2:
            f = float_form(L, p, 85)
            assert np.allclose(np.asarray(bc.delta(bc.delta(f))), 0, atol=1e-10)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_laplacian_is_d_delta_plus_delta_d(D, N):
    r'''The Laplacian equals $d\delta + \delta d$ and keeps the degree.'''
    # SPEC §5.2, §7 item 8 (as test_lattice.py::test_laplacian_matches_d_delta).
    L = lattice(D, N)
    for p in range(D + 1):
        f = float_form(L, p, 86)
        lf = bc.laplacian(f)
        assert isinstance(lf, bc.BarycubicForm) and lf.degree == p
        assert np.allclose(np.asarray(lf), _laplacian_via_d_delta(f), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_laplacian_self_adjoint_and_positive(D, N):
    r'''$\langle\Delta a, b\rangle = \langle a, \Delta b\rangle$ and $\langle\Delta f, f\rangle = \|df\|^2 + \|\delta f\|^2 \ge 0$.'''
    # SPEC §7 item 8.
    L = lattice(D, N)
    for p in range(D + 1):
        a, b = float_form(L, p, 87), float_form(L, p, 88)
        assert bc.inner(bc.laplacian(a), b) == pytest.approx(bc.inner(a, bc.laplacian(b)), rel=1e-10)
        energy = _inner_or_zero(bc.d(a), bc.d(a)) + _inner_or_zero(bc.delta(a), bc.delta(a))
        assert bc.inner(bc.laplacian(a), a) == pytest.approx(energy, rel=1e-10)
        assert bc.inner(bc.laplacian(a), a) >= 0


# =============================================================================================
# Item 13: coarsening to the hypercubic lattice
# =============================================================================================

@pytest.mark.parametrize("D,N", DN_ALL)
def test_coarsen_is_cochain_map(D, N):
    r'''$Sd^\top d_S = d_K Sd^\top$ exactly on integer cochains, with ``compact.d`` as $d_K$; the result is an integer hypercubic ``Form``.'''
    # SPEC §6.1, §7 item 13.
    L = lattice(D, N)
    for p in range(D):
        omega = int_form(L, p, 90)
        coarse = bc.coarsen(omega)
        assert isinstance(coarse, Form) and coarse.degree == p
        assert coarse.lattice.D == D and coarse.lattice.N == N
        assert np.issubdtype(coarse.dtype, np.integer)
        assert np.array_equal(np.asarray(bc.coarsen(bc.d(omega))), np.asarray(compact.d(coarse)))


@pytest.mark.parametrize("D,N", DN_ALL)
def test_coarsen_preserves_top_integral(D, N):
    r'''$\int_M Sd^\top\omega = \int_M \omega$ for top forms (the hypercubic integral is the plain sum).'''
    # SPEC §3, §6.1, §7 item 13.
    L = lattice(D, N)
    omega = int_form(L, D, 91)
    assert np.asarray(bc.coarsen(omega)).sum() == bc.integrate(omega)


# =============================================================================================
# Item 15: dual forms
# =============================================================================================

@pytest.mark.parametrize("D,N", DN_ALL)
def test_star_is_typed_dual_form(D, N):
    r'''``star(a)`` is a :class:`BarycubicDualForm` with degree $D-p$, ``primal_degree`` p, ``dual`` True, primal shape, and values $M_p a$.'''
    # SPEC §5.3, §7 item 15.
    L = lattice(D, N)
    for p in range(D + 1):
        a = float_form(L, p, 92)
        s = bc.star(a)
        assert isinstance(s, bc.BarycubicDualForm)
        assert s.degree == D - p and s.primal_degree == p and s.dual is True
        assert s.lattice is L and s.shape == a.shape
        M = np.asarray(L.hodge_weights(p)).reshape((-1,) + (1,) * D)
        assert np.allclose(np.asarray(s), M * np.asarray(a), rtol=1e-15, atol=0)
        assert not getattr(a, 'dual', False)


@pytest.mark.parametrize("D,N", DN_ALL)
def test_inverse_star_undoes_star(D, N):
    r'''``inverse_star(star(a)) = a`` and returns a primal :class:`BarycubicForm` of degree p.'''
    # SPEC §7 item 15.
    L = lattice(D, N)
    for p in range(D + 1):
        a = float_form(L, p, 93)
        back = bc.inverse_star(bc.star(a))
        assert type(back) is bc.BarycubicForm and back.degree == p
        assert np.allclose(np.asarray(back), np.asarray(a), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("D,N", [(2, 3), (4, 2)])
def test_dual_arithmetic_and_pairing(D, N):
    r'''Dual forms add as dual forms ($\star a + \star b = \star(a+b)$), and $\sum a\cdot\star b = \langle a, b\rangle$ for real forms.'''
    # SPEC §5.3: the kinetic term is inner(ξ, ξ) = Σ ξ·star(ξ).
    L = lattice(D, N)
    for p in range(D + 1):
        a, b = float_form(L, p, 94), float_form(L, p, 95)
        total = bc.star(a) + bc.star(b)
        assert isinstance(total, bc.BarycubicDualForm) and total.degree == D - p
        assert np.allclose(np.asarray(total), np.asarray(bc.star(a + b)), rtol=1e-14)
        assert (np.asarray(a) * np.asarray(bc.star(b))).sum() == pytest.approx(bc.inner(a, b), rel=1e-12)


@pytest.mark.parametrize("D,N", [(2, 3), (3, 2)])
def test_unsupported_dual_operations_raise(D, N):
    r'''There is no dual formulation: $d$, $\delta$, the Laplacian and cup products of a dual form raise.'''
    # SPEC §5.3, §7 item 15.  Interpretation: any of TypeError / ValueError / NotImplementedError.
    L = lattice(D, N)
    s = bc.star(float_form(L, 1, 96))
    for op in (bc.d, bc.delta, bc.laplacian, lattice_pkg.d):
        with pytest.raises((TypeError, ValueError, NotImplementedError)):
            op(s)
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        bc.cup(s, float_form(L, 0, 97))


# =============================================================================================
# Integer forms stay integer (SPEC §2.3 "Dtypes")
# =============================================================================================

@pytest.mark.parametrize("D,N", [(2, 3), (3, 2), (4, 2)])
def test_integer_forms_stay_integer(D, N):
    r'''$d$, $\cup_i$, ``coarsen``, ``transform`` and ``half_shift`` keep integer dtypes.'''
    L = lattice(D, N)
    a, b = int_form(L, 1, 98), int_form(L, 1, 99)
    for x in (bc.d(a), bc.cup(a, b), bc.cup(a, b, 1), bc.coarsen(a),
              bc.transform(a, tuple(range(D)), (1,) * D, (1,) + (0,) * (D - 1)), bc.half_shift(a)):
        assert np.issubdtype(np.asarray(x).dtype, np.integer)


# =============================================================================================
# Item 14: HDF5 round trips
# =============================================================================================

IO_DN = [(1, 2), (2, 3), (3, 2), (4, 2)]
DTYPES = [np.int64, np.float64, np.complex128]


def _typed_form(L, p, dtype, seed):
    if np.issubdtype(np.dtype(dtype), np.integer):
        return int_form(L, p, seed, dtype=dtype)
    return float_form(L, p, seed, dtype=dtype)


def _assert_same_form(got, want):
    assert type(got) is type(want)
    assert got.degree == want.degree
    assert bool(getattr(got, 'dual', False)) == bool(getattr(want, 'dual', False))
    assert got.lattice.D == want.lattice.D and got.lattice.N == want.lattice.N
    assert np.asarray(got).dtype == np.asarray(want).dtype
    assert np.array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("D,N", IO_DN)
def test_lattice_h5_roundtrip(D, N, tmp_path):
    r'''The lattice survives ``to_h5``/``from_h5`` and ``Data.write``/``read`` with the same D, N and labels.'''
    # SPEC §7 item 14.
    L = lattice(D, N)
    with h5.File(tmp_path / 'lattice.h5', 'w') as f:
        L.to_h5(f.create_group('lattice'))
        Data.write(f, 'data_lattice', L)
    with h5.File(tmp_path / 'lattice.h5', 'r') as f:
        for back in (bc.BarycubicLattice.from_h5(f['lattice']), Data.read(f['data_lattice'])):
            assert isinstance(back, bc.BarycubicLattice)
            assert back.D == D and back.N == N
            for p in range(D + 1):
                assert list(back.labels[p]) == list(L.labels[p])


@pytest.mark.parametrize("D,N", IO_DN)
@pytest.mark.parametrize("dtype", DTYPES)
def test_form_h5_roundtrip_standalone(D, N, dtype, tmp_path):
    r'''A standalone :class:`BarycubicForm` of every degree reads back with its class, degree, lattice and dtype (not as a plain ndarray).'''
    # SPEC §2.3 "Serialization work in scope" (2), §7 item 14.
    L = lattice(D, N)
    forms = {p: _typed_form(L, p, dtype, 100 + p) for p in range(D + 1)}
    with h5.File(tmp_path / 'form.h5', 'w') as f:
        for p, form in forms.items():
            Data.write(f, f'form{p}', form)
    with h5.File(tmp_path / 'form.h5', 'r') as f:
        for p, form in forms.items():
            _assert_same_form(Data.read(f[f'form{p}']), form)


@pytest.mark.parametrize("D,N", IO_DN)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_dual_form_h5_roundtrip_standalone(D, N, dtype, tmp_path):
    r'''A standalone :class:`BarycubicDualForm` reads back as a dual form with its degree, ``primal_degree`` and dual flag.'''
    # SPEC §2.3, §5.3, §7 items 14-15.
    L = lattice(D, N)
    forms = {p: bc.star(float_form(L, p, 110 + p, dtype=dtype)) for p in range(D + 1)}
    with h5.File(tmp_path / 'dual.h5', 'w') as f:
        for p, form in forms.items():
            Data.write(f, f'dual{p}', form)
    with h5.File(tmp_path / 'dual.h5', 'r') as f:
        for p, form in forms.items():
            back = Data.read(f[f'dual{p}'])
            _assert_same_form(back, form)
            assert back.primal_degree == p


def test_batch_tags_registered():
    r'''``resolve_batch_cls`` knows both barycubic tags (and still ``'Form'``), and rejects an unknown tag.'''
    # SPEC §2.3 "Serialization work in scope" (1), §7 item 14 "mismatched type metadata rejected".
    assert resolve_batch_cls('BarycubicForm') is bc.BarycubicForm
    assert resolve_batch_cls('BarycubicDualForm') is bc.BarycubicDualForm
    assert resolve_batch_cls('Form') is Form
    assert bc.BarycubicForm.__batch_tag__ != Form.__batch_tag__
    with pytest.raises(ValueError):
        resolve_batch_cls('NotAForm')


@pytest.mark.parametrize("D,N", IO_DN)
@pytest.mark.parametrize("dtype", DTYPES)
def test_form_batch_h5_roundtrip(D, N, dtype, tmp_path):
    r'''A :class:`Batch` of barycubic forms (every degree) reads back with its element class, and each draw keeps class, degree, lattice, dtype and values.'''
    # SPEC §7 item 14.
    L = lattice(D, N)
    columns = {}
    for p in range(D + 1):
        batch = Batch(3, cls=bc.BarycubicForm, dtype=dtype, degree=p, lattice=L)
        for draw in range(3):
            batch[draw] = _typed_form(L, p, dtype, 120 + 10 * p + draw)
        columns[p] = batch
    with h5.File(tmp_path / 'batch.h5', 'w') as f:
        for p, batch in columns.items():
            Data.write(f, f'column{p}', batch)
    with h5.File(tmp_path / 'batch.h5', 'r') as f:
        for p, batch in columns.items():
            back = Data.read(f[f'column{p}'])
            assert isinstance(back, Batch) and back.cls is bc.BarycubicForm
            assert back.shape == batch.shape and back.dtype == batch.dtype
            for draw in range(3):
                _assert_same_form(back[draw], batch[draw])


@pytest.mark.parametrize("D,N", [(2, 3), (4, 2)])
def test_dual_form_batch_h5_roundtrip(D, N, tmp_path):
    r'''A :class:`Batch` of dual forms reads back as :class:`BarycubicDualForm` draws with the dual degree.'''
    # SPEC §7 items 14-15.  The Batch item kwarg ``degree`` is the form's own ``degree`` attribute.
    L = lattice(D, N)
    p = 1
    draws = [bc.star(float_form(L, p, 130 + k)) for k in range(2)]
    batch = Batch(2, cls=bc.BarycubicDualForm, degree=draws[0].degree, lattice=L)
    for k, s in enumerate(draws):
        batch[k] = s
    with h5.File(tmp_path / 'dual-batch.h5', 'w') as f:
        Data.write(f, 'column', batch)
    with h5.File(tmp_path / 'dual-batch.h5', 'r') as f:
        back = Data.read(f['column'])
    assert back.cls is bc.BarycubicDualForm
    for k, s in enumerate(draws):
        _assert_same_form(back[k], s)


def test_hypercubic_and_barycubic_columns_never_confused(tmp_path):
    r'''Hypercubic and barycubic columns in one file read back as their own classes; a column with an unknown tag is rejected.'''
    # SPEC §2.3 ("a separate batch/h5 tag"), §7 item 14.
    H, L = Lattice(2, 3), lattice(2, 3)
    hyper = Batch(2, cls=Form, degree=1, lattice=H)
    bary = Batch(2, cls=bc.BarycubicForm, degree=1, lattice=L)
    with h5.File(tmp_path / 'mixed.h5', 'w') as f:
        Data.write(f, 'hyper', hyper)
        Data.write(f, 'bary', bary)
        Data.write(f, 'tampered', bary)
        f['tampered'].attrs['H5Batch_cls'] = 'NotAForm'
    with h5.File(tmp_path / 'mixed.h5', 'r') as f:
        assert Data.read(f['hyper']).cls is Form
        assert Data.read(f['bary']).cls is bc.BarycubicForm
        assert type(Data.read(f['bary'])[0]) is bc.BarycubicForm
        with pytest.raises(ValueError):
            Data.read(f['tampered'])


# =============================================================================================
# SPEC §11 Q2: dispatch in supervillain.lattice
# =============================================================================================

@pytest.mark.parametrize("name", ['d', 'delta', 'laplacian', 'star'])
def test_dispatch_routes_barycubic_forms(name):
    r'''``supervillain.lattice.<op>`` on a barycubic form returns exactly what ``barycubic.<op>`` returns, with the same type.'''
    # SPEC §11 Q2.
    L = lattice(2, 3)
    for p in range(3):
        f = float_form(L, p, 140)
        got, want = getattr(lattice_pkg, name)(f), getattr(bc, name)(f)
        assert type(got) is type(want)
        assert np.array_equal(np.asarray(got), np.asarray(want))


def test_dispatch_routes_barycubic_products():
    r'''``supervillain.lattice.wedge`` is $\cup_0$ and ``supervillain.lattice.cup(a, b, i)`` is ``barycubic.cup`` on barycubic forms.'''
    # SPEC §11 Q2.
    L = lattice(3, 2)
    a, b = int_form(L, 2, 141), int_form(L, 1, 142)
    assert np.array_equal(np.asarray(lattice_pkg.wedge(a, b)), np.asarray(bc.cup(a, b, 0)))
    assert np.array_equal(np.asarray(lattice_pkg.cup(a, b)), np.asarray(bc.cup(a, b, 0)))
    assert np.array_equal(np.asarray(lattice_pkg.cup(a, b, 1)), np.asarray(bc.cup(a, b, 1)))
    assert type(lattice_pkg.cup(a, b, 1)) is bc.BarycubicForm


@pytest.mark.parametrize("D,N", [(2, 3), (3, 4)])
def test_dispatch_leaves_hypercubic_forms_unchanged(D, N):
    r'''On hypercubic ``Form``s the dispatchers equal ``compact.d, delta, laplacian, star, wedge``; ``cup(a, b, 0)`` is the wedge.'''
    # SPEC §11 Q2.  These pass before the barycubic implementation exists.
    H = Lattice(D, N)
    for p in range(D + 1):
        f = H.random(p)
        for name in ('d', 'delta', 'laplacian', 'star'):
            got, want = getattr(lattice_pkg, name)(f), getattr(compact, name)(f)
            assert np.array_equal(np.asarray(got), np.asarray(want)), name
    for p in range(D + 1):
        for q in range(D + 1 - p):
            a, b = H.random(p), H.random(q)
            want = np.asarray(compact.wedge(a, b))
            assert np.array_equal(np.asarray(lattice_pkg.wedge(a, b)), want)
            assert np.array_equal(np.asarray(lattice_pkg.cup(a, b, 0)), want)
            assert np.array_equal(np.asarray(lattice_pkg.cup(a, b)), want)


def test_dispatch_hypercubic_higher_cup_raises():
    r'''Upstream has no hypercubic $\cup_{i>0}$: ``supervillain.lattice.cup(a, b, i)`` on ``Form``s raises ``NotImplementedError`` for $i > 0$.'''
    # SPEC §11 Q2.
    H = Lattice(3, 3)
    for i in (1, 2):
        with pytest.raises(NotImplementedError):
            lattice_pkg.cup(H.random(2), H.random(2), i)


def test_dispatchers_keep_hypercubic_docstrings():
    r'''Each dispatcher carries the hypercubic docstring (plus a note), so the existing documentation is unchanged.'''
    # SPEC §11 Q2.
    for name in ('d', 'delta', 'laplacian', 'star', 'wedge'):
        dispatcher = getattr(lattice_pkg, name)
        assert dispatcher.__doc__.startswith(getattr(compact, name).__doc__)
        assert 'barycubic' in dispatcher.__doc__


# =============================================================================================
# Regression tests from the adversarial review of the implementation.
#
# The review found no error in the mathematics, but mutation testing showed that the contracts
# below were unguarded: breaking them left every test above green.  Each test pins one contract.
# =============================================================================================

import pickle


# ---- Mixing hypercubic and barycubic forms --------------------------------------------------

@pytest.mark.parametrize("i", [0, 1])
def test_mixing_hypercubic_and_barycubic_raises(i):
    r'''``supervillain.lattice.wedge`` and ``cup`` refuse a hypercubic ``Form`` paired with a ``BarycubicForm``, in either order, with ``TypeError``.'''
    # Review finding: cup(hyper 0-form, bary q-form) used to return a hypercubic Form silently.
    # Degrees are chosen so that (p, q, i) is legal in both orders: only the kind mismatch is wrong.
    H, L = Lattice(2, 3), lattice(2, 3)
    pairs = [(H.random(1), int_form(L, 1, 151)), (H.random(1), int_form(L, 2, 152))]
    if i == 0:
        pairs.append((H.zeros(0) + 1.0, int_form(L, 1, 150)))
    for hyper, bary in pairs:
        for x, y in [(hyper, bary), (bary, hyper)]:
            if i == 0:
                with pytest.raises(TypeError):
                    lattice_pkg.wedge(x, y)
            with pytest.raises(TypeError):
                lattice_pkg.cup(x, y, i)


def test_barycubic_products_reject_foreign_second_argument():
    r'''``barycubic.cup`` and ``barycubic.inner`` raise ``TypeError`` when the second argument is a plain ndarray or a hypercubic ``Form``.'''
    L, H = lattice(2, 3), Lattice(2, 3)
    a = int_form(L, 1, 152)
    for foreign in (np.asarray(a), H.random(1)):
        with pytest.raises(TypeError):
            bc.cup(a, foreign)
        with pytest.raises(TypeError):
            bc.inner(a, foreign)


# ---- HDF5 of views and of orphaned forms ------------------------------------------------------

def test_h5_views_are_written_as_plain_arrays(tmp_path):
    r'''Views of a form (``f[0]``, ``f.reshape(-1)``, ``f.T``) are written and read back as plain ndarrays equal to the view; the full form still reads back as a ``BarycubicForm``.'''
    L = lattice(2, 3)
    f = int_form(L, 1, 153)
    views = {'row': f[0], 'flat': f.reshape(-1), 'transpose': f.T}
    with h5.File(tmp_path / 'views.h5', 'w') as h:
        for key, v in views.items():
            Data.write(h, key, v)
        Data.write(h, 'full', f)
    with h5.File(tmp_path / 'views.h5', 'r') as h:
        for key, v in views.items():
            back = Data.read(h[key])
            assert type(back) is np.ndarray, key
            assert np.array_equal(back, np.asarray(v)), key
        _assert_same_form(Data.read(h['full']), f)


def test_h5_orphaned_form_leaves_no_broken_dataset(tmp_path):
    r'''Writing a lattice-less ``np.zeros(shape).view(BarycubicForm)`` either stores a plain array or raises and leaves no partial dataset behind; the file stays usable.'''
    L = lattice(2, 3)
    shape = (ref.simplices_per_site(2, 1), 3, 3)
    orphan = np.arange(np.prod(shape), dtype=float).reshape(shape).view(bc.BarycubicForm)
    path = tmp_path / 'orphan.h5'
    with h5.File(path, 'w') as h:
        try:
            Data.write(h, 'orphan', orphan)
            written = True
        except Exception:
            written = False
            assert 'orphan' not in h
        Data.write(h, 'after', int_form(L, 1, 154))
    with h5.File(path, 'r') as h:
        if written:
            back = Data.read(h['orphan'])
            assert type(back) is np.ndarray
            assert np.array_equal(back, np.asarray(orphan))
        else:
            assert 'orphan' not in h
        _assert_same_form(Data.read(h['after']), int_form(L, 1, 154))


# ---- Cached tables cannot be corrupted through the returned arrays ---------------------------

_TABLE_GETTERS = {
    'operator_table_d': lambda L: L.operator_table('d', 1),
    'operator_table_dT': lambda L: L.operator_table('dT', 1),
    'cup_table': lambda L: L.cup_table(1, 1, 1),
    'hodge_weights': lambda L: L.hodge_weights(1),
    'local_vertices': lambda L: L.local_vertices(1),
    'rank_sets': lambda L: L.rank_sets[1],
    'orientation': lambda L: L.orientation,
}


def _fingerprint(L):
    r'''Results on L that depend on every cached table: d, δ, ∪_1, ∫, and the tables themselves.'''
    D = L.D
    return [np.asarray(bc.d(int_form(L, p, 160))) for p in range(D)] + \
           [np.asarray(bc.delta(int_form(L, 1, 161))), np.asarray(bc.cup(int_form(L, 1, 162), int_form(L, 1, 163), 1)),
            np.asarray(bc.integrate(int_form(L, D, 164))),
            np.array(L.hodge_weights(1)), np.array(L.local_vertices(1)), np.array(L.rank_sets[1])]


@pytest.mark.parametrize("name", list(_TABLE_GETTERS))
def test_cached_tables_are_read_only(name):
    r'''
    An array returned by a table accessor is either read-only (``setflags(write=True)`` raises) or a
    private copy: mutating it leaves ``d``, ``δ``, ``∪_1``, ``∫`` and the tables of a fresh lattice unchanged.
    '''
    D, N = 2, 3
    before = _fingerprint(bc.BarycubicLattice(D, N))
    victim = bc.BarycubicLattice(D, N)
    arr = _TABLE_GETTERS[name](victim)
    try:
        arr.setflags(write=True)
    except ValueError:
        return                                          # read-only and cannot be made writable
    arr[...] = 7 if np.issubdtype(np.asarray(arr).dtype, np.integer) else 3.5
    after = _fingerprint(bc.BarycubicLattice(D, N))
    for x, y in zip(before, after):
        assert np.array_equal(x, y), name


# ---- Dtypes ---------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.uint64, np.bool_])
def test_unsigned_and_bool_forms_rejected(dtype):
    r'''
    Unsigned and boolean cochains cannot represent $d$ or signed products: ``d``, ``cup`` (either slot),
    ``coarsen``, ``half_shift`` and ``integrate`` raise ``TypeError`` (the constructor may reject them first).
    '''
    L = lattice(2, 3)
    good = int_form(L, 1, 170)

    def bad(p):
        return bc.BarycubicForm(np.ones((ref.simplices_per_site(2, p), 3, 3), dtype=dtype), degree=p, lattice=L)

    for call in (lambda: bc.d(bad(1)), lambda: bc.cup(bad(1), good), lambda: bc.cup(good, bad(1)),
                 lambda: bc.coarsen(bad(1)), lambda: bc.half_shift(bad(1)), lambda: bc.integrate(bad(2))):
        with pytest.raises(TypeError):
            call()


@pytest.mark.parametrize("D,N", [(2, 3), (3, 2)])
def test_int32_forms_stay_integer_and_exact(D, N):
    r'''``int32`` and ``int64`` forms give integer results under ``d``, ``cup``, ``coarsen``, ``half_shift`` and ``integrate``, equal to each other exactly.'''
    L = lattice(D, N)
    results = {}
    for dtype in (np.int32, np.int64):
        a, b, top = int_form(L, 1, 171, dtype=dtype), int_form(L, 1, 172, dtype=dtype), int_form(L, D, 173, dtype=dtype)
        out = [bc.d(a), bc.cup(a, b), bc.cup(a, b, 1), bc.coarsen(a), bc.half_shift(a), bc.integrate(top)]
        for x in out:
            assert np.issubdtype(np.asarray(x).dtype, np.integer), dtype
        results[dtype] = [np.asarray(x) for x in out]
    for x, y in zip(results[np.int32], results[np.int64]):
        assert np.array_equal(x, y)


# ---- Degree and range checks ----------------------------------------------------------------

@pytest.mark.parametrize("D,N", [(1, 3), (2, 3)])
def test_inner_rejects_mixed_degrees(D, N):
    r'''``inner`` of forms of different degree raises ``ValueError``, even at D = 1 where $f_0 = f_1$ and the shapes coincide.'''
    L = lattice(D, N)
    for p in range(D + 1):
        for q in range(D + 1):
            if p != q:
                with pytest.raises(ValueError):
                    bc.inner(float_form(L, p, 180), float_form(L, q, 181))


@pytest.mark.parametrize("D,N", [(1, 3), (2, 3)])
def test_integrate_rejects_non_top_forms(D, N):
    r'''``integrate`` of a p-form with p < D raises ``ValueError``, even at D = 1 where a 0-form has the shape of a 1-form.'''
    L = lattice(D, N)
    for p in range(D):
        with pytest.raises(ValueError):
            bc.integrate(int_form(L, p, 182))


@pytest.mark.parametrize("D,N", [(1, 2), (3, 2)])
def test_degree_out_of_range_rejected(D, N):
    r'''``zeros(D+1)``, ``zeros(-1)``, and a ``BarycubicForm`` constructed with degree D+1 or −1 raise ``ValueError``.'''
    L = lattice(D, N)
    for p in (D + 1, -1):
        with pytest.raises(ValueError):
            L.zeros(p)
        with pytest.raises(ValueError):
            bc.BarycubicForm(np.zeros((ref.simplices_per_site(D, D),) + (N,) * D), degree=p, lattice=L)


def _wrong_shape_views(L, p, seed):
    f = int_form(L, p, seed)
    return [f[:, :-1], f[:-1]]          # too few sites along an axis; too few labels


@pytest.mark.parametrize("D,N", [(1, 3), (2, 3)])
def test_operators_reject_wrong_shapes(D, N):
    r'''An input whose array shape does not match its degree and lattice makes every operator raise ``ValueError``.'''
    L = lattice(D, N)
    ok1 = int_form(L, 1, 191)
    for bad in _wrong_shape_views(L, 1, 190):
        for call in (lambda: bc.d(bad), lambda: bc.delta(bad), lambda: bc.laplacian(bad), lambda: bc.star(bad),
                     lambda: bc.cup(bad, int_form(L, 0, 192)), lambda: bc.cup(int_form(L, 0, 192), bad),
                     lambda: bc.inner(bad, bad), lambda: bc.inner(ok1, bad), lambda: bc.coarsen(bad),
                     lambda: bc.half_shift(bad), lambda: bc.transform(bad, tuple(range(D)), (1,) * D, (0,) * D)):
            with pytest.raises(ValueError):
                call()
    for bad in _wrong_shape_views(L, D, 193):
        with pytest.raises(ValueError):
            bc.integrate(bad)


# ---- Pickling -------------------------------------------------------------------------------

@pytest.mark.parametrize("D,N", [(1, 2), (3, 2), (4, 2)])
def test_forms_pickle(D, N):
    r'''``pickle`` round trips of a ``BarycubicForm`` and a ``BarycubicDualForm`` keep class, degree, an equal lattice, dtype and values; arithmetic works afterwards.'''
    L = lattice(D, N)
    for f in (int_form(L, 1, 200), float_form(L, D, 201, dtype=complex), bc.star(float_form(L, 1, 202))):
        g = pickle.loads(pickle.dumps(f))
        assert type(g) is type(f) and g.degree == f.degree
        assert g.lattice == L and g.lattice.D == D and g.lattice.N == N
        assert g.dtype == f.dtype and np.array_equal(np.asarray(g), np.asarray(f))
        if isinstance(f, bc.BarycubicDualForm):
            assert g.primal_degree == f.primal_degree and g.dual is True
        h = g + g
        assert type(h) is type(f) and h.degree == f.degree
        assert np.array_equal(np.asarray(h), 2 * np.asarray(f))


# ---- Developer self-check -------------------------------------------------------------------

@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_developer_self_check(D):
    r'''``barycubic._self_check(D)``, the implementation's own closed-form checks of its per-D tables, runs without error.'''
    bc._self_check(D)


# =============================================================================================
# Regression tests from adversarial review.
# =============================================================================================

def _boundary_int_form(L, p, dtype):
    r'''A form whose entries alternate between the dtype's maximum and minimum (flat C order).'''
    info = np.iinfo(dtype)
    shape = (ref.simplices_per_site(L.D, p),) + (L.N,) * L.D
    flat = np.where(np.arange(int(np.prod(shape))) % 2 == 0, info.max, info.min).astype(dtype)
    return bc.BarycubicForm(flat.reshape(shape), degree=p, lattice=L)


def _promoted(f):
    return bc.BarycubicForm(np.asarray(f).astype(np.float64), degree=f.degree, lattice=f.lattice)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
@pytest.mark.parametrize("D,N", [(1, 2), (2, 3)])
def test_weighted_operators_do_not_overflow_integers(D, N, dtype):
    r'''
    On signed integers at their dtype limits, ``laplacian`` and ``delta`` equal their values on the float64-promoted
    input exactly, ``inner(f, f)`` equals the float computation, and $\langle f, \Delta f\rangle \ge 0$.
    '''
    # Review reproduction: D=1, N=2, int8 0-form of 127 and −128 gave inner(f, laplacian(f)) = −2040.
    L = lattice(D, N)
    for p in range(D + 1):
        f = _boundary_int_form(L, p, dtype)
        g = _promoted(f)
        assert np.array_equal(np.asarray(bc.laplacian(f)), np.asarray(bc.laplacian(g))), p
        if p > 0:
            assert np.array_equal(np.asarray(bc.delta(f)), np.asarray(bc.delta(g))), p
        assert bc.inner(f, f) == pytest.approx(bc.inner(g, g), rel=1e-12)
        assert bc.inner(f, bc.laplacian(f)) == pytest.approx(bc.inner(g, bc.laplacian(g)), rel=1e-12, abs=1e-6)
        assert bc.inner(f, bc.laplacian(f)) >= 0


def test_int8_laplacian_review_reproduction():
    r'''The review's case: at D = 1, N = 2 an int8 0-form of 127 and −128 has $\langle f, \Delta f\rangle \ge 0$ (it was −2040).'''
    L = lattice(1, 2)
    f = bc.BarycubicForm(np.array([[127, -128], [-128, 127]], dtype=np.int8), degree=0, lattice=L)
    assert bc.inner(f, bc.laplacian(f)) == pytest.approx(bc.inner(_promoted(f), bc.laplacian(_promoted(f))), rel=1e-12)
    assert bc.inner(f, bc.laplacian(f)) >= 0


@pytest.mark.parametrize("bad", [
    dict(shift=(0.5, 0, 0)), dict(shift=(1.0, 0, 0)),
    dict(perm=(0.0, 1.0, 2.0)), dict(perm=(0.5, 1, 2)),
    dict(flips=(1.0, 1.0, 1.0)), dict(flips=(-1.0, 1, 1)),
])
def test_transform_rejects_non_integer_arguments(bad):
    r'''``transform`` with a fractional or float ``shift``, ``perm`` or ``flips`` raises ``TypeError``, even for integral floats.'''
    L = lattice(3, 3)
    args = dict(perm=(0, 1, 2), flips=(1, 1, 1), shift=(0, 0, 0))
    args.update(bad)
    with pytest.raises(TypeError):
        bc.transform(int_form(L, 1, 210), args['perm'], args['flips'], args['shift'])


@pytest.mark.parametrize("bad", [dict(perm=(0, 0, 1)), dict(perm=(0, 1, 3)), dict(perm=(0, 1)),
                                 dict(flips=(2, 1, 1)), dict(flips=(0, 1, 1))])
def test_transform_rejects_invalid_group_elements(bad):
    r'''A ``perm`` that is not a permutation of the axes, or a flip other than ±1, raises ``ValueError``.'''
    L = lattice(3, 3)
    args = dict(perm=(0, 1, 2), flips=(1, 1, 1), shift=(0, 0, 0))
    args.update(bad)
    with pytest.raises(ValueError):
        bc.transform(int_form(L, 1, 211), args['perm'], args['flips'], args['shift'])


def test_lattice_constructor_validates_arguments():
    r'''``BarycubicLattice(D, N)`` needs integers (``TypeError`` for 2.9 or 3.0) with $D \ge 1$ and $N \ge 2$ (``ValueError``).'''
    for D, N in [(2.9, 3), (2, 3.0), (2.0, 3)]:
        with pytest.raises(TypeError):
            bc.BarycubicLattice(D, N)
    for D, N in [(0, 3), (2, 1), (-1, 3), (2, 0)]:
        with pytest.raises(ValueError):
            bc.BarycubicLattice(D, N)
