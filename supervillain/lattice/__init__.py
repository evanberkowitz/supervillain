import numpy as np


def _dimension(n):
    """FFT-convention coordinates for a periodic direction of size n."""
    return np.array(
        list(range(0, n // 2 + 1)) + list(range(-n // 2 + 1, 0)),
        dtype=int,
    )


from supervillain.lattice.two_dimensional import Lattice2D, _Lattice2D
from supervillain.lattice.compact import Lattice, Form, delta_sparse, d_sparse, coface_sum_at, face_sum_at, push, pull
from supervillain.lattice import compact, barycubic
from supervillain.lattice.barycubic import (
    BarycubicLattice, BarycubicForm, BarycubicDualForm,
    inverse_star, inner, integrate, coarsen, transform, half_shift,
)


# The exterior-calculus operators below are shared by both lattices.  Each package-level name
# looks at the type of its argument and hands the work to the module that owns that lattice:
# hypercubic Forms go to compact.py, barycubic forms to barycubic.py.  The per-module functions
# stay importable directly (supervillain.lattice.compact.d, supervillain.lattice.barycubic.d),
# and a dispatcher carries the hypercubic docstring so the existing documentation is unchanged.

def _dispatching(hypercubic, barycubic_version, name):
    def operator(f, *args, **kwargs):
        if isinstance(f, BarycubicForm):
            return barycubic_version(f, *args, **kwargs)
        return hypercubic(f, *args, **kwargs)
    operator.__name__ = operator.__qualname__ = name
    operator.__module__ = __name__
    operator.__wrapped__ = hypercubic
    operator.__doc__ = (hypercubic.__doc__ or '') + f"""
    .. note::
        On a :class:`~supervillain.lattice.barycubic.BarycubicForm` this calls
        :func:`supervillain.lattice.barycubic.{barycubic_version.__name__}` instead.
    """
    return operator


def _dispatching_product(hypercubic, barycubic_version, name):
    # A product has two form arguments, and both must belong to the same lattice: a barycubic
    # form multiplied by a hypercubic one (or by any array that is not a barycubic form) has no
    # meaning, and must not be silently routed by the type of the first argument alone.
    def operator(a, b, *args, **kwargs):
        a_barycubic, b_barycubic = isinstance(a, BarycubicForm), isinstance(b, BarycubicForm)
        if a_barycubic != b_barycubic:
            raise TypeError(f"cannot mix hypercubic and barycubic forms in {name}: got "
                            f"{type(a).__name__} and {type(b).__name__}")
        if a_barycubic:
            return barycubic_version(a, b, *args, **kwargs)
        return hypercubic(a, b, *args, **kwargs)
    operator.__name__ = operator.__qualname__ = name
    operator.__module__ = __name__
    operator.__wrapped__ = hypercubic
    operator.__doc__ = (hypercubic.__doc__ or '') + f"""
    .. note::
        On two :class:`~supervillain.lattice.barycubic.BarycubicForm` arguments this calls
        :func:`supervillain.lattice.barycubic.{barycubic_version.__name__}` instead.  Mixing a
        barycubic form with a hypercubic one raises ``TypeError``.
    """
    return operator


def _hypercubic_cup(a, b, i=0):
    r"""
    Cup product $a \cup_i b$ of two hypercubic forms.

    Only $i = 0$ is available on the hypercubic lattice, where the cup product is the
    :func:`wedge`.  The higher products $\cup_{i>0}$ are implemented on the barycubic lattice.
    """
    if i == 0:
        return compact.wedge(a, b)
    raise NotImplementedError(f"cup_{i} is not implemented on the hypercubic lattice; only cup_0 (the wedge) is.")


d = _dispatching(compact.d, barycubic.d, 'd')
delta = _dispatching(compact.delta, barycubic.delta, 'delta')
laplacian = _dispatching(compact.laplacian, barycubic.laplacian, 'laplacian')
star = _dispatching(compact.star, barycubic.star, 'star')
wedge = _dispatching_product(compact.wedge, barycubic.wedge, 'wedge')
cup = _dispatching_product(_hypercubic_cup, barycubic.cup, 'cup')
