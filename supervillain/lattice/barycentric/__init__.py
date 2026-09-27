r"""
The barycentric subdivision of the hypercubic lattice.

Two implementations.  :mod:`~supervillain.lattice.barycentric.compact` stores
one entry per fine simplex and is what production code should use; its names
are hoisted here.  :mod:`~supervillain.lattice.barycentric.doubled` stores a
cochain on the doubled $(2N)^D$ array, the obvious way, and is the fixed target
for correctness; reach it as
``supervillain.lattice.barycentric.doubled``.

The names are deliberately **not** hoisted into :mod:`supervillain.lattice`,
where ``Lattice``, ``Form``, ``d``, ``delta`` and ``wedge`` already mean the
hypercubic ones.
"""
from supervillain.lattice.barycentric import doubled
from supervillain.lattice.barycentric import compact
from supervillain.lattice.barycentric.compact import (
    Lattice, Form, d, delta, cup, wedge,
    poincare_star, hodge_star, hodge_star_inverse,
    push, pull, translate, reflect, permute, tau, translate_diagonal,
)
