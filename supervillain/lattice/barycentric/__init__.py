r"""
The barycentric subdivision of the hypercubic lattice.

:mod:`~supervillain.lattice.barycentric.doubled` stores a cochain on the
doubled $(2N)^D$ array, the obvious way, and is the fixed target for
correctness.  :mod:`~supervillain.lattice.barycentric.compact` stores one
entry per simplex and is what production code should use.
"""
from supervillain.lattice.barycentric import doubled
