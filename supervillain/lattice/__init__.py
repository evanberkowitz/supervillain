import numpy as np


def _dimension(n):
    """FFT-convention coordinates for a periodic direction of size n."""
    return np.array(
        list(range(0, n // 2 + 1)) + list(range(-n // 2 + 1, 0)),
        dtype=int,
    )


from supervillain.lattice.two_dimensional import Lattice2D, _Lattice2D
from supervillain.lattice.compact import Lattice
