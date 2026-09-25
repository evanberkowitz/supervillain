import os
import subprocess
import sys
from pathlib import Path

import supervillain

# The eight kernel variants in supervillain/lattice/_kernels.py share one factory.  numba
# names each compiled function `qualname$<process-local counter>` and writes that name into
# its on-disk cache, so if the variants shared a qualname, a variant compiled in one process
# and loaded from the cache in another could land on the same symbol as a variant compiled
# fresh there, and calling it raised RuntimeError: In 'NRT_adapt_ndarray_to_python', 'descr'
# is NULL.  This test compiles the variants in conflicting orders in separate processes that
# share one cache directory; each process checks every kernel it calls against the reference.

_CHILD = r"""
import sys
import numpy as np
from math import comb
from supervillain.lattice import Lattice, _kernels
from supervillain.lattice import reference as ref

L = Lattice(D=3, N=5)
spec = {
    "d":          ("D_KERNELS",      1, 2, ref.reference_d),
    "delta":      ("DELTA_KERNELS",  2, 1, ref.reference_delta),
    "face_sum":   ("FACE_KERNELS",   2, 1, ref.reference_face_sum),
    "coface_sum": ("COFACE_KERNELS", 1, 2, ref.reference_coface_sum),
}
for variant in sys.argv[1:]:
    op, which = variant.rsplit(":", 1)
    kernels, degree, out_degree, reference = spec[op]
    f = L.random(degree)
    src = np.ascontiguousarray(np.asarray(f)).reshape(f.shape[0], L.sites)
    out = np.zeros((comb(L.D, out_degree), L.sites))
    getattr(_kernels, kernels)[int(which)](src, out, L.operator_table(op, degree), L.N, L.D)
    expected = np.asarray(reference(f)).reshape(out.shape)
    assert (out == expected).all(), f"{variant} disagrees with the reference"

# A shared name is the bug even when the call happens not to crash: check it directly.
names = [cres.fndesc.mangled_name
         for group in ("D_KERNELS", "DELTA_KERNELS", "FACE_KERNELS", "COFACE_KERNELS")
         for kernel in getattr(_kernels, group) for cres in kernel.overloads.values()]
assert len(names) == len(set(names)), "two kernel variants share a compiled symbol name"
"""

_VARIANTS = [f"{op}:{which}" for which in (0, 1) for op in ("d", "delta", "face_sum", "coface_sum")]


def _run(order, cache_dir):
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_dir))
    root = str(Path(supervillain.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([root] + [p for p in (os.environ.get("PYTHONPATH"),) if p])
    result = subprocess.run([sys.executable, "-c", _CHILD, *order], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"order {order} failed:\n{result.stderr}"


def test_kernel_variants_share_a_cache_directory(tmp_path):
    # Each process first compiles variants nobody has cached, then loads the ones earlier
    # processes cached, so fresh compilations and cache loads meet in one process.
    # The second order is the one that crashed: face_sum compiles fresh and d loads from the
    # cache under the same name.
    _run(["d:0"], tmp_path)
    _run(["face_sum:0", "coface_sum:0", "d:0"], tmp_path)
    _run(["delta:0"] + _VARIANTS[4:] + ["d:0", "face_sum:0", "coface_sum:0"], tmp_path)
    _run(_VARIANTS[::-1], tmp_path)
