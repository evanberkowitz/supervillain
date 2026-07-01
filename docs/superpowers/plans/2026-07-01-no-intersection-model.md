# No-Intersection Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the prototype `TheoAction` and the three worm/link/wrapping generators into the library as the `NoIntersections` action plus a `supervillain.generator.no_intersection` subpackage, and flesh out `supervillain/no_intersection.rst`.

**Architecture:** `NoIntersections` subclasses `Villain`, reusing its energy/links/configurations/gauge machinery and overriding only `__init__` (hard D=4), `__str__`, and `valid()` (constraint `dn∧dn=0`). The three generators (`ThetaWorm`, `ConstrainedLinkUpdate`, `WrappingLoopUpdate`) are ported verbatim from the working-tree file `theoWorm.py` into one-class-per-file modules, sharing a `charge()` helper that delegates to the existing `supervillain.observable.topological._topological_charge`. A `Hammer(S)` combiner sequences them with `villain.SiteUpdate` (for φ).

**Tech Stack:** Python, numpy, the in-repo `supervillain` package, pytest, Sphinx.

## Global Constraints

- The model is **D=4 only**; every action and generator constructor raises `ValueError` when `lattice.D != 4`.
- θ is a Lagrange multiplier, never a sampled field; field content stays `{phi, n}` (real 0-form, integer 1-form).
- Do **not** commit `theo.py`, `theoWorm.py`, `theo-worm.md`, `worm_probe.py`, `two_defect_move.py`, `two_worm_move.py`. `theoWorm.py` is the *source to port from* but stays untracked; the `theo-worm.md` derivation is woven narratively into docstrings/rst, not committed as a file.
- Reuse `supervillain.observable.topological._topological_charge` for `dn∧dn`; do not duplicate the computation.
- Generators guard on `isinstance(S, supervillain.action.NoIntersections)` (not `Villain`).
- Run everything with `uv run` (e.g. `uv run pytest`), per project convention.
- Docstring math is LaTeX in `r"""..."""` (`.. math::` / `$…$`), `\texttt{}` for code identifiers — never unicode.
- Work happens on the current branch `feature/intersection-model`.

---

### Task 1: `NoIntersections` action

**Files:**
- Create: `supervillain/action/no_intersection.py`
- Modify: `supervillain/action/__init__.py`
- Test: `test/test_no_intersection.py`

**Interfaces:**
- Consumes: `supervillain.action.Villain` (base class); `supervillain.lattice.{Lattice, d, wedge}`.
- Produces: `supervillain.action.NoIntersections(lattice, kappa)` with `.W == 1`, inherited `__call__(phi, n)`, `links`, `local`, `configurations`, `gauge_transform`, and overridden `valid(configuration) -> bool` enforcing `dn∧dn == 0`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_no_intersection.py`:

```python
#!/usr/bin/env python

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d, wedge


def _cold(S):
    # A single cold configuration dict {'phi': 0-form, 'n': 0-form}.
    return S.configurations(1)[0]


def test_no_intersections_requires_D4():
    L = Lattice(3, 4)
    with pytest.raises(ValueError):
        supervillain.action.NoIntersections(L, kappa=0.5)


def test_no_intersections_constructs_in_D4():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.5)
    assert S.W == 1
    assert S.Lattice is L
    assert 'NoIntersections' in str(S)


def test_valid_accepts_cold_config():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.5)
    assert S.valid(_cold(S))


def test_valid_rejects_intersecting_config():
    # Two crossing vortex sheets: F_{01}=1 and F_{23}=1 on the hypercube at the
    # origin, so (dn ∧ dn) is nonzero there.
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.5)
    n = L.zeros(1, dtype=int)
    n[1, 1, 0, 0, 0] = 1   # n_1 at origin + e_0  -> F_{01}(origin) = 1
    n[3, 0, 0, 1, 0] = 1   # n_3 at origin + e_2  -> F_{23}(origin) = 1
    assert np.any(np.asarray(wedge(d(n), d(n))) != 0)
    assert not S.valid({'phi': L.zeros(0), 'n': n})


def test_valid_and_action_are_gauge_invariant():
    L = Lattice(4, 5)
    S = supervillain.action.NoIntersections(L, kappa=0.7)
    cfg = _cold(S)
    k = L.zeros(0, dtype=int)
    k += np.random.default_rng(0).integers(-3, 4, size=k.shape)
    gauged = S.gauge_transform(cfg, k)
    assert S.valid(gauged) == S.valid(cfg)
    assert np.isclose(S(**gauged), S(**cfg))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_no_intersection.py -q`
Expected: FAIL — `AttributeError: module 'supervillain.action' has no attribute 'NoIntersections'`.

- [ ] **Step 3: Create the action module**

Create `supervillain/action/no_intersection.py`:

```python
#!/usr/bin/env python

import numpy as np
from supervillain.action.villain import Villain
from supervillain.lattice import Lattice, d, wedge

import logging
logger = logging.getLogger(__name__)


class NoIntersections(Villain):
    r'''
    The No-Intersection model is the modified-Villain action

    .. math::
       S = \frac{\kappa}{2} \sum_\ell (d\phi - 2\pi n)_\ell^2

    restricted to configurations obeying the *no-intersection constraint*

    .. math::
       Q = (dn \wedge dn) = 0 \quad\text{on every hypercube.}

    The constraint is the path integral of a Lagrange-multiplier top-form
    $\theta$ entering the action as $S = S_\text{Villain} + i\,\theta\,(dn\wedge dn)$;
    $\theta$ is never sampled, so the field content is the Villain content
    $\{\phi, n\}$.  Because $dn\wedge dn$ is a 4-form the model is only defined
    (and only interesting --- it carries a mixed axial-vector-vector anomaly) in
    $D = 4$, which this class hard-assumes.

    Parameters
    ----------
    lattice: supervillain.lattice.Lattice
        A four-dimensional lattice on which $\phi$ and $n$ live.
    kappa: float
        The $\kappa$ in the overall coefficient.
    '''

    def __init__(self, lattice, kappa):
        if not isinstance(lattice, Lattice):
            raise TypeError(f'NoIntersections requires a supervillain.lattice.Lattice, got {type(lattice).__name__}')
        if lattice.D != 4:
            raise ValueError(f'The No-Intersection model is only defined in D = 4, got D = {lattice.D}.')
        super().__init__(lattice, kappa, W=1)

    def __str__(self):
        return f'NoIntersections({self.Lattice}, κ={self.kappa})'

    def valid(self, configuration):
        r'''
        Returns true if the no-intersection constraint $dn \wedge dn = 0$ holds on
        every hypercube.

        Parameters
        ----------
        configuration: dict
            A dictionary that at least contains ``n``.

        Returns
        -------
        bool:
            Is the constraint satisfied everywhere?
        '''
        dn = d(configuration['n'])
        return bool(np.isclose(wedge(dn, dn), 0).all())
```

- [ ] **Step 4: Register the class**

Modify `supervillain/action/__init__.py` to add the export after the existing lines:

```python
from .villain import Villain
from .worldline import Worldline
from .no_intersection import NoIntersections
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_no_intersection.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add supervillain/action/no_intersection.py supervillain/action/__init__.py test/test_no_intersection.py
git commit -m "Add the NoIntersections action (dn∧dn = 0, D=4)"
```

---

### Task 2: shared `charge()` helper + `ThetaWorm` generator

**Files:**
- Create: `supervillain/generator/no_intersection/__init__.py`
- Create: `supervillain/generator/no_intersection/charge.py`
- Create: `supervillain/generator/no_intersection/worm.py`
- Modify: `supervillain/generator/__init__.py`
- Test: `test/test_no_intersection_generators.py`

**Interfaces:**
- Consumes: `supervillain.action.NoIntersections`; `supervillain.observable.topological._topological_charge`; `supervillain.generator.Generator`; `supervillain.h5.ReadWriteable`; `supervillain.batch.Batch`; `supervillain.lattice.{Form, d}`.
- Produces:
  - `supervillain.generator.no_intersection.charge.charge(n) -> np.ndarray` of shape `(1,) + lattice.dims` (the density `dn∧dn`).
  - `supervillain.generator.no_intersection.ThetaWorm(S)` with `.step(configuration) -> dict` (adds `'Theta_Theta'` shape `lattice.dims` and `'Worm_Length'` scalar), `.inline_observables(steps)`, `.report()`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_no_intersection_generators.py`:

```python
#!/usr/bin/env python

import numpy as np
import pytest
import supervillain
from supervillain.lattice import Lattice, d, wedge


def _action(kappa=0.3, N=5):
    L = Lattice(4, N)
    return supervillain.action.NoIntersections(L, kappa=kappa)


def _cold(S):
    return S.configurations(1)[0]


def test_charge_matches_topological_charge():
    from supervillain.generator.no_intersection.charge import charge
    from supervillain.observable.topological import _topological_charge
    L = Lattice(4, 5)
    n = L.zeros(1, dtype=int)
    n[1, 1, 0, 0, 0] = 1
    n[3, 0, 0, 1, 0] = 1
    assert np.array_equal(charge(n), np.asarray(_topological_charge(L, n)))


def test_theta_worm_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.ThetaWorm(V)


def test_theta_worm_requires_D4():
    # NoIntersections cannot even be built in D != 4, so a Villain stand-in in D=2
    # exercises the worm's own dimensional guard.
    L = Lattice(2, 4)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.ThetaWorm(V)


def test_theta_worm_preserves_validity_and_closes():
    S = _action()
    worm = supervillain.generator.no_intersection.ThetaWorm(S)
    out = worm.step(_cold(S))
    assert S.valid(out)
    assert np.asarray(out['Theta_Theta']).shape == S.Lattice.dims
    assert np.isscalar(out['Worm_Length']) or np.asarray(out['Worm_Length']).shape == ()


def test_theta_worm_inline_observable_keys():
    S = _action()
    worm = supervillain.generator.no_intersection.ThetaWorm(S)
    obs = worm.inline_observables(3)
    assert set(obs) == {'Theta_Theta', 'Worm_Length'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_no_intersection_generators.py -q`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` for `supervillain.generator.no_intersection`.

- [ ] **Step 3: Create the shared `charge` helper**

Create `supervillain/generator/no_intersection/charge.py`:

```python
#!/usr/bin/env python

import numpy as np
from supervillain.observable.topological import _topological_charge


def charge(n):
    r"""
    The topological-charge density $Q = dn \wedge dn$ as a plain array carrying one
    integer per hypercube (shape ``(1,) + lattice.dims``).

    Delegates to :func:`supervillain.observable.topological._topological_charge`;
    ``n`` must be a :class:`~supervillain.lattice.Form` (it carries its own lattice).
    """
    return np.asarray(_topological_charge(n.lattice, n))
```

- [ ] **Step 4: Port the `ThetaWorm` class**

Create `supervillain/generator/no_intersection/worm.py` by copying the `ThetaWorm` class (and the `_SEED_HEAD` / `_SEED` module constants) verbatim from the working-tree file `theoWorm.py` (the class spans roughly `theoWorm.py:56`–`316`), with these exact changes:

1. Module header imports become:

```python
#!/usr/bin/env python

from collections import deque
from itertools import permutations
import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.batch import Batch
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)
```

   (Drop the old top-of-file `charge` function and the `wedge` import — `charge` now comes from `.charge`.)

2. In `ThetaWorm.__init__`, change the guard from
   `if not isinstance(S, supervillain.action.Villain):` to
   `if not isinstance(S, supervillain.action.NoIntersections):`
   and its message to `'ThetaWorm requires a NoIntersections action.'`

3. In the class docstring, replace the pointer "See theo-worm.md" / "See theo-worm.md." prose with "See :ref:`the No-Intersection model <no_intersection>`." Remove any remaining bare references to the untracked `theo-worm.md`/`two_worm_move.py` filenames, keeping the surrounding physics prose intact.

Everything else (the `_build_library`, `_change_from_shape`, `_sheet_segment`, `_delta_S`, `inline_observables`, `step`, `report` methods) is copied unchanged.

- [ ] **Step 5: Create the subpackage `__init__` and register it**

Create `supervillain/generator/no_intersection/__init__.py`:

```python
#!/usr/bin/env python

from .worm import ThetaWorm
```

Modify `supervillain/generator/__init__.py` to add, after the existing `import supervillain.generator.worldline` line:

```python
import supervillain.generator.no_intersection
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest test/test_no_intersection_generators.py -q`
Expected: PASS (5 passed). If a `KeyError`/stall makes `test_theta_worm_preserves_validity_and_closes` hang, it indicates the worm never closes on the cold background — re-run; the closure probability is `1/(2D+1)` per visit to the tail and termination is expected quickly for `N=5`.

- [ ] **Step 7: Commit**

```bash
git add supervillain/generator/no_intersection/ supervillain/generator/__init__.py test/test_no_intersection_generators.py
git commit -m "Add the ThetaWorm generator for the No-Intersection model"
```

---

### Task 3: `ConstrainedLinkUpdate` generator

**Files:**
- Create: `supervillain/generator/no_intersection/link.py`
- Modify: `supervillain/generator/no_intersection/__init__.py`
- Test: `test/test_no_intersection_generators.py` (append)

**Interfaces:**
- Consumes: `charge` from `.charge`; `supervillain.action.NoIntersections`; `supervillain.lattice.{Form, d}`.
- Produces: `supervillain.generator.no_intersection.ConstrainedLinkUpdate(S, interval_n=1)` with `.step(cfg) -> dict`, `.inline_observables(steps) -> {}`, `.report()`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_no_intersection_generators.py`:

```python
def test_constrained_link_update_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.ConstrainedLinkUpdate(V)


def test_constrained_link_update_preserves_validity():
    S = _action()
    gen = supervillain.generator.no_intersection.ConstrainedLinkUpdate(S)
    cfg = _cold(S)
    out = gen.step(cfg)
    assert S.valid(out)
    assert out['n'].shape == cfg['n'].shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_no_intersection_generators.py -k constrained_link -q`
Expected: FAIL — `AttributeError: ... has no attribute 'ConstrainedLinkUpdate'`.

- [ ] **Step 3: Port the class**

Create `supervillain/generator/no_intersection/link.py` by copying the `ConstrainedLinkUpdate` class verbatim from `theoWorm.py` (roughly `theoWorm.py:318`–`429`), with these changes:

1. Module header:

```python
#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)
```

2. Change the `__init__` guard from `isinstance(S, supervillain.action.Villain)` to `isinstance(S, supervillain.action.NoIntersections)` and its message to `'ConstrainedLinkUpdate requires a NoIntersections action.'`

3. In the docstring, replace any bare `theo-worm.md`/`two_worm_move.py` references with `:ref:`the No-Intersection model <no_intersection>``, keeping the physics prose (the coexact-update analogy, the quadratic-constraint discussion) intact.

All methods (`inline_observables`, `step`, `report`) copied unchanged.

- [ ] **Step 4: Export it**

Modify `supervillain/generator/no_intersection/__init__.py`:

```python
from .worm import ThetaWorm
from .link import ConstrainedLinkUpdate
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_no_intersection_generators.py -k constrained_link -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add supervillain/generator/no_intersection/link.py supervillain/generator/no_intersection/__init__.py test/test_no_intersection_generators.py
git commit -m "Add the ConstrainedLinkUpdate generator for the No-Intersection model"
```

---

### Task 4: `WrappingLoopUpdate` generator

**Files:**
- Create: `supervillain/generator/no_intersection/wrapping.py`
- Modify: `supervillain/generator/no_intersection/__init__.py`
- Test: `test/test_no_intersection_generators.py` (append)

**Interfaces:**
- Consumes: `charge` from `.charge`; `supervillain.action.NoIntersections`; `supervillain.lattice.{Form, d}`.
- Produces: `supervillain.generator.no_intersection.WrappingLoopUpdate(S, diagonal=True)` with `.step(configuration) -> dict`, `.inline_observables(steps) -> {}`, `.report()`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_no_intersection_generators.py`:

```python
def test_wrapping_loop_update_requires_no_intersections_action():
    L = Lattice(4, 5)
    V = supervillain.action.Villain(L, kappa=0.3, W=1)
    with pytest.raises(ValueError):
        supervillain.generator.no_intersection.WrappingLoopUpdate(V)


def test_wrapping_loop_update_preserves_validity():
    S = _action()
    gen = supervillain.generator.no_intersection.WrappingLoopUpdate(S)
    cfg = _cold(S)
    for _ in range(5):
        cfg = gen.step(cfg)
        assert S.valid(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_no_intersection_generators.py -k wrapping_loop -q`
Expected: FAIL — `AttributeError: ... has no attribute 'WrappingLoopUpdate'`.

- [ ] **Step 3: Port the class**

Create `supervillain/generator/no_intersection/wrapping.py` by copying the `WrappingLoopUpdate` class verbatim from `theoWorm.py` (roughly `theoWorm.py:432`–end), with these changes:

1. Module header:

```python
#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)
```

2. Change the `__init__` guard from `isinstance(S, supervillain.action.Villain)` to `isinstance(S, supervillain.action.NoIntersections)` and its message to `'WrappingLoopUpdate requires a NoIntersections action.'`

3. In the (long) docstring, replace every bare reference to `two_worm_move.py` and `theo-worm.md` with prose or `:ref:`the No-Intersection model <no_intersection>``. Keep all the physics reasoning (single-direction trick, why-closed-wrapping, why-atomic, detailed-balance) intact — just drop the untracked-filename pointers.

4. Ensure the ported `report()` method is complete: it must end with a closing `f'... )'` and `return`. If the copied fragment is truncated, complete it as:

```python
    def report(self):
        if self.proposed == 0:
            return 'WrappingLoopUpdate: no proposals.'
        return (f'WrappingLoopUpdate: {self.accepted} / {self.proposed} wrapping loops '
                f'accepted ({self.accepted / self.proposed:.6f}); '
                f'{self.clean} / {self.proposed} were constraint-preserving '
                f'({self.clean / self.proposed:.6f}).')
```

All other methods (`inline_observables`, `_propose_loop`, `_delta_S`, `step`) copied unchanged.

- [ ] **Step 4: Export it**

Modify `supervillain/generator/no_intersection/__init__.py`:

```python
from .worm import ThetaWorm
from .link import ConstrainedLinkUpdate
from .wrapping import WrappingLoopUpdate
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_no_intersection_generators.py -k wrapping_loop -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add supervillain/generator/no_intersection/wrapping.py supervillain/generator/no_intersection/__init__.py test/test_no_intersection_generators.py
git commit -m "Add the WrappingLoopUpdate generator for the No-Intersection model"
```

---

### Task 5: `Hammer` combiner + end-to-end smoke

**Files:**
- Modify: `supervillain/generator/no_intersection/__init__.py`
- Test: `test/test_no_intersection_generators.py` (append)

**Interfaces:**
- Consumes: `supervillain.generator.villain.SiteUpdate`; `supervillain.generator.combining.Sequentially`; the three generators above.
- Produces: `supervillain.generator.no_intersection.Hammer(S) -> Sequentially` combining `villain.SiteUpdate(S)`, `ConstrainedLinkUpdate(S)`, `WrappingLoopUpdate(S)`, `ThetaWorm(S)`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_no_intersection_generators.py`:

```python
def test_hammer_steps_stay_valid():
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S)
    cfg = _cold(S)
    for _ in range(5):
        cfg = H.step(cfg)
        assert S.valid(cfg)


def test_ensemble_generate_stays_valid():
    S = _action()
    H = supervillain.generator.no_intersection.Hammer(S)
    e = supervillain.Ensemble(S).generate(10, H, start='cold')
    for c in e.configuration:
        assert S.valid(c)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_no_intersection_generators.py -k "hammer or ensemble_generate" -q`
Expected: FAIL — `AttributeError: ... has no attribute 'Hammer'`.

- [ ] **Step 3: Add the `Hammer`**

Append to `supervillain/generator/no_intersection/__init__.py`:

```python
import supervillain.generator.villain as _villain
import supervillain.generator.combining as _combining


def Hammer(S):
    r'''
    Syntactic sugar for an ergodic :class:`~.Sequentially` combination of the
    No-Intersection generators.  It may change from version to version as new
    generators become available or get improved.

    The :class:`ThetaWorm`, :class:`ConstrainedLinkUpdate`, and
    :class:`WrappingLoopUpdate` all update $n$ only; a
    :class:`~supervillain.generator.villain.SiteUpdate` is included to update
    $\phi$, so the combination is ergodic.

    Parameters
    ----------
    S: a NoIntersections action

    Returns
    -------
    An ergodic generator for updating No-Intersection configurations.
    '''
    return _combining.Sequentially((
        _villain.SiteUpdate(S),
        ConstrainedLinkUpdate(S),
        WrappingLoopUpdate(S),
        ThetaWorm(S),
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_no_intersection_generators.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Run the whole suite to check for regressions**

Run: `uv run pytest test/test_no_intersection.py test/test_no_intersection_generators.py -q`
Expected: PASS. Then a broad sanity check: `uv run pytest -q -k "villain or topological or no_intersection"`.
Expected: PASS (no regressions in neighboring suites).

- [ ] **Step 6: Commit**

```bash
git add supervillain/generator/no_intersection/__init__.py test/test_no_intersection_generators.py
git commit -m "Add the No-Intersection Hammer and end-to-end smoke tests"
```

---

### Task 6: Flesh out `supervillain/no_intersection.rst`

**Files:**
- Modify: `supervillain/no_intersection.rst`

**Interfaces:**
- Consumes: the documented classes `supervillain.action.NoIntersections`, `supervillain.generator.no_intersection.{ThetaWorm, ConstrainedLinkUpdate, WrappingLoopUpdate, Hammer}`.
- Produces: rendered HTML docs under the "Interesting Models" toctree (already wired in `index.rst`).

- [ ] **Step 1: Extend the rst**

Append to `supervillain/no_intersection.rst` (after the existing narrative), keeping the existing `.. _no_intersection:` label and physics prose:

```rst
The Action
==========

.. autoclass:: supervillain.action.NoIntersections
   :members:

The Topological Charge and the Constraint
=========================================

The no-intersection constraint asks that the topological-charge density

.. math::

   Q = dn \wedge dn = d(n \wedge dn)

vanish on every hypercube.  The second equality is exact on the lattice (the
Leibniz rule and $d^2 = 0$ both hold), so $Q$ is the divergence of the 3-form
current $J = n \wedge dn$ and is locally conserved and integer-valued.  A
localized closed $F = dn$ carries zero total charge, so violations of the
constraint always come as a $+1$ / $-1$ dipole --- the fact that the
:class:`~supervillain.generator.no_intersection.ThetaWorm` exploits.

Generators
==========

The No-Intersection generators update $n$ while preserving $Q = 0$; combine any
of them with a $\phi$-update (they are bundled with a
:class:`~supervillain.generator.villain.SiteUpdate` in the :func:`Hammer` below).
They are pure-python reference implementations restricted to $D = 4$.

.. autofunction:: supervillain.generator.no_intersection.Hammer

.. autoclass:: supervillain.generator.no_intersection.ThetaWorm
   :members:

.. autoclass:: supervillain.generator.no_intersection.ConstrainedLinkUpdate
   :members:

.. autoclass:: supervillain.generator.no_intersection.WrappingLoopUpdate
   :members:
```

- [ ] **Step 2: Build the docs**

Run: `make html`
Expected: `build succeeded` with **no new warnings** referencing `no_intersection` (an unrelated pre-existing warning count is acceptable, but there must be no "undefined label", "autodoc import error", or "document isn't included in any toctree" warning naming no_intersection).

- [ ] **Step 3: Verify the classes rendered**

Run: `grep -c -E 'NoIntersections|ThetaWorm|ConstrainedLinkUpdate|WrappingLoopUpdate' _build/html/supervillain/no_intersection.html`
Expected: a count greater than 0 (the autoclasses rendered into the page).

- [ ] **Step 4: Commit**

```bash
git add supervillain/no_intersection.rst
git commit -m "Docs: flesh out the No-Intersection model page"
```

---

## Self-Review notes

- **Spec coverage:** action (Task 1), three generators (Tasks 2–4), `charge` reuse of `_topological_charge` (Task 2), `Hammer` + smoke (Task 5), docs with narrative weaving (Task 6), tests in every task. All spec sections covered.
- **Excluded per spec:** `worm_probe.py`, `two_defect_move.py`, `two_worm_move.py`, and committing `theo*.py`/`theo-worm.md` — none are added by any task.
- **Type consistency:** `charge(n)` returns `(1,)+dims`; the worm indexes `h[1:]` accordingly and stores `Theta_Theta` of shape `dims`. Generators consistently guard on `NoIntersections`. `Hammer` returns a `Sequentially` as consumed by `Ensemble.generate`.
- **Known risk (carried from prototype):** the worm/wrapping updates are not guaranteed ergodic on frozen backgrounds; documented in the docstrings, not fixed here. From a cold start (the tested path) they keep `Q = 0` throughout.
