# The No-Intersection Model — design

**Date:** 2026-07-01
**Status:** approved (pending spec review)

## Goal

Promote the prototype `TheoAction` (in the untracked `theo.py`) and the three
polished worm/link/wrapping generators (in the untracked `theoWorm.py`) into
first-class library citizens as the **No-Intersection Model**, and flesh out
`supervillain/no_intersection.rst` to document them.

The model is the standard modified-Villain action with an extra constraint that
suppresses vortex self-intersection:

```
S = (κ/2) (dφ − 2π n)²        with the constraint   Q = dn ∧ dn = 0   on every hypercube.
```

θ is a Lagrange-multiplier top-form (`S = S_Villain + i θ (dn ∧ dn)`); it is
never sampled as a dynamical field, so the field content is `{φ, n}`, exactly as
in Villain. The physics is only interesting in **D = 4** (mixed AVV anomaly,
possible back-door to 4D bosonization), so the implementation hard-assumes
D = 4.

## Non-goals

- No θ dynamical field, no new sampler for θ.
- The exploratory research scripts (`worm_probe.py`, `two_defect_move.py`,
  `two_worm_move.py`) are **not** incorporated — they are analysis/exploration
  tools with `main()` entry points, not library generators.
- `theo-worm.md` is **not** committed as a standalone file; its derivation is
  woven narratively into the docstrings and the `.rst`.

## Components

### 1. Action — `supervillain/action/no_intersection.py`

`class NoIntersections(Villain)` subclasses `Villain` to reuse all tested
machinery.

- `__init__(self, lattice, kappa)`:
  - Raise `ValueError` if `lattice.D != 4` (the constraint `dn ∧ dn` is a
    4-form; the model is only meaningful/exciting at D = 4 and fixing D = 4
    simplifies development).
  - Call `super().__init__(lattice, kappa, W=1)`.
- `__str__` → `NoIntersections({lattice}, κ={kappa})`.
- Override `valid(configuration)` → `np.isclose(wedge(d(n), d(n)), 0).all()`.
- **Inherited unchanged:** `__call__`, `links`, `local`, `configurations`,
  `gauge_transform`. Field content is `{φ, n}` (real 0-form, integer 1-form).
- Export: add `from .no_intersection import NoIntersections` to
  `supervillain/action/__init__.py`.

The gauge transformation `φ → φ + 2πk`, `n → n + dk` is inherited; it preserves
`dn` (since `d(dk) = 0`) and hence preserves both the energy and the
no-intersection constraint, so gauge orbits stay within the valid set.

### 2. Generators — `supervillain/generator/no_intersection/`

A new subpackage parallel to `generator/villain/` and `generator/worldline/`,
one class per file, ported verbatim (modulo the guard tightening below and
docstring weaving) from `theoWorm.py`:

- `worm.py` → `ThetaWorm` — Prokof'ev–Svistunov worm for the `Q = dn ∧ dn = 0`
  constraint. Samples the enlarged space where the constraint is violated at
  exactly two hypercubes (head +1 / tail −1), dragging a sheet of `F = dn`
  between them via per-step verified 3-link moves drawn from the axis-permutation
  orbit of one seed move. Updates `n` only. Emits inline observables
  `Theta_Theta` (head−tail displacement histogram, shape `L.dims`, giving
  `⟨e^{iθ_h} e^{−iθ_t}⟩`) and `Worm_Length`. Carries the `charge(n) = dn ∧ dn`
  helper and the `_SEED` / `_build_library` move machinery.
- `link.py` → `ConstrainedLinkUpdate` — local fluctuation of `F = dn` that
  preserves the constraint.
- `wrapping.py` → `WrappingLoopUpdate` — updates the torus-wrapping modes.
- `__init__.py`:
  - `from .worm import ThetaWorm`
  - `from .link import ConstrainedLinkUpdate`
  - `from .wrapping import WrappingLoopUpdate`
  - `def Hammer(S, worms=1)` — an ergodic `combining.Sequentially` of a
    `villain.SiteUpdate(S)` (to update φ, which the worm leaves untouched),
    `ConstrainedLinkUpdate(S)`, `WrappingLoopUpdate(S)`, and `ThetaWorm(S)`.
    Mirrors the docstring/structure of `villain.Hammer`.
- Register `import supervillain.generator.no_intersection` in
  `supervillain/generator/__init__.py`.
- **Guard tightening:** each generator's constructor check changes from
  `isinstance(S, supervillain.action.Villain)` to
  `isinstance(S, supervillain.action.NoIntersections)`, since they assume the
  no-intersection constraint. (Still an `isinstance` pass, as `NoIntersections`
  subclasses `Villain`.) Each also keeps its existing `D == 4` guard.

### 3. Docs — `supervillain/no_intersection.rst`

Keep the existing physics narrative (Jacobson's observation, the two U(1)s and
their mixed AVV anomaly, the transition/CFT speculation). Add:

- `.. autoclass:: supervillain.action.NoIntersections` with members.
- A short narrative subsection on the constraint and the exactly-conserved
  integer topological charge `Q = dn ∧ dn = d(n ∧ dn)` (Leibniz + d² = 0 on the
  lattice), so charge comes in ±1 dipoles — motivating the worm.
- `.. autoclass::` for `ThetaWorm`, `ConstrainedLinkUpdate`,
  `WrappingLoopUpdate`, with the `theo-worm.md` derivation woven into their
  docstrings and/or the surrounding `.rst` prose (no separate `.md` committed).

Already wired into the "Interesting Models" toctree in `index.rst`.

### 4. Tests — `test/test_no_intersection.py`

- **Constructor guard:** `NoIntersections` raises for `D != 4`; constructs for
  D = 4.
- **`valid()`:** accepts a hand-built non-intersecting `n`; rejects a hand-built
  intersecting `n` (two vortex sheets crossing so `dn ∧ dn ≠ 0`).
- **Gauge invariance:** a valid config stays valid under `gauge_transform`, and
  the action value is unchanged.
- **Smoke (from `theo.py`):** D = 4, N = 5, run `no_intersection.Hammer(S)` for a
  handful of steps from a cold start; assert every emitted configuration
  satisfies `S.valid(...)`.
- **Worm closure:** `ThetaWorm(S).step(cfg)` on a valid config returns a valid
  config and populates `Theta_Theta` (shape `L.dims`) and `Worm_Length`.

## Risks / notes

- The worm's move library is the orbit of a single seed shape; stalled proposals
  are rejected (detailed-balance safe) but the worm is not guaranteed ergodic on
  every background. This is a known limitation carried over from the prototype,
  documented in the docstring, not fixed here.
- `ThetaWorm` and friends are pure-python reference implementations (no numba),
  D = 4 only.
