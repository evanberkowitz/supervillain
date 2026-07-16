# Parallel tempering across a κ ladder

**Date:** 2026-07-15
**Status:** approved design, pre-implementation

## Problem

Below the NoIntersections spin-U(1) transition (κ ≲ 0.04) single-start chains land
in a metastable high-action basin about half the time (basin censoring, established
2026-07-14 in the no-intersections notebook).  κ-annealing down the healthy branch
cures a single point but is one-way and costs an extra production run per point.
Parallel tempering across a ladder of κs keeps every rung in equilibrium with
continuous re-annealing: healthy configurations generated at large κ diffuse down
the ladder, and trapped rungs escape by diffusing up.

The library has no multi-chain machinery: `Ensemble.generate` drives one generator
on one action.  This design adds tempering without touching the `Generator` API,
the actions, or any observable.

## Physics the design exploits

The Villain-family actions are **exactly linear in κ**:
$S_\kappa(x) = \kappa E(x)$ with $E(x) = \frac{1}{2}\sum_\ell (d\phi - 2\pi n)^2_\ell$,
and the NoIntersections constraint ($dn \cup dn = 0$) is κ-independent.  Therefore:

- A configuration valid at one rung is valid at every rung; swaps can never
  violate the constraint.
- The swap of configurations between rungs $i$ and $j$ accepts with
  $\min(1, e^{-\Delta S})$, $\Delta S = (\kappa_i - \kappa_j)(E_j - E_i)$:
  **one scalar per rung** decides, evaluated by each rung on its *own* current
  configuration.  Configurations move only on acceptance.  (This is what makes a
  future distributed transport cheap: the wire carries one float per rung per
  attempt, full fields only on acceptance.)

## Architecture decision

Three architectures were weighed in the 2026-07-15 session:

- **Joint-chain single Ensemble** (product action, rung axis): rejected — field
  names are hard-coded into Actions/Observables, the rung axis is invasive, and
  memory multiplies by the ladder size in one allocation.
- **Communicating `TemperingProposal` generators** inside each rung's
  `Sequentially` stack, one execution context per rung: the right *distributed*
  realization, but it cannot run serially (rung 0's generate loop would block on
  rung 1's energy before rung 1 ever runs) and brings barrier/abort machinery.
- **Serial orchestrator**: a driver that owns all rungs and interleaves them in
  lockstep.  A tempered ladder costs about the same total work as the serial
  κ-loop campaigns it replaces, so serial meets the science need now.

**v1 is the serial orchestrator**, with the swap kernel, pair schedule, and RNG
discipline specified transport-independently (the *seam*) so the distributed
realization added later executes the *identical* composite Markov kernel with
*identical* random numbers and can be certified bitwise against the serial ladder.

## Design

| Object | Role | Plane |
|---|---|---|
| `ParallelTempering` | serial orchestrator: lockstep local sweeps + swap sweeps, emits per-rung `Ensemble`s | control |
| swap kernel (module function) | pure accept/reject from $(\Delta\kappa, \Delta E, u)$ | data (the seam) |
| `EvenOddPairs` | deterministic pair schedule as a function of the sweep index | data (the seam) |
| `TemperedRung` | generator-shaped marker stored on each rung's ensemble; delegates bookkeeping, refuses to `step` | guard |
| `ParallelTemperingTuner` | ladder *geometry* recommendations from a pilot leg | control |

### 1. The composite kernel (transport-independent)

One `ParallelTempering` step, emitting configuration $t$ on every rung:

1. **Local sweep**: every rung applies its own local generator stack once,
   `cfg[i] = G[i].step(cfg[i])`, exactly as `Ensemble.generate` would.
2. **Swap sweep**: each rung evaluates $E_i = S_i(x_i)/\kappa_i$ on its own,
   post-local-sweep configuration.  For each pair $(i, i+1)$ in
   `EvenOddPairs(t)` — pairs $(0,1),(2,3),\dots$ on even $t$, $(1,2),(3,4),\dots$
   on odd $t$ — accept the exchange of the two configurations with probability
   $\min(1, e^{-(\kappa_i-\kappa_j)(E_j-E_i)})$, drawing one uniform from the
   pair's dedicated stream.  On acceptance the configuration dictionaries (and
   their cached $E$s) are exchanged wholesale.
3. **Emit**: rung $i$ stores `cfg[i]` as its configuration $t$.

Each factor (local update, single-pair swap) satisfies detailed balance for the
product distribution $\prod_i e^{-S_i(x_i)}$, so any fixed composition preserves
it; the composition above is *pinned* so every backend executes the same one.

Swapping **configurations, not κ labels**, is a hard decision, not a preference:
each rung's Ensemble stays at fixed κ, so the entire existing analysis stack
(h5 layout, `Bootstrap`, observables, campaign drivers) applies per rung
untouched.  κ-label swapping is rejected outright — it breaks the fixed-κ
assumption everywhere downstream.

### 2. RNG discipline (the seam, part 1)

`ParallelTempering(..., seed=...)` builds a `numpy.random.SeedSequence(seed)` and
spawns **one child stream per adjacent pair**: pair $(i, i+1)$ uses child $i$.
Every attempt on a pair draws exactly one uniform from that pair's stream, in
sweep order.  A distributed backend reproduces the identical decisions because
each pair's attempt count is a deterministic function of the sweep index alone
(via `EvenOddPairs`) — no cross-pair draw ordering exists to violate.

Local generators keep their own rngs, seeded (or not) by the caller as today;
the ladder machinery neither owns nor perturbs them.  Bitwise reproducibility of
a full run therefore has two independent ingredients: the ladder seed (this
design) and seedable local stacks (already true of `DefectGas`; `SiteUpdate`
currently self-seeds, a pre-existing limitation this design does not address).

### 3. The swap kernel (the seam, part 2)

A module-level pure function, individually testable and shared verbatim by every
backend:

```python
def swap_accepted(dkappa, dE, uniform):
    # dS = dkappa * dE; accept with probability min(1, exp(-dS)).
    dS = dkappa * dE
    return (dS <= 0) or (uniform < np.exp(-dS))
```

with `dkappa = kappa_i - kappa_j`, `dE = E_j - E_i`, and $E$ evaluated as
`S(**cfg) / S.kappa` (one lattice reduction per rung per sweep; `Villain.__call__`
already swallows the configuration dictionary's non-field keys).  The arithmetic
order is part of the contract: backends must compute `dS` the same way or bitwise
certification fails.

### 4. `ParallelTempering`

New module `supervillain/tempering.py` (peer of `ensemble.py`), exported from
`supervillain/__init__.py`.

```python
pt = ParallelTempering(actions, generators, seed=...)
ensembles = pt.generate(steps, start='cold', progress=tqdm)   # list of Ensembles
```

- `actions`: ordered by κ (ascending; validated).  All must share a lattice and
  field content; each must expose `.kappa` and be κ-linear (documented
  requirement, satisfied by the Villain family).
- `generators`: one local stack per rung, built by the caller exactly as today
  (e.g. `Sequentially((SiteUpdate, ExactUpdate, CohomologyUpdate, DefectGas))`).
  Local generators need zero changes and stay ignorant of tempering.
- `.generate(steps, start='cold', progress=_no_op, starting_index=0,
  index_stride=1)` mirrors `Ensemble.generate`'s semantics rung by rung
  (configuration allocation, inline-observable merge, index bookkeeping,
  `Timer` logging, post-run `report()` logging).  `start` is `'cold'` or a
  **list of per-rung configuration dictionaries** — the ladder analogue of
  `Ensemble.generate`'s `start`, and the enabler of multi-leg driving.
  Returns per-rung `Ensemble`s that are standard in every respect.
- `.continue_from(ensembles, steps, seed=...)` (classmethod): accepts the list
  of per-rung ensembles (or h5 groups), unwraps their `TemperedRung` markers,
  and resumes every rung from its last configuration with continued indexing
  and the correct schedule parity.  A fresh ladder seed is acceptable (chain
  correctness never depends on continuing an rng stream; only single-run
  bitwise reproducibility does, and that is what `seed` is for).
- Diagnostics: per-pair attempted/accepted tallies, `pair_acceptance`,
  an online per-replica round-trip counter, and `report()`.

### 5. Replica tracking

The orchestrator declares one ladder inline observable, **`Replica`**
(`Batch(steps, shape=(), dtype=int)`), initialized to the rung index at a cold
start (or read from the start configurations when continuing).  The label lives
*inside the configuration dictionary*, so a swap transports it automatically and
each rung's stored `Replica` trace records which walker occupied it at every
step — replica-flow and round-trip diagnostics then need no extra machinery.
Like `Ticks`/`VacuumTicks`, `Replica` gets **no `Observable` class** and no docs
autoclass entry: it is ladder bookkeeping, not physics.

### 6. Tempered-rung marking

A tempered rung is **not a Markov chain by itself**: `Ensemble.continue_from` on
a single rung would silently drop the tempering and produce wrong physics.  Each
emitted ensemble therefore stores `e.generator = TemperedRung(local_stack,
rung=i, ladder=kappas)` — a `ReadWriteable` generator-shaped marker that
delegates `report()`/`inline_observables()` to the wrapped stack but whose
`step()` raises `RuntimeError` pointing at `ParallelTempering.continue_from`.
The marker's attributes double as the h5 metadata that identifies a tempered
ensemble.  (Campaign drivers that strip generators before writing — the issue
#65 workaround — lose the marker along with everything else; they already record
provenance in file attributes and that policy is theirs.)

### 7. Lifecycle stays out of the infrastructure

`ParallelTempering` is a pure composite-kernel driver: **no tuning, no
thermalization, no hooks**.  The therm → tune → produce recipe is driver-script
choreography, exactly as in `campaign-2026-07-08/campaign.py` today, expressed
as tempering *legs*:

1. **Thermalization leg**: conservative fixed-ζ stacks, cold start; the ladder
   itself replaces κ-annealing (healthy high-κ configurations diffuse down).
2. **Per-rung tuning between legs** (driver code): `DefectGasFugacityTuner` on
   each rung's last configuration — valid by construction, since every gas
   emission is a vacuum tick.
3. **Production leg**: a new `ParallelTempering` with the tuned stacks,
   `start=[e.configuration[-1] for e in therm_ensembles]`.

Legality: swap acceptance depends only on the *actions* (κ, E), never on
proposal parameters, so per-rung ζs may differ freely and change between legs
without touching detailed balance.

### 8. `ParallelTemperingTuner` (ladder geometry only)

The other tuning problem — *where the rungs sit* — is tempering-specific and
action-agnostic: acceptance between neighbors depends only on
$\Delta\kappa$ and the E-distributions.  From a short pilot leg the tuner reads
per-pair acceptances and per-rung $E$ statistics (mean and standard deviation of
`S(**cfg)/kappa` over the pilot ensembles) and **recommends** a respaced ladder:

- Estimate the thermodynamic length $\lambda(\kappa) = \int \sigma_E \,d\kappa$
  by trapezoid over the pilot rungs.
- Place rungs (endpoints fixed) at equal increments of $\lambda$; choose the
  rung count so the predicted per-pair acceptance meets a target (~25% by
  default, uniform by construction).

`tuner.ladder(target=0.25)` returns a κ list; the driver rebuilds actions
through a caller-supplied factory (`lambda kappa: NoIntersections(L, kappa)`) —
the tuner never constructs or inspects actions.  v1 recommends from one pilot;
iterate-until-uniform loops are manual.  Proposal tuning (ζ) explicitly remains
with the per-action tuners in the driver.

### 9. Docs

`supervillain/tempering.rst` with autoclass entries for `ParallelTempering` and
`ParallelTemperingTuner`, added to the toctree beside `ensemble.rst`.  House
docstring style: notes/warnings/seealso before Parameters; raw `r'''` strings
around LaTeX, never f-strings.

## Verification

Targeted only (the NoIntersections suite is slow; no bare full-suite runs):

- **Swap-kernel unit tests**: acceptance probability against direct evaluation,
  the $\Delta\kappa \to 0$ limit (always accept), sign conventions (a hotter
  configuration always moves up in κ with probability 1 when $\Delta S \le 0$).
- **Degenerate ladder** (all κ equal): every swap accepts; each rung's
  observable marginals match a single un-tempered chain at the same κ.
- **Exactness**: a 2–3 rung ladder on a tiny lattice vs independent single-κ
  ensembles — matching distributions (ActionDensity, WindingSquared) within
  bootstrap errors.
- **Determinism**: same ladder seed and seedable local stacks (test-local dummy
  generator) → bitwise-identical ensembles, twice.
- **Replica bookkeeping**: `Replica` traces are permutations at every step;
  round-trip counter agrees with a post-hoc recount from the traces.

First physics happens in the no-intersections notebook
(`tempering-2026-07-15/`): an N = 6 ladder spanning κ ∈ [0.02, 0.1] via the
two-leg recipe, validated against the 2026-07-14 κ-annealed healthy-basin anchor.

## Non-goals

- No MPI / multiprocessing / thread backend in v1.  The seam (kernel, schedule,
  per-pair streams, E-based ΔS) is the contract such a backend implements; it is
  certified bitwise against the serial ladder when built.
- No κ-label swapping — rejected outright, not deferred.
- No tuning or thermalization hooks inside `ParallelTempering` — lifecycle is
  permanently driver-script territory.
- No automatic ladder-adaptation loop (v1 tuner recommends from one pilot).
- No change to `Generator`, any action, any observable, or `Ensemble` itself.

## Known limitation

All rungs' configuration arrays live in one process: memory is
n_rungs × a single campaign point.  Acceptable for the near-term N ≤ 12 low-κ
physics; the distributed backend is the eventual answer at the N = 16 tier.
