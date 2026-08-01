#!/usr/bin/env python

r"""The Surface Worm Gas (SWG): a grand-canonical extended-ensemble sampler of
the Jacobson No Intersections model that walks the space of integer 2-forms
$F$ rather than the constraint-satisfying 1-forms $n$ the rest of the
:mod:`~supervillain.generator.no_intersection` package moves.  Where
:class:`~.defect_gas.DefectGas` transports *defects* at fixed $dn = 0$, the
SWG **relaxes and prices both** constraints the physical theory imposes ---
closedness and self-intersection-freedom --- and only emits where they are
jointly satisfied again.  It is not a replacement for the roster
:func:`~.Hammer` assembles; it is a second, independent route to the same
measurement, useful precisely where a defect gas transported at fixed $dn$
cannot go: through topologically nontrivial intermediate states that change
which torus-wrapping sector the configuration sits in.

The extended ensemble
----------------------

The sampler's stationary distribution is

.. math ::

    \pi_\text{ext}(F) \propto e^{-2\pi^2\kappa C(F)}
        \cdot w\big(D(F)\big) \cdot \eta_q^{Q(F)}
        \cdot Z_\text{wind}(F) \cdot w_2\big(r^2(F)\big)

with $C(F)$ the coexact ($\phi$-marginalized) norm, $D(F) = \#\{\text{cubes
with } dF \neq 0\}$ the open-surface (closedness) defect count priced by the
open-surface sector table $w(D)$ (:class:`~.weights.SectorWeights`), $Q(F) =
\#\{\text{hypercubes with } q = F\wedge F \neq 0\}$ the self-intersection
defect count priced by the fugacity $\eta_q = \texttt{intersectionFugacity}$,
$Z_\text{wind}(F)$ the winding partition function summed over the torus
coset (below), and $w_2(r^2)$ an optional umbrella on the separation of a
$\pm1$ charge pair (:class:`~.weights.PairUmbrella`) that keeps the chain
from getting trapped at large separation.  Two move types keep this
distribution manifestly reachable: a plaquette $\pm1$ toggle (uniform, or
--- with probability ``targetFraction`` --- targeted at a plaquette incident
on an open cell, corrected by an explicit Hastings ratio) and an exact
coboundary heatbath: a **local** move, the 6-plaquette coboundary of one unit
$\mu$-link translated to a randomly drawn anchor $y$, shifted by an integer
$\Delta$ resampled from its exact conditional. Both are Metropolis- or
heatbath-correct against $\pi_\text{ext}$
by construction, so **detailed balance is manifest** --- there is no
adjacency bookkeeping to get wrong, and no orphaned defect a move might
strand.

The legal vacuum and why closed is not exact
----------------------------------------------

A configuration is physical --- corresponds to some genuine Villain $n$ ---
only on the **joint vacuum** $D = 0 \wedge Q = 0 \wedge \texttt{periods} =
0$ (:attr:`~.state.FState.legal_vacuum`). The first two conditions are the
familiar ones: no open surface, no self-intersection. The third is easy to
miss and is what makes this sampler's book-keeping non-negotiable: **closed
is not exact on $T^4$.** $D = 0$ only says $F$ is closed, $dF = 0$; it says
nothing about whether $F$ is also *exact*, $F = dn$ for some integer 1-form
$n$. A worm that recloses around one of $T^4$'s non-contractible 2-cycles
can return to $D = 0$ while leaving a nonzero de Rham class in $H^2(T^4)$
behind --- a nonzero component total (:attr:`~.state.FState.periods`) that
no integer primitive can undo. For such an $F$, no $n$ with $dn = F$ exists
anywhere on the lattice: it is a "vacuum-shaped" state that represents
nothing physical. The periods gate is what turns "closed" into "exact" and
is exactly what makes emission below meaningful rather than merely
plausible.

Emission
--------

:meth:`~.gas.SurfaceWormGas.emit` only ever fires on a legal vacuum, where
$\pi_\text{ext}$ reduces to the true $F$-marginal $\pi(F) \propto
e^{-2\pi^2\kappa C(F)}\cdot Z_\text{wind}(F)$. From there it builds an
integer primitive $n$ of $F$ via the staircase construction
(:mod:`~.staircase`), **resamples the torus-winding coset** $M \mid F$ from
its exact conditional $\pi(M \mid F) \propto e^{-2\pi^2\kappa\|M\|^2/V}$
restricted to $M \in M_0(F) + N^3\mathbb Z^4$ (the staircase primitive only
ever returns one representative $M_0(F)$ of that coset; skipping the
resample would silently pin every emission to it), and draws $\varphi$
exactly from its Gaussian conditional (:mod:`~.reconstruct`). Because the
winding tilt $Z_\text{wind}$ is **unconditionally part of the chain's
stationary distribution** --- there is no "off" switch, unlike the audited
reference this subpackage ports --- every emitted row is already a physical,
unbiased draw. No importance weight is attached to an emission, and none
exists to omit: there is nothing left to double-count.

.. warning ::

    At large ``openSurfaceFugacity`` (equivalently, a $w(D)$ that favors
    large $D$ too generously) the *class direction* --- which torus-wrapping
    sector $F$ sits in --- is unpriced off the joint-vacuum shell: nothing in
    $\pi_\text{ext}$ penalizes a spanning, system-filling open-surface
    network over a handful of isolated open-surface bubbles at the same $D$.
    A chain that wanders onto the spanning-network branch can spend a very
    long time away from $D = 0$, so the exactness-gated vacuum starves ---
    the runtime symptom is :meth:`~.gas.SurfaceWormGas.step`'s repeated
    ``maxWaitTicks`` "slow emit" warning as it keeps sweeping in search of a
    legal vacuum that a poorly tuned table makes rare. This is not a bug to
    patch in the sampler; it is why :class:`~.tuners.SectorWeightTuner` and
    :class:`~.tuners.PairUmbrellaTuner` exist. Production regimes use their
    tuned tables, which keep the closed shell populated by flattening the
    visit histogram across $D$ (and, with an umbrella, across pair
    separation) rather than leaving it to a bare fugacity.

What the accumulator measures
------------------------------

:class:`~.accumulator.CorrelatorAccumulator` accumulates the intersection
correlator as a **strided dwell-ratio**: $\Theta_{\Delta x} =
\langle\texttt{Theta\_Theta}\rangle / \langle\texttt{VacuumTicks}\rangle$,
both tallied only on the closed shell $D = 0$ and absolutely normalized
against the known sector prices, so no origin bin is needed the way a worm
histogram would need one. Ticks where $F$ is closed but carries a nonzero
$H^2$ class --- closed but not exact, exactly the failure mode described
above --- are excluded from every closed-shell dwell (they represent no
physical configuration) and are instead counted separately as
``NontrivialClassTicks``, so a parameter regime where the chain has started
wandering in class is visible in the harvest rather than silently
contaminating the correlator.

Exports
-------

The public API, re-exported from
:mod:`supervillain.generator.no_intersection`: :class:`~.gas.SurfaceWormGas`,
:class:`~.tuners.SectorWeightTuner`, :class:`~.tuners.PairUmbrellaTuner`,
:class:`~.tuners.TransportTuner`.
Also importable from this subpackage directly, for power users who want the
pieces rather than the assembled sampler:
:class:`~.weights.SectorWeights`, :class:`~.weights.PairUmbrella`,
:class:`~.state.FState`, :class:`~.accumulator.CorrelatorAccumulator`.
"""

from .gas import SurfaceWormGas
from .weights import SectorWeights, PairUmbrella
from .state import FState
from .accumulator import CorrelatorAccumulator
from .tuners import SectorWeightTuner, PairUmbrellaTuner, TransportTuner
