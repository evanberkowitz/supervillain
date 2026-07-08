#!/usr/bin/env python

import numpy as np

from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.villain.site import SiteUpdate
from supervillain.generator.no_intersection.charge import charge
from supervillain.generator.no_intersection import local_charge
import supervillain.action


class FugacityWorm(ReadWriteable):
    r"""
    Grand-canonical defect sampler for the $q = dn \wedge dn = 0$ constraint in 4D,
    with the inline estimator of the Lagrange-multiplier correlator
    $\left\langle e^{+i\theta_{x}} e^{-i\theta_{y}} \right\rangle$.

    Where the clean-set worms (:class:`~.AdaptiveIntersectionWorm` and friends) walk a
    defect pair through the *valid* configurations and demand every step be an exact
    unit-dipole transport --- steps that often do not exist at all on the dense sheets of
    small $\kappa$ --- this sampler stops demanding cleanliness and instead *prices the
    mess*.  It samples the enlarged ensemble

    .. math ::
        \pi(\phi, n) = e^{-S_{V}(\phi, n)}\, \zeta^{D(n)},
        \qquad D(n) = \sum_{x} \left|Q_{x}(n)\right|,
        \quad Q = dn \wedge dn,

    with single-link Metropolis: a link and $c = \pm 1$ are drawn uniformly (a
    symmetric proposal) and accepted with $\min\!\left(1, e^{-\Delta S}\,
    \zeta^{\Delta D}\right)$.  Every link is always proposable --- a messy $\Delta q$ is
    a *legal* state with more defects, exponentially discounted by the fugacity
    $\zeta < 1$, and defect-annihilating moves are correspondingly rewarded.  The
    higher-defect sectors are the corridors through jammed backgrounds that clean worms
    lack, so the sampler cannot jam; single-link moves connect every $n$ with nonzero
    acceptance, making ergodicity on the enlarged space manifest.  One
    :class:`~supervillain.generator.villain.SiteUpdate` sweep of $\phi$ accompanies each
    link sweep.

    The correlator is read off by bookkeeping rather than steering.  Since inserting
    $e^{+i\theta_{x}} e^{-i\theta_{y}}$ shifts the constraint to
    $Q = \delta_{x} - \delta_{y}$,

    .. math ::
        \left\langle e^{+i\theta_{x}} e^{-i\theta_{y}} \right\rangle
        = \frac{E\!\left[\mathbf{1}_{Q = \delta_{x} - \delta_{y}}\right]}
               {\zeta^{2}\, E\!\left[\mathbf{1}_{Q \equiv 0}\right]}
        \qquad\Longrightarrow\qquad
        G(r) = \frac{H_{\text{pair}}(r)}{V\, \zeta^{2}\, H_{Z}},

    where $H_{Z}$ counts the Monte-Carlo clock ticks spent in the vacuum sector,
    $H_{\text{pair}}(r)$ those spent in the exact single-pair sector at head$-$tail
    displacement $r$, and $V = N^{4}$ is the translation multiplicity.  Two properties
    follow immediately and are worth internalizing:

    * $G(0) = 1$ **identically** --- a coincident pair *is* the vacuum --- so $G$ is
      **absolutely normalized**, directly comparable to
      :class:`~.Intersection_Intersection` normalized at the origin, with no origin
      division and none of the pivot bookkeeping of the walking worms.
    * $G$ is **independent of** $\zeta$ --- the fugacity price the sampler charged the
      pair sector is divided back out --- so $\zeta$ tunes only the variance.  Running
      at two values of $\zeta$ and comparing is a sharp end-to-end exactness test.

    Tuning: entropy pushes $D$ upward (each defect may live anywhere, and the denser
    the sheet the larger a single link's $\left|\Delta D\right|$), so the right $\zeta$
    shrinks with volume and with $1/\kappa$.  Symptoms of a bad choice are loud: too
    large and $D$ pins at ``D_max`` with the measured sectors never visited; too small
    and pair excursions become needlessly rare.  :meth:`tune` automates the choice.
    ``D_max`` truncates the state space (proposals beyond it are ordinary zero-weight
    rejections); it exists to keep a badly-tuned chain out of the defect condensate,
    not for correctness.

    Statistical errors on $G$ come from a block jackknife over :meth:`close_block`
    boundaries; note $\left\langle e^{i\theta} \right\rangle$ itself vanishes
    identically on the torus (total charge is conserved), so the large-$r$ plateau of
    $G$ is the only order-parameter diagnostic for the $\theta$ shift symmetry.

    Parameters
    ----------
    S: a NoIntersections action
        Supplies $S_{V}$, $\kappa$, and the lattice.
    zeta: float
        The per-defect fugacity $\zeta \in (0, 1]$.
    D_max: int or None
        Hard cap on $D$; ``None`` uncaps.
    rng: numpy Generator, optional

    .. warning ::

        Restricted to $D = 4$.  This is **not** an
        :class:`~supervillain.generator.Generator` for
        :class:`~supervillain.Ensemble`: it deliberately visits invalid ($Q \neq 0$)
        configurations, so its samples must not feed observables that assume the
        constraint.  Valid-sector physics is available by conditioning on the vacuum
        sector; the intended use is :meth:`run` + :meth:`correlator`.

    .. seealso ::

        ``example/no-intersection/fugacity_worm.py`` for the command-line driver and
        ``dev/fugacity-worm.md`` / ``dev/theo-worm.md`` for the design notes this
        implements.
    """

    def __init__(self, S, zeta, D_max=None, rng=None):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('FugacityWorm requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('FugacityWorm is only implemented for D = 4.')
        if not (0 < zeta <= 1):
            raise ValueError(f'zeta must be in (0, 1]; got {zeta}.')

        self.S = S
        self.L = S.Lattice
        self.N = self.L.N
        self.kappa = S.kappa
        self.zeta = float(zeta)
        self.D_max = D_max
        self.rng = rng if rng is not None else np.random.default_rng()
        # Tallies (blocked for jackknife errors).
        self.blocks = []            # (H_pair array, H_Z scalar) per block
        self._new_block()
        self.D_trace = []           # per-sweep defect count (diagnostic)
        self.accepted = 0
        self.proposed = 0

    def __str__(self):
        return f'FugacityWorm(zeta={self.zeta}, D_max={self.D_max})'

    def _new_block(self):
        self._H_pair = np.zeros(self.L.dims)
        self._H_Z = 0

    def close_block(self):
        r"""End the current jackknife block and start a new one."""
        self.blocks.append((self._H_pair, self._H_Z))
        self._new_block()

    @classmethod
    def tune(cls, S, D_max=None, rng=None, phi=None, n=None,
             ladder=(0.1, 0.05, 0.02, 0.01, 0.005, 0.002),
             sweeps=60, target=0.15):
        r"""
        Pick $\zeta$ by short probes down a ladder, keeping the first value whose
        vacuum dwell exceeds ``target`` (the pair sector follows, being the vacuum's
        nearest excursion).  Because the estimator is $\zeta$-independent, tuning
        affects only the variance, never the answer.

        Parameters
        ----------
        S: a NoIntersections action
        phi, n: arrays, optional
            A thermalized starting configuration (cold if omitted).

        Returns
        -------
        float
            The chosen $\zeta$.
        """
        L = S.Lattice
        if phi is None:
            phi = np.zeros(L.dims)
        if n is None:
            n = np.zeros((L.D,) + L.dims, dtype=np.int64)
        zeta = ladder[-1]
        for z in ladder:
            probe = cls(S, zeta=z, D_max=D_max,
                        rng=rng if rng is not None else np.random.default_rng())
            probe.run(phi, n, sweeps, tally=False)
            probe.blocks = []
            probe._new_block()
            probe.run(phi, n, sweeps, tally=True)
            probe.close_block()
            vac = probe.blocks[0][1] / max(1, probe.proposed / 2)
            zeta = z
            if vac > target:
                break
        return zeta

    def run(self, phi, n, sweeps, tally=True, progress=None):
        r"""
        Evolve ``sweeps`` sweeps --- each $4 N^{4}$ single-link proposals plus one
        $\phi$ sweep --- from ``(phi, n)``, tallying the sector dwell after every
        proposal unless ``tally`` is ``False`` (thermalization).

        Parameters
        ----------
        phi: array
            Site angles (any real values).
        n: integer array
            Link field; need **not** satisfy the constraint.
        sweeps: int
        tally: bool
        progress: callable, optional
            e.g. ``tqdm``.

        Returns
        -------
        (phi, n)
            The evolved configuration, suitable for chaining calls.
        """
        L, N, rng, zeta = self.L, self.N, self.rng, self.zeta
        n = np.asarray(n).astype(np.int64).copy()
        phi = np.asarray(phi).astype(float).copy()
        # The whole state the moves need, maintained incrementally so a proposal costs
        # O(1): the field strength F = dn (the per-link charge stencils read it), the
        # charge Q = dn∧dn only through its NONZERO cells (the sparse `defects` dict --
        # on a well-tuned chain almost all of Q is zero), and the scalar defect count
        # D = Σ|Q| that the fugacity prices.
        F = np.asarray(d(Form(n, degree=1, lattice=L))).astype(np.int64)
        Q = np.asarray(charge(Form(n, degree=1, lattice=L))).astype(np.int64)
        # defects: nonzero hypercubes, keyed by the 4-tuple cell (component axis stripped).
        defects = {tuple(int(x) for x in z[1:]): int(Q[tuple(z)])
                   for z in np.argwhere(Q != 0)}
        D = int(np.abs(Q).sum())
        site_update = SiteUpdate(self.S)
        site_update.rng = rng
        dphi = np.asarray(d(Form(phi, degree=0, lattice=L)))
        n_links = 4 * N**4
        halfk = self.kappa / 2
        twopi = 2 * np.pi

        iterator = range(sweeps)
        if progress is not None:
            iterator = progress(iterator)
        for _ in iterator:
            # Draw a sweep's worth of proposals up front (numpy batching); the proposal
            # distribution -- uniform link, uniform c = ±1 -- is SYMMETRIC, so plain
            # Metropolis needs no Hastings factor.  This is the whole move set: no
            # templates, no clean sets, no directions.
            mus = rng.integers(0, 4, size=n_links)
            sites = rng.integers(0, N, size=(n_links, 4))
            cs = rng.choice((-1, 1), size=n_links)
            us = rng.uniform(0, 1, size=n_links)
            for i in range(n_links):
                mu = int(mus[i])
                site = (int(sites[i, 0]), int(sites[i, 1]),
                        int(sites[i, 2]), int(sites[i, 3]))
                c = int(cs[i])
                # The link's charge response on the CURRENT background, from the same
                # local stencils the clean worms use -- but here a messy Δq is not a
                # rejection, it is a price: ΔD counts how many units of |Q| the move
                # creates (+) or annihilates (-), summed over the touched hypercubes.
                dq = local_charge.charge_change_from_link(F, mu, site, c, N)
                dD = 0
                for cell, dv in dq.items():
                    q0 = defects.get(cell, 0)
                    dD += abs(q0 + dv) - abs(q0)
                self.proposed += 1
                if self.D_max is None or D + dD <= self.D_max:
                    # Metropolis on the ENLARGED weight e^{-S_V} ζ^D: the Villain ΔS is
                    # local to this one link, and the constraint enters only through
                    # ζ^ΔD -- defect-annihilating moves (ΔD < 0) are REWARDED, which is
                    # what lets the mess clean itself up.  (The D_max cap above is just
                    # a truncated state space: proposals past it are ordinary
                    # zero-weight rejections.)
                    link = (mu,) + site
                    A = dphi[link] - twopi * n[link]
                    dS = halfk * ((A - twopi * c)**2 - A**2)
                    if us[i] < np.exp(-dS) * zeta**dD:
                        n[link] += c
                        local_charge.apply_link_to_F(F, mu, site, c, N)
                        for cell, dv in dq.items():
                            q1 = defects.get(cell, 0) + dv
                            if q1:
                                defects[cell] = q1
                            else:
                                defects.pop(cell, None)
                        D += dD
                        self.accepted += 1
                # ---- tally the sector on every clock tick (accepted or not).  This is
                # where the physics is read off: the estimator is pure bookkeeping of
                # WHERE the chain happens to sit.  A rejection is a genuine self-loop
                # and must be counted, or the dwell-time ratio is biased.
                if tally:
                    if D == 0:
                        # Vacuum sector: a valid Q ≡ 0 configuration -- one tick of Z.
                        self._H_Z += 1
                    elif D == 2 and len(defects) == 2:
                        # Exactly the worm's G-sector: a single ±1 pair.  (D == 2 alone
                        # is not enough -- one cell with |Q| = 2 also has D = 2.)
                        (c1, v1), (c2, v2) = defects.items()
                        if v1 == -v2 and abs(v1) == 1:
                            plus, minus = (c1, c2) if v1 == 1 else (c2, c1)
                            disp = tuple((plus[k] - minus[k]) % N for k in range(4))
                            self._H_pair[disp] += 1
                    # Every other sector (4 defects, charge-2 cells, ...) is scaffolding:
                    # legal states that carry the chain THROUGH jammed backgrounds but
                    # never enter the estimator.
            # ---- one φ sweep at fixed n, then refresh dphi.  φ must fluctuate or the
            # Villain weights are sampled at frozen dφ; SiteUpdate is exact for any n
            # (it never touches the constraint), and dφ is fixed during the link sweep
            # so the cached dphi array stays valid until here.
            cfg = site_update.step({'phi': Form(phi, degree=0, lattice=L),
                                    'n': Form(n, degree=1, lattice=L)})
            phi = np.asarray(cfg['phi']).astype(float)
            dphi = np.asarray(d(cfg['phi']))
            self.D_trace.append(D)
        return phi, n

    # ---------------------------------------------------------------- estimator

    def correlator(self):
        r"""
        The block-jackknife mean and error of the absolutely-normalized correlator
        $G(r) = H_{\text{pair}}(r) / (V \zeta^{2} H_{Z})$ over the accumulated blocks.

        Returns
        -------
        (G, dG)
            Arrays of spatial shape ``L.dims``; recall $G(0) = 1$ by definition and the
            $r = 0$ bin of $H_{\text{pair}}$ is empty by construction.
        """
        # The estimator: E[1_{pair at displacement r}] / E[1_{vacuum}] equals
        # ζ² V G(r) in the enlarged ensemble -- the pair sector carries the known
        # fugacity price ζ² (divided back out here, which is why the answer cannot
        # depend on ζ) and V translated copies contribute to each displacement bin.
        # G(0) = 1 identically: a coincident pair IS the vacuum, so no origin
        # normalization is needed -- G comes out ABSOLUTE.
        V = self.N**4
        H_pair = np.stack([b[0] for b in self.blocks])
        H_Z = np.array([b[1] for b in self.blocks], dtype=float)
        if H_Z.sum() == 0:
            nan = np.full(self.L.dims, np.nan)
            return nan, nan
        total = H_pair.sum(axis=0) / (V * self.zeta**2 * H_Z.sum())
        rows = [j for j in range(len(self.blocks)) if H_Z.sum() - H_Z[j] > 0]
        jack = np.stack([
            (H_pair.sum(axis=0) - H_pair[j]) / (V * self.zeta**2 * (H_Z.sum() - H_Z[j]))
            for j in rows])
        err = np.sqrt((len(rows) - 1) * jack.var(axis=0)) if len(rows) > 1 \
            else np.full(self.L.dims, np.nan)
        return total, err

    def report(self):
        r"""A short summary: acceptance, the defect-count trace, and the sector dwell."""
        Dt = np.array(self.D_trace)
        H_Z = sum(b[1] for b in self.blocks)
        H_pair = sum(b[0].sum() for b in self.blocks)
        lines = [f'proposals {self.proposed}  acceptance {self.accepted/max(1,self.proposed):.4f}',
                 f'defect count D: mean {Dt.mean():.2f}  max {int(Dt.max())}' if len(Dt) else 'no sweeps',
                 f'sector dwell: vacuum {H_Z}  single-pair {int(H_pair)}  '
                 f'other {self.proposed - H_Z - int(H_pair)}']
        return '\n'.join(lines)
