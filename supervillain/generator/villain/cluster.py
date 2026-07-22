#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form

import logging
logger = logging.getLogger(__name__)


class VillainWolff(ReadWriteable, Generator):
    r'''
    A Wolff single-cluster reflection update for the :class:`~.Villain` action --- the
    embedded-reflection cluster algorithm for the O(2) spin sector.

    Draw a reflection value $r \sim U[0, 2\pi)$ and a seed site.  A cluster is grown and
    then reflected: on cluster sites $\phi_x \to 2r - \phi_x$, and on *internal* links
    (both endpoints in the cluster) $n \to -n$.  Because the gauge-invariant link
    $\theta = (d\phi - 2\pi n)$ then maps to $-\theta$ on every internal link, each
    internal-bond energy $\frac{\kappa}{2}\theta^2$ is exactly invariant; only boundary
    bonds change the action.  Whole-lattice reflection is the exact global symmetry
    $\theta \to -\theta$ (a reflection of the O(2) spin combined with charge conjugation
    $n \to -n$).

    The cluster grows across a boundary bond from an in-cluster site $x$ to an out-of-cluster
    neighbor $y$ with the Fortuin--Kasteleyn / Wolff probability

    .. math ::

        q = 1 - \exp\!\left(-\left[E_b^R - E_b\right]_+\right),
        \qquad E_b = \tfrac{\kappa}{2}\theta_b^2,

    where $E_b^R$ is the bond energy with $x$ reflected (and $y$, $n$ held fixed): $y$ is
    bound into the cluster precisely when reflecting $x$ alone would *raise* the bond's
    energy, so the bond becomes internal (energy-preserving) rather than paying that cost.
    This makes the whole cluster move rejection-free (acceptance 1) while satisfying
    detailed balance.

    .. note ::

        A torus-wrapping cluster *does* change the winding holonomy
        ${w}_\mu = \sum_\text{cycle} {n}_\mu$ (the $H^1$ class): the move realizes the global charge-conjugation reflection
        $n \rightarrow -n$, so a cluster wrapping a cycle flips ${w}_\mu \rightarrow -{w}_\mu$.  It
        does **not** leave $H^1$ fixed.  (Do not confuse this holonomy with the
        :class:`~.WindingSquared` observable $\langle (dn)^2\rangle$, the local vortex density,
        which is a different quantity.)  The move is nonetheless **not ergodic on its own**,
        because it only *reflects* existing structure: from $n = 0$ it can never produce
        $n \neq 0$ (since $-0 = 0$), and with the current uniform reflection value the clusters
        are near-global and realize sign flips $w \rightarrow -w$ rather than moves between
        winding magnitudes.  Combine it with the local updates (e.g. the
        :class:`~.SiteHeatbath`/:class:`~.LinkHeatbath`); its value is decorrelating the
        long-wavelength spin modes near criticality, where local updates critically slow.

    .. warning ::

        This is a correctness-first reference implementation: the cluster is grown with a
        Python breadth-first search, so it is $O(\text{cluster size})$ per step but not
        numba-accelerated.  Large lattices will want a compiled cluster grower.

    .. warning ::

        Implemented for the unconstrained $W = 1$ :class:`~.Villain` model only.  The
        boundary $n \rightarrow -n$ flip changes $dn$ on boundary plaquettes, which is
        harmless when $dn$ is unconstrained ($W = 1$) but breaks the winding constraint
        $dn \equiv 0\ (\bmod W)$ for $W > 1$ and the no-intersection constraint
        $dn \wedge dn = 0$ of the :class:`~.NoIntersections` model; the constructor rejects
        both.

    .. seealso ::
        :class:`~.SiteHeatbath` and :class:`~.LinkHeatbath` for the local exact samplers it
        is combined with.

    Parameters
    ----------
    action: supervillain.action.Villain
        The unconstrained ($W = 1$) Villain action whose spin sector is updated.
    rng: numpy.random.Generator
        A source of randomness; if omitted a fresh default generator is used.
    '''

    def __init__(self, action, rng=None):
        if not isinstance(action, supervillain.action.Villain):
            raise ValueError('The VillainWolff requires the Villain action.')
        if isinstance(action, supervillain.action.NoIntersections):
            raise ValueError(
                'VillainWolff is not valid for the No-Intersection model: the boundary '
                'n -> -n flip breaks the dn wedge dn = 0 constraint.')
        if action.W != 1:
            raise ValueError(
                f'VillainWolff is implemented for W=1 only, got W={action.W}: at W>1 the '
                'boundary n -> -n flip is not guaranteed to preserve dn = 0 (mod W).')

        self.Action = action
        self.Lattice = action.Lattice
        self.kappa = action.kappa

        self.rng = rng if rng is not None else np.random.default_rng()

        self.clusters = 0
        self.cluster_sizes = 0

    def __str__(self):
        return 'VillainWolff'

    def step(self, cfg):
        r'''
        Grow and reflect one Wolff cluster.

        Parameters
        ----------
        cfg: dict
            A dictionary with phi and n as Forms.

        Returns
        -------
        dict
            Updated configuration (phi reflected on the cluster, n flipped on internal links).
        '''
        L = self.Lattice
        N, D, kappa = L.N, L.D, self.kappa
        phi = np.asarray(cfg['phi']).copy()
        n = np.asarray(cfg['n']).copy()
        p = phi[0]

        r = self.rng.uniform(0.0, 2.0 * np.pi)
        seed = tuple(int(self.rng.integers(0, N)) for _ in range(D))

        in_cluster = np.zeros((N,) * D, dtype=bool)
        in_cluster[seed] = True
        stack = [seed]

        while stack:
            x = stack.pop()
            px = p[x]
            for mu in range(D):
                for sign in (1, -1):
                    y = list(x); y[mu] = (y[mu] + sign) % N; y = tuple(y)
                    if in_cluster[y]:
                        continue
                    if sign == 1:                    # link (x,mu): θ = p[y] − p[x] − 2π n[mu,x]
                        nval = n[(mu,) + x]
                        theta = p[y] - px - 2 * np.pi * nval
                        thetaR = p[y] - (2 * r - px) - 2 * np.pi * nval
                    else:                            # link (y,mu): θ = p[x] − p[y] − 2π n[mu,y]
                        nval = n[(mu,) + y]
                        theta = px - p[y] - 2 * np.pi * nval
                        thetaR = (2 * r - px) - p[y] - 2 * np.pi * nval
                    dE = (kappa / 2) * (thetaR**2 - theta**2)   # E_b^R − E_b
                    if dE > 0 and self.rng.uniform() < 1.0 - np.exp(-dE):
                        in_cluster[y] = True
                        stack.append(y)

        p[in_cluster] = 2 * r - p[in_cluster]
        for mu in range(D):
            internal = in_cluster & np.roll(in_cluster, -1, axis=mu)
            n[mu][internal] *= -1

        self.clusters += 1
        self.cluster_sizes += int(in_cluster.sum())
        return cfg | {'phi': Form(phi, degree=0, lattice=L),
                      'n': Form(n, degree=1, lattice=L)}

    def inline_observables(self, steps):
        return {}

    def report(self):
        avg = self.cluster_sizes / max(self.clusters, 1)
        return f'{self.clusters} Wolff clusters, mean size {avg:.1f}.'
