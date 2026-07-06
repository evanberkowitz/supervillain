#!/usr/bin/env python

import numpy as np

import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable
from supervillain.lattice import Form, d
from supervillain.generator.no_intersection.charge import charge

import logging
logger = logging.getLogger(__name__)


class ScattershotUpdate(ReadWriteable, Generator):
    r"""
    A single **atomic, joint** Metropolis proposal that offers every link an independent
    shift at once,

    .. math::

        P(\Delta n_{\ell} = 0) = 1 - p
        \qquad
        P(\Delta n_{\ell} = \pm k) = \frac{p}{2}\, (1 - r)\, r^{k-1} \quad (k \geq 1),

    drawn independently on every link and accepted or rejected **as one move**.  A
    typical draw touches only $\lambda = p \times (\text{number of links})$ links
    scattered across the lattice --- a few pellets here and there --- but the *support*
    of the proposal is every integer-valued $\Delta n$ whatsoever, and that full support
    is the point.

    Typical draws will include joint proposals of a few links at a time with no fixed template.
    Some will be proposals that single-link sweeps can never make because they would reject
    each link separately for constraint violations.

    This generator guarantees that every valid configuration is proposed
    (with extremely small but positive probability) from every other valid configuration in a single step.
    It is the ergodically perfect proposal for the No-Intersection model.
    No topological sectors, knots or links, or other obstacles to updates can withstand it.

    BUT, of course, the issue is that the proposals will almost always be rejected---either 
    for constraint violations or for changing too many links at once and triggering a large
    change in the action.
    So really it's purpose is a purely formal guarantee, but with any finite amount of computer time
    what really matters are the actual physically-insightful proposals that are accepted
    and can rapidly explore the configuration space.

    .. note ::

        Restricted to $D = 4$.  Updates $n$ only; combine with a $\phi$-update such as
        :class:`~.villain.SiteUpdate`.

    .. warning ::

        This reference implementation verifies the constraint with a global charge
        recompute per proposal.  Unlike the local generators, the worst case here is
        genuinely $O(\text{volume})$ no matter the algorithm --- the proposal's support
        is unbounded --- but a typical draw touches only $\sim$ \texttt{links} links,
        and $\Delta q$ is supported within one lattice shift of the touched plaquettes,
        so a masked, local check would cost $O(\texttt{links})$ in expectation and
        degrade gracefully to the global cost only on the astronomically rare large
        draws.  At reference-implementation lattice sizes the vectorised global
        recompute is in fact competitive with (or faster than) indexed masking; the
        local check becomes worthwhile at production volumes.

    Parameters
    ----------
    S: NoIntersections
        The action.
    links: float
        The expected number of touched links per proposal, $\lambda$ (default 2).
        Because one bad component vetoes the whole joint move, acceptance falls
        exponentially in \texttt{links}; keep it small.
    ratio: float or None
        The geometric ratio $r \in (0, 1)$ of the magnitude distribution.  The default
        (``None``) selects $r = \min(\frac{1}{2},\, e^{-6\pi^{2}\kappa})$: on a flat
        background, raising a single link's $|\Delta n_{\ell}|$ from 1 to 2 costs
        $\Delta S = \frac{\kappa}{2}(2\pi)^{2}(2^{2}-1^{2}) = 6\pi^{2}\kappa$, so the
        proposal tail tracks the Metropolis acceptance cliff and mass beyond $\pm 1$ is
        not wasted.  Any $r \in (0, 1)$ is equally *correct* --- $r$ tunes only
        efficiency --- but $r > 0$ matters: it is what makes the support full and the
        irreducibility argument a theorem.
    """

    def __init__(self, S, links=2, ratio=None):
        if not isinstance(S, supervillain.action.NoIntersections):
            raise ValueError('ScattershotUpdate requires a NoIntersections action.')
        if S.Lattice.D != 4:
            raise ValueError('ScattershotUpdate is only implemented for D = 4.')

        self.Action = S
        self.Lattice = S.Lattice
        self.kappa = S.kappa
        self.rng = np.random.default_rng()

        self.links = links
        if ratio is None:
            # Match the proposal tail to the acceptance cliff (see the class docstring).
            # Guard the exponential against underflow so the tail stays strictly
            # positive --- that positivity is what the irreducibility proof consumes ---
            # and cap at 1/2 so κ = 0 still has a normalizable, thin-tailed magnitude
            # distribution.
            ratio = min(0.5, max(float(np.exp(-6 * np.pi ** 2 * S.kappa)), 1e-300))
        if not (0 < ratio < 1):
            raise ValueError(f'ratio must be in (0, 1), got {ratio}')
        self.ratio = ratio

        self.proposed = 0
        self.null = 0           # draws that touched no link at all
        self.clean = 0          # proposals that preserved q = 0
        self.accepted = 0       # clean proposals that passed Metropolis
        self.acceptance = 0.    # summed Metropolis acceptance probability over clean proposals

    def __str__(self):
        return 'ScattershotUpdate'

    def inline_observables(self, steps):
        return {}

    def step(self, configuration):
        r'''
        Draw one joint $\Delta n$, verify its endpoint satisfies
        $q = dn \wedge dn = 0$, and Metropolis-test it against the Villain action,
        atomically.
        '''
        L = self.Lattice
        n = configuration['n']
        dphi = d(configuration['phi'])
        q_now = charge(n)

        self.proposed += 1

        shape = (L.D,) + L.dims
        p = min(1.0, self.links / (L.D * L.N ** L.D))
        mask = self.rng.random(shape) < p
        count = int(mask.sum())
        if count == 0:
            self.null += 1
            return configuration | {'n': n}

        c = np.zeros(shape, dtype=int)
        magnitudes = self.rng.geometric(1 - self.ratio, size=count)
        signs = 2 * self.rng.integers(0, 2, size=count) - 1
        c[mask] = signs * magnitudes

        trial = Form(np.asarray(n) + c, degree=1, lattice=L)

        # A Metropolis move is an atomic jump: legality is endpoint validity, and no
        # intermediate single-link state needs to exist, let alone be valid.
        if not np.array_equal(charge(trial), q_now):
            return configuration | {'n': n}
        self.clean += 1

        # ΔS in the Villain action over every touched link (vectorised; untouched links
        # contribute exactly zero).
        A = np.asarray(dphi) - 2 * np.pi * np.asarray(n)
        dS = (self.kappa / 2) * ((A - 2 * np.pi * c) ** 2 - A ** 2).sum()
        prob = min(1.0, np.exp(-dS))
        self.acceptance += prob
        if self.rng.uniform(0, 1) < prob:
            self.accepted += 1
            return configuration | {'n': trial}
        return configuration | {'n': n}

    def report(self):
        if self.proposed == 0:
            return 'There were 0 proposed scattershot updates.'
        return (
            f'There were {self.accepted} joint proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.clean / self.proposed:.6f} constraint-preserving fraction'
            +'\n'+
            f'    {self.accepted / self.proposed:.6f} acceptance rate'
            +'\n'+
            f'    {self.acceptance / self.proposed:.6f} expected Metropolis acceptance'
            +'\n'+
            f'    {self.null / self.proposed:.6f} touched-no-links fraction'
        )
