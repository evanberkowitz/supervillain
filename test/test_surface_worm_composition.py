#!/usr/bin/env python

r"""The SurfaceWormGas honors the Generator composition contract (2026-08-04).

Until then ``step()`` ignored its incoming configuration outright, which broke
``Sequentially``: any upstream generator's change to $n$ (and hence $F = dn$)
was silently discarded.  The contract now: ``step`` ALWAYS resumes from the
incoming configuration (building $F = dn$ is essentially free next to the
sweep's moves); a configuration without ``'n'`` (``step({})``) trusts the
internal state, which is how a mid-excursion chain (``equilibrate``) is
advanced without a reset.

``ticksPerStep=0`` makes ``step()`` = (resume) + ``emit`` with no moves, so
the rule is tested deterministically: emission resamples the winding coset
and $\phi$ but preserves $dn = F$ exactly.
"""

import numpy as np
import pytest

from supervillain.action import NoIntersections
from supervillain.lattice import Lattice, d
from supervillain.generator.no_intersection import SurfaceWormGas


N = 4


def dn(S, n):
    f = S.Lattice.form(1, dtype=np.int64)
    np.asarray(f)[...] = n
    return np.asarray(d(f)).astype(np.int64)


@pytest.fixture
def S():
    return NoIntersections(Lattice(4, N), kappa=0.1)


@pytest.fixture
def gas(S):
    # ticksPerStep=0: step() makes no moves, so it is exactly (resume) + emit
    return SurfaceWormGas(S, openSurfaceFugacity=0.1, ticksPerStep=0,
                          measure=False, seed=1)


def link_configuration(mu=0, amplitude=1):
    r"""A valid nontrivial configuration: one occupied link.  Its $dn$ is
    closed and exact, and $dn \wedge dn = 0$ identically for a single link."""
    n = np.zeros((4,) + (N,) * 4, dtype=np.int64)
    n[(mu,) + (0,) * 4] = amplitude
    return {'n': n, 'phi': np.zeros((N,) * 4)}


def test_step_resumes_from_start(S, gas):
    cfg = link_configuration()
    out = gas.step(cfg)
    assert np.array_equal(dn(S, out['n']), dn(S, cfg['n']))


def test_step_resumes_from_upstream_change(S, gas):
    out = gas.step(link_configuration(mu=0))
    changed = dict(out)
    changed['n'] = link_configuration(mu=1, amplitude=2)['n']
    out2 = gas.step(changed)
    assert np.array_equal(dn(S, out2['n']), dn(S, changed['n']))


def test_own_record_roundtrip_is_lossless(S, gas):
    out = gas.step(link_configuration(mu=2))
    F = gas._state.F.copy()
    gas.step(out)          # resume from our own emission: same F, bit for bit
    assert np.array_equal(gas._state.F, F)


def test_configuration_without_n_trusts_internal_state(S, gas):
    target = link_configuration(mu=3, amplitude=3)
    gas.warm_start(target)
    out = gas.step({})     # no 'n': the equilibrate/warm-start bridge
    assert np.array_equal(dn(S, out['n']), dn(S, target['n']))


def test_cold_seed_is_a_genuine_cold_start(S, gas):
    gas.step(link_configuration(mu=1))              # even after a warm history
    cold = {'n': np.zeros((4,) + (N,) * 4, dtype=np.int64),
            'phi': np.zeros((N,) * 4)}
    out = gas.step(cold)
    assert not dn(S, out['n']).any()


if __name__ == '__main__':
    exit(pytest.main([__file__, '-v']))
