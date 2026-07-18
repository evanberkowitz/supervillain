#!/usr/bin/env python
r"""Tests for the streaming bootstrap: EnsembleStreamer (memory-bounded block
iteration of a serialized Ensemble) and StreamingBootstrap (block-accumulated
resample with a write-through disk-cache gate).  The load-bearing test is
`test_streaming_equivalence`: given the SAME resampling indices, the streaming
estimate must equal a plain Bootstrap's to floating point."""

import numpy as np
import h5py
import pytest

import supervillain
from supervillain.action import NoIntersections
from supervillain.lattice import Lattice
import supervillain.generator.no_intersection as gen
import supervillain.generator.villain as villain
from supervillain.generator.combining import Sequentially
from supervillain.analysis import Bootstrap
from supervillain.analysis.bootstrap import EnsembleStreamer, StreamingBootstrap


def _small_ensemble(tmp_path, N=4, kappa=0.1, configs=40, seed=17):
    """Generate a small NoIntersections DefectGas ensemble and store it to h5.
    Returns the path; the ensemble group is at '/ensemble'."""
    L = Lattice(4, N)
    S = NoIntersections(L, kappa=kappa)
    rng = np.random.default_rng(seed)
    companions = (villain.SiteUpdate(S), villain.ExactUpdate(S),
                  villain.CohomologyUpdate(S), gen.WrappingLoopUpdate(S))
    for g in companions:
        g.rng = rng
    tuner = gen.DefectGasFugacityTuner(S, companions=companions, max_defects=8, rng=rng)
    zeta = tuner.tune(start='cold')
    gas = gen.DefectGas(S, fugacity=zeta, max_defects=8, rng=rng)
    chain = Sequentially((*companions, gas))
    e = supervillain.Ensemble(S).generate(configs, chain, start='cold')
    path = tmp_path / 'ens.h5'
    with h5py.File(path, 'w') as f:
        e.__dict__.pop('generator', None)
        e.__dict__.pop('start', None)
        e.to_h5(f.create_group('ensemble'))
    return path


def test_streamer_fidelity(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        full.generator = None
        streamer = EnsembleStreamer(f['ensemble'], block=7)

        assert len(streamer) == len(full)

        # Field concatenation, block by block.
        phi_blocks, theta_blocks = [], []
        for start, sub in streamer.blocks():
            phi_blocks.append(np.asarray(sub.phi))
            theta_blocks.append(np.asarray(sub.Theta_Theta))
        phi = np.concatenate(phi_blocks, axis=0)
        theta = np.concatenate(theta_blocks, axis=0)
        assert np.array_equal(phi, np.asarray(full.phi))
        assert np.array_equal(theta, np.asarray(full.Theta_Theta))

        # A computed (non-inline) observable concatenated equals the full computation.
        iw_blocks = [np.asarray(sub.IntersectionWinding) for _, sub in streamer.blocks()]
        iw = np.concatenate(iw_blocks, axis=0)
        assert np.allclose(iw, np.asarray(full.IntersectionWinding))


def test_streaming_equivalence(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r+') as f:
        full = supervillain.Ensemble.from_h5(f['ensemble'])
        full.generator = None

        # A reference Bootstrap on the in-memory ensemble.
        ref = Bootstrap(full, draws=50)

        # A StreamingBootstrap forced to use the SAME indices.
        streamer = EnsembleStreamer(f['ensemble'], block=9)
        target = f.create_group('boot')
        sb = StreamingBootstrap(streamer, target, draws=50)
        sb.indices = ref.indices        # force identical resampling
        sb._rebuild_counts()            # rebuild the count matrix from those indices

        for name in ('ActionDensity', 'Theta_Theta', 'IntersectionSusceptibility'):
            m_ref, e_ref = ref.estimate(name)
            m_sb, e_sb = sb.estimate(name)
            assert np.allclose(np.asarray(m_ref), np.asarray(m_sb), atol=1e-10, rtol=1e-8), name
            assert np.allclose(np.asarray(e_ref), np.asarray(e_sb), atol=1e-10, rtol=1e-8), name


def test_write_through_roundtrip_and_portability(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], block=9)
        target = f.create_group('boot')
        sb = StreamingBootstrap(streamer, target, draws=40)
        m_theta, e_theta = sb.estimate('Theta_Theta')
        m_chi, e_chi = sb.estimate('IntersectionSusceptibility')
        assert 'Theta_Theta' in target
        assert 'IntersectionSusceptibility' in target

    # Reload via StreamingBootstrap.from_h5: cached values match; streamer rebuilt.
    with h5py.File(path, 'r+') as f:
        sb2 = StreamingBootstrap.from_h5(f['boot'])
        assert isinstance(sb2.streamer, EnsembleStreamer)
        m2, e2 = sb2.estimate('Theta_Theta')
        assert np.allclose(np.asarray(m2), np.asarray(m_theta))
        # An un-streamed observable now streams through the reconstructed streamer.
        m3, e3 = sb2.estimate('ActionDensity')
        assert np.isfinite(np.asarray(m3)).all()
        assert 'ActionDensity' in f['boot']

    # Portability: a plain Bootstrap.from_h5 estimates the streamed observable.
    with h5py.File(path, 'r') as f:
        plain = Bootstrap.from_h5(f['boot'])
        mp, ep = plain.estimate('Theta_Theta')
        assert np.allclose(np.asarray(mp), np.asarray(m_theta))


def test_resumability_no_recompute(tmp_path):
    path = _small_ensemble(tmp_path)
    with h5py.File(path, 'r+') as f:
        streamer = EnsembleStreamer(f['ensemble'], block=9)
        target = f.create_group('boot')
        sb = StreamingBootstrap(streamer, target, draws=30)
        sb.estimate('Theta_Theta')

        # Break the streamer so any recompute would raise; a cached read must not.
        sb.streamer._source = None
        m, e = sb.estimate('Theta_Theta')    # served from disk
        assert np.isfinite(np.asarray(m)).all()
        with pytest.raises(RuntimeError):
            sb.estimate('ActionDensity')      # not cached -> must stream -> raises


def test_public_exports():
    from supervillain.analysis import StreamingBootstrap as SB, EnsembleStreamer as ES
    assert SB is StreamingBootstrap
    assert ES is EnsembleStreamer
