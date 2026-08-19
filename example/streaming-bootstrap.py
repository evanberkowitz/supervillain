#!/usr/bin/env python

r'''
A worked example of growing an ensemble on disk and then bootstrapping it without
ever reading the whole thing into memory.

The script

 1. generates a 2D Villain ensemble at small κ and writes it with
    :meth:`~.Ensemble.to_h5`,
 2. twice reads the last configuration back off disk, continues the Markov chain
    from it with :meth:`~.Ensemble.continue_from`, and appends the continuation
    with :meth:`~.Extendable.extend_h5`,
 3. bootstraps the resulting ensemble two ways --- once by streaming it with an
    :class:`~.EnsembleStreamer` and a :class:`~.StreamingBootstrap`, once by
    reading it whole with :meth:`~.Ensemble.from_h5` and resampling it with a
    plain :class:`~.Bootstrap` --- and compares the two, and
 4. reopens the file to show that a :class:`~.StreamingBootstrap` picks up where
    it left off, serving what it already streamed and streaming what it has not.

The two bootstraps are given the *same* resampling indices.  They are therefore
not merely consistent within errors; they must agree to floating point, since
they compute the same sum in a different order.  Any disagreement beyond roundoff
is a bug, which is exactly what makes this a useful check.
'''

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import supervillain
from supervillain.analysis import Bootstrap, EnsembleStreamer, StreamingBootstrap
from supervillain.analysis import Uncertain
supervillain.observable.progress = tqdm

import logging
logger = logging.getLogger(__name__)

parser = supervillain.cli.ArgumentParser(description=__doc__)
parser.add_argument('--N', type=int, default=5, help='Sites on a side.  Defaults to 5.')
parser.add_argument('--kappa', type=float, default=0.2, help='κ.  Small, so that vortices are plentiful.  Defaults to 0.2.')
parser.add_argument('--configurations', type=int, default=1000, help='Configurations per generation pass.  Defaults to 1000.')
parser.add_argument('--continuations', type=int, default=2, help='How many times to continue from disk and extend.  Defaults to 2.')
parser.add_argument('--draws', type=int, default=100, help='Bootstrap resamplings.  Defaults to 100.')
parser.add_argument('--block', type=int, default=64, help='Configurations held in memory at once while streaming.  Defaults to 64.')
parser.add_argument('--file', type=str, default='streaming-bootstrap.h5', help='Where to write the ensemble.  Defaults to streaming-bootstrap.h5.')

args = parser.parse_args()

# A scalar, two correlators, and two derived quantities built out of primaries.
QUANTITIES = (
        'ActionDensity',
        'InternalEnergyDensity',
        'Spin_Spin',
        'Vortex_Vortex',
        'InternalEnergyDensityVariance',
        'SpinSusceptibility',
        )


####
#### 1. Generate an ensemble and write it to disk.
####

L = supervillain.lattice.Lattice2D(args.N)
S = supervillain.action.Villain(L, args.kappa)
G = supervillain.generator.villain.Hammer(S)

logger.info(f'Generating {args.configurations} configurations of {S}.')
with logging_redirect_tqdm():
    E = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)

with h5.File(args.file, 'w') as f:
    E.to_h5(f.create_group('ensemble'))

####
#### 2. Continue from the last configuration on disk, and extend the stored
####    ensemble with the continuation.  Twice.
####
#### Nothing but the file carries state from one pass to the next; each pass
#### reads the ensemble it continues from off the disk.  This is what a long
#### production campaign looks like --- a chain grown a job at a time.
####

for continuation in range(args.continuations):
    with h5.File(args.file, 'r+') as f:
        logger.info(f'Continuation {continuation+1} of {args.continuations}.')
        with logging_redirect_tqdm():
            F = supervillain.Ensemble.continue_from(
                    f['ensemble'], args.configurations, progress=tqdm)
        F.extend_h5(f['ensemble'])

####
#### 3. Bootstrap the ensemble both ways and compare.
####

with h5.File(args.file, 'r+') as f:

    # The whole ensemble, in memory, resampled the ordinary way.
    whole = supervillain.Ensemble.from_h5(f['ensemble'])
    plain = Bootstrap(whole, draws=args.draws)

    # The same ensemble, never held in memory, resampled block by block.  Handing
    # over the plain bootstrap's indices is what makes the comparison exact: the
    # two resample identically, so they must agree to roundoff.
    streamer  = EnsembleStreamer(f['ensemble'], block=args.block)
    streaming = StreamingBootstrap(
            streamer, f.create_group('bootstrap'), indices=plain.indices)

    print(f'\n{len(whole)} configurations of {S}')
    print(f'{args.draws} draws; streaming {args.block} configurations at a time.\n')
    print(f'{"quantity":35s} {"streaming":>26s} {"in memory":>26s} {"|Δ|/σ":>10s}')
    print('-' * 101)

    for quantity in QUANTITIES:
        streamed_mean, streamed_error = (np.asarray(_).real for _ in streaming.estimate(quantity))
        mean,          error          = (np.asarray(_).real for _ in plain.estimate(quantity))

        # A correlator has one estimate per lattice site; quote the worst
        # discrepancy over all of them, which is the only one that could hide a bug.
        discrepancy = np.abs(streamed_mean - mean) / np.where(error == 0., np.inf, error)

        # Report a representative component of a correlator, but the worst discrepancy.
        report = () if mean.shape == () else np.unravel_index(np.argmax(discrepancy), mean.shape)
        print(f'{quantity:35s} '
              f'{str(Uncertain(streamed_mean[report], streamed_error[report])):>26s} '
              f'{str(Uncertain(mean[report], error[report])):>26s} '
              f'{discrepancy.max():10.2e}')

    print('\nThe two agree to roundoff: they are the same sum, accumulated in a '
          'different order.')

####
#### 4. A streaming bootstrap resumes.  Reopening the file recovers everything
####    already streamed --- and the link back to the ensemble, so that whatever
####    was not streamed still can be.
####

with h5.File(args.file, 'r+') as f:

    resumed = StreamingBootstrap.from_h5(f['bootstrap'])

    cached = sorted(k for k in f['bootstrap'] if k in supervillain.observables
                    or k in supervillain.derivedQuantities)
    print(f'\nResumed a bootstrap of {len(resumed.streamer)} configurations, with '
          f'{len(cached)} quantities already on disk:')
    print('   ' + ', '.join(cached))

    # Served from disk; the ensemble is not touched.
    mean, error = resumed.estimate('ActionDensity')
    print(f'\n   ActionDensity, read back:   {Uncertain(float(mean), float(error))}')

    # Not on disk, so this one streams the ensemble now, and is written through.
    fresh = 'WindingSquared'
    assert fresh not in cached
    mean, error = resumed.estimate(fresh)
    print(f'   {fresh}, freshly streamed: {Uncertain(float(mean), float(error))}')
    print(f'   ... and now on disk: {fresh in f["bootstrap"]}')

print(f'\nEverything is in {args.file}; a plain Bootstrap.from_h5 of its '
      '/bootstrap group reads these results on a machine that never sees the ensemble.')
