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
    plain :class:`~.Bootstrap`,
 4. reopens the file to show that a :class:`~.StreamingBootstrap` picks up where
    it left off, serving what it already streamed and streaming what it has not,
    and
 5. tabulates the two estimates, for every scalar observable and derived quantity
    the action implements.  Correlators are left out; a table of per-site
    estimates would bury the comparison rather than sharpen it, and
 6. runs the whole analysis pipeline --- thermalize, decorrelate, block, bootstrap
    --- both ways, since an ensemble large enough to need streaming still needs
    all of it.

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
from supervillain.analysis import Bootstrap, Blocking, Uncertain
from supervillain.analysis import EnsembleStreamer, StreamingBlocking, StreamingBootstrap

# supervillain.observable.progress is left at its no-op default: the table below
# measures every observable twice over, and a progress bar per measurement would
# bury the output we actually care about.

import logging
logger = logging.getLogger(__name__)

parser = supervillain.cli.ArgumentParser(description=__doc__)
parser.add_argument('--N', type=int, default=5, help='Sites on a side.  Defaults to 5.')
parser.add_argument('--kappa', type=float, default=0.2, help='κ.  Small, so that vortices are plentiful.  Defaults to 0.2.')
parser.add_argument('--configurations', type=int, default=1000, help='Configurations per generation pass.  Defaults to 1000.')
parser.add_argument('--continuations', type=int, default=2, help='How many times to continue from disk and extend.  Defaults to 2.')
parser.add_argument('--draws', type=int, default=100, help='Bootstrap resamplings.  Defaults to 100.')
parser.add_argument('--chunk', type=int, default=64, help='Configurations held in memory at once while streaming.  Defaults to 64.')
parser.add_argument('--file', type=str, default='streaming-bootstrap.h5', help='Where to write the ensemble.  Defaults to streaming-bootstrap.h5.')

args = parser.parse_args()

# Streamed in step 3 and left on disk, so that step 4 has something to resume
# from: a scalar, a correlator, and a derived quantity built out of primaries.
SEEDED = ('ActionDensity', 'Spin_Spin', 'InternalEnergyDensityVariance')


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
#### 3. Bootstrap the ensemble both ways.
####

with h5.File(args.file, 'r+') as f:

    # The whole ensemble, in memory, resampled the ordinary way.  Reading it is
    # eager, so `whole` and `plain` outlive the open file; only the streaming
    # side needs the file to stay open.
    whole = supervillain.Ensemble.from_h5(f['ensemble'])
    plain = Bootstrap(whole, draws=args.draws)

    # The same ensemble, never held in memory, resampled a chunk at a time.  Handing
    # over the plain bootstrap's indices is what makes the comparison exact: the
    # two resample identically, so they must agree to roundoff.
    streamer  = EnsembleStreamer(f['ensemble'], chunk=args.chunk)
    streaming = StreamingBootstrap(
            streamer, f.create_group('bootstrap'), indices=plain.indices)

    print(f'\n{len(whole)} configurations of {S}')
    print(f'{args.draws} draws; streaming {args.chunk} configurations at a time.')

    # Touching a quantity is what stores it, so this is all it takes to leave
    # something on disk for the next pass to find.
    for quantity in SEEDED:
        streaming.estimate(quantity)

####
#### 4. A streaming bootstrap resumes.  Reopening the file recovers everything
####    already streamed --- and the link back to the ensemble, so that whatever
####    was not streamed still can be.
####

with h5.File(args.file, 'r+') as f:

    resumed = StreamingBootstrap.from_h5(f['bootstrap'])

    cached = sorted(k for k in f['bootstrap'] if k in supervillain.observables
                    or k in supervillain.derivedQuantities)
    print(f'\nResumed a bootstrap of {len(resumed.source)} configurations, with '
          f'{len(cached)} quantities already on disk:')
    print('   ' + ', '.join(cached))

    # One that was already stored, read back with the bootstrap rather than
    # recomputed, and one that was not, which streams the ensemble now and is
    # written through as it goes.
    fresh = 'WindingSquared'
    assert fresh not in cached

    asked = (('ActionDensity', 'read back'), (fresh, 'freshly streamed'))
    label = {quantity: f'{quantity}, {how}:' for quantity, how in asked}
    width = max(len(_) for _ in label.values())

    print()
    for quantity, _ in asked:
        mean, error = resumed.estimate(quantity)
        print(f'   {label[quantity]:{width}s} {Uncertain(float(mean), float(error))}')
    print(f'   ... and {fresh} is now on disk: {fresh in f["bootstrap"]}')

####
#### 5. Every scalar the action implements, estimated both ways.
####
#### The streaming column comes from a bootstrap resumed off the disk: a handful
#### of quantities are read back from step 3, the rest stream now.  Which is
#### which makes no difference to the numbers, which is the point.
####

def compare(streaming, plain, quantity):
    r'''Estimate a scalar ``quantity`` both ways and reduce the pair to one row.

    Only scalars are compared.  A correlator carries one estimate per lattice
    site, and a per-site table would bury the comparison rather than sharpen it;
    some of those sites are fixed by construction anyway (a normalized correlator
    is identically 1 at the origin, where the estimates and their uncertainties
    are both roundoff and their ratio is meaningless).

    The spin and vortex correlators are still represented, by their
    susceptibilities among the derived quantities.  They have to be derived
    quantities: a susceptibility integrates a *normalized* correlator, and
    normalizing divides by an expectation value, which no single configuration can
    supply.  Nothing scalar stands in for ``ActionTwoPoint`` or ``Links`` at all,
    and :class:`~.WindingSquared` is not a summary of
    :class:`~.Winding_Winding` --- it is that correlator at zero separation, the
    contact term rather than the integral.

    So this table is not a complete account of the correlators.  That a streamed
    correlator matches an in-memory one element by element is checked in
    test/test_streaming.py, which is the right place for it.

    The streaming estimate is taken first, deliberately: it is the memory-bounded
    one, so it is always safe to ask, and a non-scalar is dropped before the plain
    :class:`~.Bootstrap` ever builds its ``configurations × draws × sites``
    tensor for it.

    Returns ``None`` if the quantity is not a scalar, or if the action does not
    implement it.
    '''
    try:
        streamed_mean, streamed_error = (np.asarray(_) for _ in streaming.estimate(quantity))
        if streamed_mean.shape != ():
            return None
        mean, error = (np.asarray(_) for _ in plain.estimate(quantity))
    except NotImplementedError:
        return None

    # A zero uncertainty would divide badly: with no difference either, that is
    # agreement; with a difference, it is infinitely many sigma, and inf says so.
    discrepancy = abs(streamed_mean - mean) / (abs(error) if error != 0. else np.inf)

    return (quantity,
            Uncertain(streamed_mean.real, streamed_error.real),
            Uncertain(mean.real, error.real),
            discrepancy)


with h5.File(args.file, 'r+') as f:

    resumed = StreamingBootstrap.from_h5(f['bootstrap'])

    print(f'\n\nEvery scalar {S.__class__.__name__} implements, estimated both ways.')
    print('Correlators are left out; the spin and vortex ones are represented by '
          'their susceptibilities.\n')

    worst_discrepancy = 0.

    for kind, registry in (('observables', supervillain.observables),
                           ('derived quantities', supervillain.derivedQuantities)):

        rows = [row for row in (compare(resumed, plain, q) for q in sorted(registry))
                if row is not None]
        worst_discrepancy = max(worst_discrepancy,
                                max((row[-1] for row in rows), default=0.))

        print(f'{kind:31s} {"streaming":>26s} {"in memory":>26s} {"|Δ|/σ":>10s}')
        print('-' * 96)
        for quantity, streamed, in_memory, discrepancy in rows:
            print(f'{quantity:31s} {str(streamed):>26s} '
                  f'{str(in_memory):>26s} {discrepancy:10.2e}')
        print()

# Not an article of faith --- the worst row in either table above.  Double
# precision carries about 16 digits, so a disagreement below ~1e-9 of an
# uncertainty is accumulated roundoff and anything above it is a real difference.
print(f'\nThe two columns disagree by at most {worst_discrepancy:.2e} of an uncertainty:')
print('roundoff, as it must be --- they are the same sum, accumulated in a different order.'
      if worst_discrepancy < 1e-9 else
      'MORE THAN ROUNDOFF.  The two should be identical resamplings; this is a bug.')
print(f'Everything is in {args.file}; a plain Bootstrap.from_h5 of its /bootstrap '
      'group reads these results on a machine that never sees the ensemble.')


####
#### 6. The rest of the pipeline.  An ensemble too large to read still has to be
####    thermalized and decorrelated, and a streamer can do both --- cut and every
####    for free, autocorrelation_time for the price of the scalars, and blocking
####    as the measurements stream past.
####

with h5.File(args.file, 'r+') as f:

    whole    = supervillain.Ensemble.from_h5(f['ensemble'])
    streamer = EnsembleStreamer(f['ensemble'], chunk=args.chunk)

    tau = streamer.autocorrelation_time()
    measurements = (('measured by streaming', tau),
                    ('from the ensemble in memory', whole.autocorrelation_time()))
    width = max(len(label) for label, _ in measurements)

    print('\n\nAutocorrelation time')
    for label, value in measurements:
        print(f'   {label:{width}s}   {value}')

    # Thermalize, decorrelate, block.  Blocking rather than decimating, because a
    # configuration that is rare and large is exactly the one every() would throw
    # away and blocking averages in.
    cut = 2 * tau
    memory   = Blocking(whole.cut(cut).every(2), width=4)
    streamed = StreamingBlocking(streamer.cut(cut).every(2), width=4)

    print(f'\nCut {cut}, kept every 2nd, blocked 4 together: '
          f'{len(whole)} configurations -> {len(streamed)} blocks')

    reference = Bootstrap(memory, draws=args.draws)
    blocked   = StreamingBootstrap(
            streamed, f.create_group('blocked'), indices=reference.indices)

    print(f'\n{"quantity":31s} {"streamed":>26s} {"in memory":>26s} {"|Δ|/σ":>10s}')
    print('-' * 96)
    worst_blocked = 0.
    for quantity in ('ActionDensity', 'InternalEnergyDensity', 'WindingSquared',
                     'SpinSusceptibility', 'VortexSusceptibility'):
        row = compare(blocked, reference, quantity)
        if row is None:
            continue
        name, streamed_estimate, memory_estimate, discrepancy = row
        worst_blocked = max(worst_blocked, discrepancy)
        print(f'{name:31s} {str(streamed_estimate):>26s} '
              f'{str(memory_estimate):>26s} {discrepancy:10.2e}')

    print(f'\nThe blocked pipeline agrees to {worst_blocked:.2e} of an uncertainty as well.')
