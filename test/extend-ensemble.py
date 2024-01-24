#!/usr/bin/env python

from pathlib import Path
here = Path(__file__).parent

from collections import deque
import numpy as np
import h5py as h5
import supervillain

parser = supervillain.cli.ArgumentParser()
parser.add_argument('--N', type=int, default=2, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.2, help='κ.')
parser.add_argument('--W', type=int, default=1, help='The constraint integer W.')
parser.add_argument('--configurations', type=int, default=100)
parser.add_argument('--action', type=str, default='villain', choices=['villain', 'worldline'])
parser.add_argument('--h5-file', type=str, default=f'{here}/extend-ensemble.h5')

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)


# The game is:
# 1. Produce an ensemble, write to disk.
# 2. Produce an extension of the ensemble, extend what is already on disk.
# 3. Compare the extended ensemble with the two produced and check that all the measurements match.

L = supervillain.Lattice2D(args.N)

if args.action == 'villain':
    S = supervillain.action.Villain(L, args.kappa, args.W)
    G = supervillain.generator.villain.NeighborhoodUpdate(S)
elif args.action == 'worldline':
    S = supervillain.action.Worldline(L, args.kappa, args.W)
    p = supervillain.generator.worldline.PlaquetteUpdate(S)
    h = supervillain.generator.worldline.WrappingUpdate(S)
    G = supervillain.generator.combining.Sequentially((p, h))

with h5.File(args.h5_file, 'w') as f:
    unmeasured= f.create_group('unmeasured')
    measured = f.create_group( 'measured')

    E = supervillain.Ensemble(S).generate(args.configurations, G, start='cold')
    E.to_h5(unmeasured)
    E.measure()
    E.to_h5(measured)

    F = supervillain.Ensemble.continue_from(unmeasured, args.configurations)
    F.extend_h5(unmeasured)
    F.measure()
    F.extend_h5(measured)

    combined_u = supervillain.Ensemble.from_h5(unmeasured)
    combined_m = supervillain.Ensemble.from_h5(measured)
    combined_u.measure()

failed = 0
for o in combined_m.measured:
    e = getattr(E, o)
    f = getattr(F, o)
    m = getattr(combined_m, o)
    u = getattr(combined_u, o)

    # Extending and then measuring gives the same results as measuring and then extending.
    commutator = (1-(m == u).all()).sum()
    if commutator == 0:
        logger.info (f'PASSED: {o} extension and measurements commute.')
    else:
        logger.error(f'FAILED: {o} extension and measurements did not commute on {commutator} entries.')
        failed += 1

    # The first ensemble is the first half of the combined measurements.
    if (mistakes:=(e != m[:args.configurations]).sum()) != 0:
        logger.error(f'FAILED: {o} read in measurements and first set of computed measurements differ; {str(mistakes)} mistakes.')
        failed += 1
    else:
        logger.info (f'PASSED: {o} read in measurements and first set of computed measurements agree.')

    # The second ensemble is the second half of the combined measurements.
    if (mistakes:=(f != m[args.configurations:]).sum()) != 0:
        logger.error(f'FAILED: {o} read in measurements and second set of computed measurements differ; {str(mistakes)} mistakes.')
        failed += 1
    else:
        logger.info (f'PASSED: {o} read in measurements and second set of computed measurements agree.')

if failed > 0:
    logger.error(f'{failed} FAILURES.')
exit(failed)
