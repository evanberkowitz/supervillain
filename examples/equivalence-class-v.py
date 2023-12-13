#!/usr/bin/env python

import numpy as np
from itertools import chain
from collections import deque
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import supervillain

parser = supervillain.cli.ArgumentParser(description='Tests the gauge invariance of a variety of observables in the Villain formulation.  Exits with the number of failures; 0 if everything passes.')
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.5, help='κ.')
parser.add_argument('--W', type=int, default=1, help='W')
parser.add_argument('--configurations', type=int, default=1000)
parser.add_argument('--equality-threshold', type=float, default=1e-12, help='Acceptable floating point differences')

args = parser.parse_args()

import logging
logger = logging.getLogger(__name__)


####
#### Observables
####

# Each observable whose history and histogram you want to see can be put into this list.
observables = (
    'XWrapping', 'TWrapping',
    'InternalEnergyDensity',
    'InternalEnergyDensitySquared',
    'WindingSquared',
    'ActionDensity',
    'SpinSusceptibilityScaled',
    'VortexSusceptibilityScaled'
    )

# We can also visualize the space-dependent correlators.
correlators = (
    'Winding_Winding', # the zero correlator?
    'Spin_Spin',
    'Vortex_Vortex',
    'ActionTwoPoint', # We're not bootstrapping, we need primary observables only.
    )

####
#### Monte Carlo generation
####

L = supervillain.Lattice2D(args.N)

S=supervillain.action.Worldline(L, 0.2, 4)

P=supervillain.generator.worldline.PlaquetteUpdate(S)
W=supervillain.generator.worldline.WrappingUpdate(S)
G=supervillain.generator.combining.Sequentially((P, W))

with logging_redirect_tqdm():
    E = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)

####
#### Perform equivalence transformation
####

transformed = S.configurations(len(E))
for i, c in enumerate(E.configuration):
    transformed[i] = S.equivalence_class_v(c)

F = supervillain.Ensemble(S).from_configurations(transformed)

####
#### Compare observables
####

failed = deque()
passed = deque()
for O in chain(observables, correlators):
    difference = getattr(E, O) - getattr(F, O)
    worst = np.max(np.abs(difference))
    if worst > args.equality_threshold:
        logger.info(f'FAILED: {O}')
        failed.append((O, worst))
    else:
        logger.info(f'PASSED: {O}')
        passed.append(O)

if len(passed) > 0:
    print(f'{len(passed)} observables were invariant:')
    for o in passed:
        print(f'PASSED: {o}')
else:
    print('No observables were invariant.')
if len(failed) > 0:
    print(f'\n{len(failed)} observables changed under transformation                worst difference')
    for (o, worst) in failed:
        print(f'FAILED: {o:50s} {worst}')

exit(len(failed))
