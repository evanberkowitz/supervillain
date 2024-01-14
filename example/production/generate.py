#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import supervillain
import example

import logging
logger = logging.getLogger(__name__)

####
#### Monte Carlo generation
####

def generate(args):

    with h5.File(example.h5, 'r') as h:
        if (e:=example.decorrelated(args)) in h:
            logger.info(f'{e} already exists.')
            return

    L = supervillain.Lattice2D(args.N)

    if args.action == 'villain':
        S = supervillain.action.Villain(L, args.kappa, args.W)
        G = supervillain.generator.villain.NeighborhoodUpdate(S)
    elif args.action == 'worldline':
        S = supervillain.action.Worldline(L, args.kappa, args.W)
        p = supervillain.generator.worldline.PlaquetteUpdate(S)
        h = supervillain.generator.worldline.WrappingUpdate(S)
        G = supervillain.generator.combining.Sequentially((p, h))

    with logging_redirect_tqdm():
        E = supervillain.Ensemble(S).generate(args.configurations, G, start='cold', progress=tqdm)

    E.measure()

    with h5.File(example.h5, 'a') as h:
        E.to_h5(h.create_group(example.ensemble(args)))

    ####
    #### Decorrelate
    ####

    # Make a first estimate without cutting the thermalization.
    tau = E.autocorrelation_time()
    logger.info(f'Estimated autocorrelation time={tau} (unthermalized)')
    thermalized = E.cut(5*tau)

    # With a thermalized ensemble we can now decorrelate with an accurate autocorrelation time.
    tau = thermalized.autocorrelation_time()
    logger.info(f'Estimated autocorrelation time={tau}')
    decorrelated = thermalized.every(tau)

    with h5.File(example.h5, 'a') as h:
        decorrelated.to_h5(h.create_group(example.decorrelated(args)))

if __name__ == '__main__':
    args = example.parser.parse_args()
    generate(args)
