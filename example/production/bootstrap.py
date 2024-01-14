#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm

import supervillain
import example

import logging
logger = logging.getLogger(__name__)

def bootstrap(args):
    with h5.File(example.h5, 'r') as h:
        if (e:=example.bootstrap(args)) in h:
            logger.info(f'{e} already exists.')
            return

        E = supervillain.Ensemble.from_h5(h[example.decorrelated(args)])

    bootstrap = supervillain.analysis.Bootstrap(E, 100)
    estimates = {o: bootstrap.estimate(o) for o in E.measured}

    with h5.File(example.h5, 'a') as h:
        bootstrap.to_h5(h.create_group(example.bootstrap(args)))

if __name__ == '__main__':
    args = example.parser.parse_args()
    bootstrap(args)
