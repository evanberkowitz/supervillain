#!/usr/bin/env python

from collections import deque
import numpy as np
import pandas as pd
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
import example



def compare(bootstraps):

    results = deque()

    for b in bootstraps:
        with h5.File(example.h5, 'r') as f:
            B = supervillain.analysis.Bootstrap.from_h5(f[b])

        r = {
                'key':   b,
                'kappa': B.Action.kappa,
                'W':     B.Action.W,
                'N':     B.Action.Lattice.nx,
                'samples': len(B.Ensemble),
        }
        results.append(r | {o: B.estimate(o) for o in B.Ensemble.measured
                        if issubclass(supervillain.observables[o], supervillain.observable.Scalar)
                        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = supervillain.cli.ArgumentParser()
    bootstraps = (
            '/bootstrap/kappa=0.50000/W=1/N=5/villain',
            '/bootstrap/kappa=0.50000/W=1/N=5/worldline',
            )

    parser.add_argument('bootstraps', nargs='*', default=bootstraps, help='As many paths to bootstraps in example.h5 as you like.  Defaults to two bootstraps, one with the default parameters and the other with the same parameters but with the worldline action.')
    args = parser.parse_args()

    results = compare(args.bootstraps)
    print(results.T)
