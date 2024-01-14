#!/usr/bin/env python

import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
import example


import logging
logger = logging.getLogger(__name__)

def plot_history(args):
    with h5.File(example.h5, 'r') as h:
        E = supervillain.Ensemble.from_h5(h[example.ensemble(args)])
        D = supervillain.Ensemble.from_h5(h[example.decorrelated(args)])

    scalars = set(o for o in D.measured if issubclass(supervillain.observables[o], supervillain.observable.Scalar))

    fig, ax = plt.subplots(len(scalars), 2,
        figsize=(10, 3*len(scalars)),
        gridspec_kw={'width_ratios': [4, 1], 'wspace': 0},
        sharey='row',
        squeeze=False
        )

    fig.suptitle(example.path(args))

    for a, o in zip(ax, scalars):
        for ensemble in E, D:
            data = getattr(ensemble, o)
            mean = data.mean()
            std  = data.std() / np.sqrt(len(data))

            ensemble.plot_history(a, o, histogram_label=f'{mean:0.4f} ± {std:0.4f}')

        a[0].set_ylabel(o)
        a[0].set_xlabel('Monte Carlo time')
        a[1].legend()

    fig.tight_layout()

    return fig, ax


if __name__ == '__main__':
    args = example.parser.parse_args()
    fig, args = plot_history(args)
    plt.show()
