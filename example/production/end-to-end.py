#!/usr/bin/env python


import numpy as np
import h5py as h5
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt

import supervillain
import example

from generate import generate
from bootstrap import bootstrap
from plot_history import plot_history
from compare import compare

ARGS=(
        [],
        ['--action', 'worldline'],
        )

ARGS = tuple(example.parser.parse_args(a) for a in ARGS)

for args in ARGS:

    generate(args)
    bootstrap(args)
    fig, ax = plot_history(args)
    fig.savefig(example.file(args, '.pdf'))
    plt.close(fig)

bootstraps = tuple(example.bootstrap(a) for a in ARGS)
results = compare(bootstraps)

print(results.T)
