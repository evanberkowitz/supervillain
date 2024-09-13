#!/usr/bin/env python

from collections import deque
from itertools import product, chain
import pandas as pd

ensembles = deque()

################################################################################
# STORAGE
# Where should we write things to disk?
################################################################################

storage = {
    'thermalization storage': 'scaling/W=2/thermalize.h5',
    'ensemble storage':       'scaling/W=2/ensemble.h5',
    'bootstrap storage':      'scaling/W=2/bootstrap.h5',
}

################################################################################
# GENERATION
################################################################################
generate = {
    'thermalize': 1000,         # How many steps to take to measure τ?
    'thermalization cut': 10,   # Multiplies τ to cut and recompute τ.
    'configurations':   2000,   # How many configurations in production?
}

################################################################################
# ANALYSIS
################################################################################
analysis = {
    'bootstraps': 100,          # How many bootstrap samples?
}

################################################################################
# ACTION
# We will use two different actions to do this calculation; some ensembles use
# the original Villain frame and some use the dual Worldline frame.
################################################################################
defaults = storage | generate | analysis

worldline = defaults | {
    'action': 'Worldline',
    'start':  'cold',
}

villain = defaults | {
    'action': 'Villain',
    'start':  'cold',
}


################################################################################
# LATTICE SIZES
################################################################################

wee = (4, 6, 8, 10, 12)
small = tuple(range(4, 20, 4))
medium = tuple(range(24, 48, 8))
large  = (64, )

################################################################################
# COUPLINGS
################################################################################

low = (0.100, 0.125, 0.150, )
near_critical = (0.170, 0.200, 0.250, 0.30, )
brackets_critical = (0.180, 0.185, 0.190, )

################################################################################
W=2
################################################################################

for kappa, N in product(low, wee):
    ensembles.append(villain | {'W': W, 'kappa': kappa, 'N':  N, })

for kappa, N in product(near_critical + brackets_critical, small + medium):
    ensembles.append(worldline | {'W': W, 'kappa': kappa, 'N':  N, })

for kappa, N in product(brackets_critical, large):
    ensembles.append(worldline | {'W': W, 'kappa': kappa, 'N':  N, })

################################################################################
# MAKE A DATAFRAME
################################################################################
ensembles = pd.DataFrame(ensembles)

################################################################################
# AND ADD ANY OBVIOUS PROCESSING
################################################################################
ensembles['path'] = ensembles.apply(lambda row:
    f"W={row['W']}/kappa={row['kappa']:0.5f}/N={row['N']}/{row['action']}",
    axis=1, raw=False
    )

################################################################################
# HUMAN INTERFACE
################################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', type=int, nargs='*')

    args = parser.parse_args()

    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 1000,
            ):
        print(ensembles)

    if args.reset:
        import h5py as h5
        for i, e in ensembles.iloc[args.reset].iterrows():
            for f in (e['bootstrap'], e['ensemble']):
                try:
                    with h5.File(f, 'a') as h:
                        del h[e['path']]
                except Exception as e:
                    print(e)
                    pass

