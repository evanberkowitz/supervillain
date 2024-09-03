#!/usr/bin/env python

from collections import deque
from itertools import product, chain
import numpy as np
import pandas as pd

ensembles = deque()

################################################################################
# STORAGE
# Where should we write things to disk?
################################################################################

storage = {
    'thermalization storage': 'demo/transition-thermalize.h5',
    'ensemble storage':  'demo/transition-ensemble.h5',
    'bootstrap storage': 'demo/transition-bootstrap.h5',
}

################################################################################
# GENERATION
################################################################################
generate = {
    'thermalize': 1000,         # How many steps to take to measure τ?
    'thermalization cut': 10,   # Multiplies τ to cut and recompute τ.
    'configurations':   100,   # How many configurations in production?
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
W=2
################################################################################

N=5
for kappa in (0.10, 0.125, ):
    ensembles.append(villain   | {'W': W, 'kappa': kappa, 'N':  N, })

for kappa in (0.15, 0.17, 0.185, 0.20, 0.25, 0.30, ):
    ensembles.append(worldline | {'W': W, 'kappa': kappa, 'N':  N, })

#N=7
#for kappa in (0.10, 0.125, ):
#    ensembles.append(villain   | {'W': W, 'kappa': kappa, 'N':  N, })
#
#for kappa in (0.15, 0.17, 0.185, 0.20, 0.25, 0.30, ):
#    ensembles.append(worldline | {'W': W, 'kappa': kappa, 'N':  N, })
#
#
#N=15
#for kappa in np.linspace(0.17, 0.20, 10):
#    ensembles.append(worldline | {'W': W, 'kappa': kappa, 'N':  N, 'configurations': 400})
#
#N=21
#for kappa in np.linspace(0.17, 0.20, 10):
#    ensembles.append(worldline | {'W': W, 'kappa': kappa, 'N':  N, 'configurations': 400})


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
    import supervillain
    parser = supervillain.cli.ArgumentParser()
    parser.add_argument('--figure', default=False, action='store_true')
    parser.add_argument('--pdf', default='', type=str)

    args = parser.parse_args()

    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', None,
            ):
        print(ensembles)

    if args.figure or args.pdf:

        import results
        import transition 

        figs = transition.visualize(results.collect(ensembles))

        if args.pdf:
            results.pdf(args.pdf, figs)
        if args.figure:
            import matplotlib.pyplot as plt
            plt.show()
