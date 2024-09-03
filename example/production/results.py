
from collections import deque
import pandas as pd

import supervillain
import steps

import logging
logger = logging.getLogger(__name__)

# Here are some utilities for collecting data.


# collect produces a dataframe with essentially all observables for each bootstrapped ensemble on disk.
def collect(ensembles, observables=()):

    data = deque()

    if not observables:

        observables = supervillain.observables | supervillain.derivedQuantities

    for idx, row in ensembles.iterrows():

        for line in str(row).split('\n'):
            logger.info(line)
        if not (B := steps.Possible(steps.Bootstrap).of(row)):
            logger.info('Bootstrap not available.')
            continue

        for o in observables:
            (mean, std) = B.estimate(o)
            row[o] = mean
            row[f'{o}±'] = std

        data.append(row)

    return pd.DataFrame(data)

# Here is an iterator which loops over all the ensembles on disk.
# It returns the ensemble, not the dataframe row.
def ensembles(df):
    r'''
    A generator which emits ensembles that are really on disk.
    '''
    for idx, row in df.iterrows():
        if not (E:=steps.Possible(steps.Ensemble).of(row)):
            continue
        yield E


# This lets us easily transform a set of figures into a multipage PDF.
def pdf(filename, figures):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(filename) as PDF:
        for fig in figures:
            fig.savefig(PDF, format='pdf')
