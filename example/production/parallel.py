#!/usr/bin/env python3

import h5py as h5
import pandas as pd
import supervillain
from production import produce

import steps
from steps import progress

# If we have a very big computational task ahead of us we may benefit from distributing an ensemble
# to each available core, which is a conceptually simple and straightforward division of labor.
# Therefore, we use the multiprocessing library
from multiprocessing import Pool, cpu_count

# The primary danger is in i/o.  The issue is that if more than one worker tries to open the same HDF5 file
# there might an an exception or file corruption.  Therefore we will need to let each worker write its own file
def temp_file_stem(row):

    return f"W{row['W']}-kappa{row['kappa']:.6f}-N{row['N']}-{row['action']}"

# and play some games with the user-prepared dataframe of storage.
# We store the user's desired storage, adding the gather tag, and rewrite the filename ensemble-by-ensemble.
def io_prep(row):
    
    for name, value in row.items():
        if 'storage' not in name:
            continue

        row[name+ ' gather'] = value
        row[name] = f'{value[:-3]}-{temp_file_stem(row)}.h5'

    return row

# Finally we are ready to set up some work.
# Parallelize takes
#
#  - a function f that loops over a dataframe of ensembles and
#  - a number of threads, which defaults to the multiprocessing cpu_count
#
# and is callable on
#
#  - ensembles that f can act on and
#  - keys of data to gather, essentially undoing the i/o splitting by linking datasets into where they were 'supposed' to be gathered.
#
class Parallelize:

    def __init__(self, f, threads=cpu_count()):
        self.f = f
        self.threads = threads

    def _gather(self, row, key):
        start = key
        target = key + ' gather'
        try:
            with h5.File(row[target], mode='a') as h5fw:
                print(f"Linking {row[start]}/{row['path']} into {row[target]}")
                h5fw[row['path']] = h5.ExternalLink(row[start], row['path'])
        except Exception as e:
            print('GATHER:', e)


    def __call__(self, ensembles, gather=()):

        rewritten = ensembles.apply(io_prep, axis=1)

        with Pool(self.threads) as p:
            try:
                p.map(self.f, (row.to_frame().T for idx, row in rewritten.iterrows()))
            except Exception as e:
                print(e)

        for g in gather:
            rewritten.apply(lambda row: self._gather(row, g), axis=1)


