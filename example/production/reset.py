#!/usr/bin/env python

import h5py as h5

import supervillain
import example

import logging
logger = logging.getLogger(__name__)

def reset(args):
    for data in (example.ensemble, example.decorrelated, example.bootstrap):
        with h5.File(example.h5, 'a') as h:
            try:
                del h[data(args)]
            except:
                pass

if __name__ == '__main__':
    args = example.parser.parse_args()
    reset(args)
