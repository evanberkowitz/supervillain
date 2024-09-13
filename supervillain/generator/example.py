#!/usr/bin/env python

import numpy as np
from supervillain.generator import Generator

class DoNothing(Generator):

    def __init__(self):
        pass

    def step(self, x):
        r'''
        Just copy the incoming configuration and 'measure' the constant 1
        and add it to the configuration.
        '''
        return x.copy() | {'one': 1}

    def inline_observables(self, steps):
        r'''
        Obviously there is no *actual* reason to compute an observable whose value
        is identically 1, but it's useful to show how inline observables work.
        '''
        return {'one': np.zeros(steps)}
