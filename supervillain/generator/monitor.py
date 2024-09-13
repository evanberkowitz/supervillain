#!/usr/bin/env python

import numpy as np
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

from supervillain.performance import Timer

class Logger(ReadWriteable, Generator):
    r'''

    Mostly good for debugging.

    .. code-block :: python

       import logging
       logger = logging.getLogger(__name__)

       ...

       L = supervillain.lattice.Lattice2D(N)
       S = supervillain.action.Worldline(L, kappa, W)
       G = supervillain.generator.worldline.Hammer(S, L.sites)
       M = supervillain.generator.monitor.Logger(G, logger.info)
       E = supervillain.Ensemble(S).generate(100, M)

    Parameters
    ----------
    generator: any generator
        The generator to wrap.
    channel: some function
        Will be applied to the resulting configuration.
    '''

    def __init__(self, generator, channel):
        
        self.generator = generator
        self.channel = channel
        self.iterations = 0
        
    def __str__(self):
        return f'Monitor((' + ', '.join(f'{str(g)}' for g in self.generators) + '))'

    def step(self, cfg):
        r'''
        Apply the wrapped generator and channel the result.
        '''
        
        with Timer(self.channel, f'Iteration {self.iterations}'):
            result = cfg
            result |= self.generator.step(result)
            
            self.channel(result)
        self.iterations += 1

        return result

    def inline_observables(self, steps):
        r'''
        Forwards to the wrapped generator.
        '''
        return self.generator.inline_observables(steps)

    def report(self):
        r'''
        Forwards to the wrapped generator.
        '''
        return self.generator.report()
