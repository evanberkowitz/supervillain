
#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.h5 import H5able

class Sequentially(H5able):
    r'''
    Sequentially applies the list of generators as a single step.

    For example we can get an ergodic :class:`~.Worldline` update scheme by combining the :class:`~.PlaquetteUpdate` and :class:`~.HolonomyUpdate`.

    >>> p = supervillain.generator.constraint.PlaquetteUpdate(S)
    >>> h = supervillain.generator.constraint.HolonomyUpdate(S)
    >>> g = supervillain.generator.combining.Sequentially((p, h))

    Parameters
    ----------
    generators: iterable of generators
        The ordered iterable of generators to apply one after the next.
    '''

    def __init__(self, generators):
        
        self.generators = generators
        
    def step(self, cfg):
        r'''
        Apply each generator's ``step`` one after the next and return the final configuration.
        '''
        
        result = cfg
        for g in self.generators:
            result = g.step(result)
            
        return result

    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''
        return '\n\n'.join(g.report() for g in self.generators)
