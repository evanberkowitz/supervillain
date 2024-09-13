
#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.generator import Generator
from supervillain.h5 import ReadWriteable

class Sequentially(ReadWriteable, Generator):
    r'''
    Sequentially applies the list of generators as a single step.

    For example we can get an ergodic :class:`~.Worldline` update scheme by combining the :class:`~.PlaquetteUpdate` and :class:`~.WrappingUpdate`.

    >>> p = supervillain.generator.worldline.PlaquetteUpdate(S)
    >>> h = supervillain.generator.worldline.WrappingUpdate(S)
    >>> g = supervillain.generator.combining.Sequentially((p, h))

    Parameters
    ----------
    generators: iterable of generators
        The ordered iterable of generators to apply one after the next.
    '''

    def __init__(self, generators):
        
        self.generators = generators
        
    def __str__(self):
        return f'Sequentially((' + ', '.join(f'{str(g)}' for g in self.generators) + '))'

    def step(self, cfg):
        r'''
        Apply each generator's ``step`` one after the next and return the final configuration.
        '''
        
        result = cfg
        for g in self.generators:
            result = g.step(result)
            
        return result

    def inline_observables(self, steps):
        combined = dict()
        for g in self.generators:
            combined |= g.inline_observables(steps)

        return combined

    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''
        return '\n\n'.join(g.report() for g in self.generators)

class KeepEvery(ReadWriteable, Generator):
    r'''
    To decorrelate and get honest error estimates it can be helpful to do MCMC but then only analyze evenly-spaced configurations.
    Rather than keep every single generated configuration and then throw a bunch away, we can keep only those we might analyze.
    
    .. note::
        The number of updates per second will decrease by a factor of n, but the autocorrelation time should be n less.
        Generating a fixed number of configurations will take n times longer.

    >>> p = supervillain.generator.worldline.PlaquetteUpdate(S)
    >>> g = supervillain.generator.combining.KeepEvery(10, p)


    Parameters
    ----------
    n: int
        How many generator updates to make before emitting a configuration.
    generator:
        The generator to use for updates.
    blocked_inline: bool
        Rather than just keeping the :py:meth:`~.Generator.inline_observables` from the last update, all of the inline measurements are averaged across all n updates.
        This helps capture rare-but-important measurements that would otherwise be missed.
    '''

    def __init__(self, n, generator, blocked_inline=True):

        self.stride = n
        self.generator = generator
        self.blocked_inline = blocked_inline

    def __str__(self):
        return f'KeepEvery({self.stride}, {str(self.generator)})'

    def step(self, cfg):
        r'''
        Applies the generator n times and returns the resulting configuration.
        '''

        result = cfg
        blocked = (self.inline_observables(1) if self.blocked_inline else dict())
        for o in blocked:
            blocked[o] = blocked[o][0]

        for i in range(self.stride):
            result = self.generator.step(result)

            for o in blocked:
                blocked[o] += result[o]/self.stride

        return (result | blocked)

    def inline_observables(self, steps):
        return self.generator.inline_observables(steps)

    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''

        return self.generator.report()
