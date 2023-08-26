
import numpy as np

from supervillain import _no_op
import supervillain
from supervillain.h5 import H5able
from supervillain.performance import Timer
import supervillain.h5

import logging
logger = logging.getLogger(__name__)


class Ensemble(H5able):
    r'''An ensemble of configurations importance-sampled according to the ``action``.

    .. todo::
        Decide how to incorporate observables.

    Parameters
    ----------
        Action: an action
            An action which describes the path integral of interest.
    '''

    def __init__(self, action):

        self.Action = action
        r'''The action for the ensemble.'''

    def from_configurations(self, configurations):
        r'''
        Parameters
        ----------
            configurations:
                A set of pre-computed configurations.

        Returns
        -------
            The ensemble itself, so that one can do ``ensemble = Ensemble(action).from_configurations(cfgs)``.
        '''

        self.configurations = configurations

        return self

    def generate(self, steps, generator, start='cold', progress=_no_op, starting_index=0):
        r'''
        Parameters
        ----------
            steps:  int
                Number of configurations to generate.
            generator
                Something which produces a new configuration if called as ``generator.step(previous_configuration)``.
            start:  'cold', or a configuration as a dictionary
                A cold start beins with the all-zero configuration.
                If a dictionary is passed it is used as the zeroeth configuration.
            progress: something which wraps an iterator and provides a progress bar.
                In a script you might use `tqdm.tqdm`_, and in a notebook `tqdm.notebook`_.
                Defaults to no progress reporting.
            starting_index: int
                An ensemble has a ``.index`` which is an array of sequential integers labeling the configurations; this sets the lower value.

        Returns
        -------
            the ensemble itself, so that one can do ``ensemble = GrandCanonical(action).generate(...)``.

        .. _tqdm.tqdm: https://pypi.org/project/tqdm/
        .. _tqdm.notebook: https://tqdm.github.io/docs/notebook/
        '''

        self.configurations = self.Action.configurations(steps)
        self.index = starting_index + np.arange(steps)
        self.weight = np.ones(steps)

        if start == 'cold':
            seed = self.Action.configurations(1)[0]
        elif type(start) is dict:
            seed = start
        else:
            raise ValueError('Not sure how to transform a {type(start)} into a starting configuration.')

        with Timer(logger.info, f'Generation of {steps} configurations', per=steps):

            self.configurations[0] = generator.step(seed)

            for mcmc_step in progress(range(1,steps)):
                self.configurations[mcmc_step] = generator.step(self.configurations[mcmc_step-1])

            self.start = start
            self.generator = generator

        return self

    def __len__(self):
        return len(self.configurations)

    def cut(self, start):
        r'''
        Good for thermalization.

        .. code::

           thermalized = ensemble.cut(start)

        Parameters
        ----------
        start: int
            How many configurations to drop from the beginning of the ensemble.

        Returns
        -------
        Ensemble
            An ensemble with fewer configurations.
        '''
        e = Ensemble(self.Action).from_configurations(self.configurations[start:])
        e.index = self.index[start:]
        e.weight = self.weight[start:]
        return e

    def every(self, stride):
        r'''
        Good for decorrelation.

        .. code::

           decorrelated = thermalized.every(stride)

        Parameters
        ----------
        stride: int
            How many configurations to skip.

        Returns
        -------
        Ensemble
            An ensemble with fewer configurations.
        '''

        e = Ensemble(self.Action).from_configurations(self.configurations[::stride])
        e.index = self.index[::stride]
        e.weight = self.weight[::stride]
        return e
