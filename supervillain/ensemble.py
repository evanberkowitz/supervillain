
import numpy as np
import h5py as h5

from supervillain import _no_op
import supervillain
from supervillain.h5 import H5able
from supervillain.performance import Timer
import supervillain.h5

import logging
logger = logging.getLogger(__name__)


class Ensemble(H5able):
    r'''An ensemble of configurations importance-sampled according to the ``action``.

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

    @classmethod
    def continue_from(cls, ensemble, steps, progress=_no_op):
        r'''
        Use the last configuration and generator of ``ensemble`` to produce a new ensemble of ``steps`` configurations.
        
        Parameters
        ----------
            ensemble: supervillain.Ensemble or an h5py.Group that encodes such an ensemble
                The ensemble to continue.  Raises a ValueError if it is not a `supervillain.Ensemble` or an `h5py.Group` with an action, generator, and at least one configuration.
            steps: int
                Number of configurations to generate.

        Returns
        -------
            supervillain.Ensemble:
                An ensemble with ``steps`` new configurataions generted in the same way as ``ensemble``.

        .. todo::
           
           The starting weight should automatically be read in; currently not.
        '''
        if isinstance(ensemble, h5.Group):
            e = supervillain.Ensemble.from_h5(ensemble)
            # TODO: as in tdg, read only the last configuration, index, and so on, rather than the whole thing.
        elif isinstance(ensemble, supervillain.Ensemble):
            e = ensemble
        else:
            raise ValueError('ensemble should be a supervillain.Ensemble or an h5 group that stores one.')

        try:
            generator = e.generator
            action    = e.Action
            last      = e.configurations[-1]
            index     = e.index[-1] + 1
        except:
            raise ValueError('The ensemble must provide a generator, an Action, and at least one configuration.')

        return Ensemble(action).generate(steps, generator, last, progress=progress, starting_index=index)

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

    def plot_history(self, axes, observable, label=None,
                     bins=31, density=True,
                     alpha=0.5, color=None,
                     ):

        data = getattr(self, observable)
        axes[0].plot(self.index, data, color=color)
        axes[1].hist(data, label=label,
                     orientation='horizontal',
                     bins=bins, density=density,
                     color=color, alpha=alpha,
                     )
 
