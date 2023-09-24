#!/usr/bin/env python

import numpy as np

from supervillain.h5 import H5able
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)

class Bootstrap(H5able):
    r'''
    The bootstrap is a resampling technique for estimating uncertainties.

    For samples with weights :math:`w` the expectation value of an observable is

    .. math::
        \left\langle O \right\rangle = \frac{\left\langle O w \right\rangle}{\left\langle w \right\rangle}

    and an accurate bootstrap estimate of the left-hand side requires tracking the correlations between the numerator and denominator.
    Moreover, quoting correlated uncertainties requires resampling different observables in the same way.

    Parameters
    ----------
        ensemble:   Ensemble
            The ensemble to resample.
        draws:      int
            The number of times to resample.

    Any :ref:`primary observables` that :class:`~.Ensemble` supports can be called from the :class:`~.Bootstrap`.
    :class:`~.Bootstrap` uses :code:`getattr` trickery under the hood to intercept calls and perform the weighted average transparently.

    Each observable returns an array of the same dimension as the ensemble's observable.  However, rather than configurations first, :code:`draws` are first.

    Each draw is a weighted average over the resampled weight, as shown above, and is therefore an estimator for the expectation value.
    These are guaranteed (by the `central limit theorem`_) to be normally distributed as long as you have not sinned.
    To get an uncertainty estimate one need only take the :code:`mean()` for a central value and :code:`std()` for the uncertainty on the mean.

    .. _central limit theorem: https://en.wikipedia.org/wiki/Central_limit_theorem
    '''

    def __init__(self, ensemble, draws=100):
        self.Ensemble = ensemble
        r'''The ensemble from which to resample.'''
        self.Action = ensemble.Action
        r'''The action underlying the ensemble.'''
        self.draws = draws
        r'''The number of resamplings.'''
        cfgs = len(ensemble.configurations)
        self.indices = np.random.randint(0, cfgs, (cfgs, draws))
        r'''The random draws themselves; configurations × draws.'''
        
    def __len__(self):
        return self.draws

    def _resample(self, obs):
        # Each observable should be multiplied by its respective weight.
        # Each draw should be divided by its average weight.
        w = self.Ensemble.weight[self.indices]

        # This index ordering is needed to broadcast the weights division correctly.
        # See https://github.com/evanberkowitz/two-dimensional-gasses/issues/55
        # We return the bootstrap axis to the front to provide an analogous interface for Bootstrap and Ensemble quantities.
        return np.einsum('...d->d...', np.einsum('cd,cd...->c...d', w, obs[self.indices]).mean(axis=0) / w.mean(axis=0))
    
    def __getattr__(self, name):
        
        with Timer(logger.info, f'Bootstrapping {name}', per=len(self)):

            try:
                forward = getattr(self.Ensemble, name)
            except:
                raise AttributeError(f"... and so 'Bootstrap' object has no attribute '{name}'")

            self.__dict__[name] = self._resample(forward)
            return self.__dict__[name]
    def estimate(self, observable):
        r'''
        Parameters
        ----------
        observable: string
            Name of the observable or derived quantity

        Returns
        -------
        tuple:
            A tuple with the central value and uncertainty estimate for the observable.  Need not be scalars, if the observable has other indices, the pieces of the tuples have those indices.
        '''
        o = getattr(self, observable)
        return (np.mean(o), np.std(o))
