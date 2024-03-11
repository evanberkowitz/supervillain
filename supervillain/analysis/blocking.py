#!/usr/bin/env python

import numpy as np

import supervillain
from supervillain.h5 import ReadWriteable
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)

class Blocking():
    r'''
    Rather than taking :py:meth:`~.Ensemble.every` nth configuration we can instead average (or 'block') the observables from consecutive configurations together.

    Any observable that the underlying ensemble supports can be evaluated in the same way; you can call ``blocking.ObservableOfInterest`` to get the blocked observable of interest.

    Parameters
    ----------
    ensemble: supervillain.Ensemble
        The ensemble to block.
    width: int or 'auto'
        The number of samples that go into each block; if 'auto' set by the ensemble's :py:meth:`~.Ensemble.autocorrelation_time`.
    '''

    def __init__(self, ensemble, width='auto'):
        self.Ensemble = ensemble
        r'''The ensemble underlying the blocking.'''

        if width == 'auto':
            self.width = ensemble.autocorrelation_time()
        else:
            self.width = width
            r'''The width over which to average'''

        cfgs  = len(ensemble)

        self.drop  = cfgs % self.width
        r'''How many configurations are dropped from the start of the ensemble to make the blocking come out evenly.'''
        self.blocks  = (cfgs - self.drop) // self.width
        r'''How many blocks are in the blocking.'''
        self.weight = ensemble.weight[self.drop:].reshape(-1, self.width).mean(axis=1)
        r'''The average weight of each block.'''
        self._block_indices = self.drop+np.arange(len(ensemble)-self.drop).reshape(-1, self.width)
        self.index =  self._block_indices.mean(axis=1)
        r'''The average index of each block.'''
        self.index_stride = ensemble.index_stride * self.width
        r'''The distance between blocks.'''

    def __len__(self):
        r'''
        The number of blocks.
        '''
        return self.blocks

    def _block(self, obs):
        shape = obs.shape[1:]

        return (
            obs[self.drop:] * np.expand_dims(
                self.Ensemble.weight[self.drop:],
                axis=tuple(range(1, 1+len(shape)))
            )
        ).reshape(-1, self.width, *shape).mean(axis=1)

    def plot_history(self, axes, observable, label=None,
                     histogram_label=None,
                     bins=31, density=True,
                     alpha=0.5, color=None,
                     history_kwargs=dict(),
                     ):
        r'''
        .. seealso ::
            :py:meth:`Ensemble.plot_history <~.Ensemble.plot_history>`.
        '''

        if 'label' not in history_kwargs:
            history_kwargs['label']=label

        if histogram_label is None:
            histogram_label=label

        data = getattr(self, observable)
        axes[0].plot(self.index, data, color=color, **history_kwargs)
        axes[1].hist(data, label=histogram_label,
                     orientation='horizontal',
                     bins=bins, density=density,
                     color=color, alpha=alpha,
                     )

    def __getattr__(self, name):

        if name in self.__dict__:
            return self.__dict__[name]

        if name in supervillain.observables:
            forward = getattr(self.Ensemble, name)
            self.__dict__[name] = self._block(forward)

            return self.__dict__[name]

        if name in self.Ensemble.__dict__:
            return self.Ensemble.__dict__[name]

        if name in ('plot_history', 'autocorrelation_time'):
            return getattr(self.Ensemble, name)

        raise AttributeError

