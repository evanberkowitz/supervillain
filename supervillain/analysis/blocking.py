#!/usr/bin/env python

import numpy as np

import supervillain
from supervillain.batch import Batch, _broadcast_over_draws
from supervillain.h5 import ReadWriteable
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)

def _telescope(blocked, weight):
    r'''
    A block's value: :math:`\langle wO\rangle_b` divided by the block's own
    average weight :math:`\langle w\rangle_b`.

    A block every configuration of which weighs zero --- ordinary once the logs
    span a few hundred, since exp() underflows long before that --- has
    :math:`\langle wO\rangle_b = 0` and :math:`\langle w\rangle_b = 0`.  Its
    value is genuinely undefined, but its *contribution* is not: paired with a
    zero weight it is zero, and it must stay a number to remain so.  Dividing
    would make it nan, and a single nan block turns every estimate it is averaged
    into --- the whole bootstrap --- into nan.
    '''
    weight = np.expand_dims(Batch.as_array(weight), axis=tuple(range(1, blocked.ndim)))
    return np.divide(blocked, weight, out=np.zeros_like(blocked), where=(weight != 0))


class Blocking(ReadWriteable):
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
        self.weight = Batch.as_array(ensemble.weight)[self.drop:].reshape(-1, self.width).mean(axis=1)
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
        r'''The per-block value of the observable, :math:`\langle wO\rangle_b / \langle w\rangle_b`.

        Dividing by the block's own average weight is what makes a block behave
        like a configuration: paired with :attr:`~.Blocking.weight` --- which is
        that same :math:`\langle w\rangle_b` --- the weighted average over blocks
        telescopes back to the average over configurations,

        .. math ::
            \frac{\sum_b \langle w\rangle_b \left(\langle wO\rangle_b/\langle w\rangle_b\right)}{\sum_b \langle w\rangle_b}
            = \frac{\sum_b \langle wO\rangle_b}{\sum_b \langle w\rangle_b}
            = \frac{\langle wO\rangle}{\langle w\rangle},

        so :class:`~.Bootstrap` of a :class:`~.Blocking` gives the same expectation
        value as :class:`~.Bootstrap` of the underlying :class:`~.Ensemble`.
        Returning the un-divided :math:`\langle wO\rangle_b` instead would let
        :class:`~.Bootstrap` apply the weight a second time; simply setting the
        block weights to 1 would be just as wrong in the other direction, leaving
        :math:`\langle wO\rangle` undivided.  With unit weights this is the plain
        block mean, as it always was.'''
        obs = Batch.as_array(obs)
        weight = Batch.as_array(self.Ensemble.weight)
        shape = obs.shape[1:]

        blocked = (
            obs[self.drop:] * np.expand_dims(
                weight[self.drop:],
                axis=tuple(range(1, 1+len(shape)))
            )
        ).reshape(-1, self.width, *shape).mean(axis=1)

        return _telescope(blocked, self.weight)

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

        # The blocked observable is already the per-block weighted mean
        # ⟨wO⟩_b/⟨w⟩_b --- the actual observable value --- so it is plotted as is;
        # the histogram is weighted by the block weight so it shows the physical
        # distribution.  Both reduce to the plain block mean when the weights are
        # all 1.
        data = Batch.as_array(getattr(self, observable))
        weight = _broadcast_over_draws(self.weight, data)
        axes[0].plot(self.index, data, color=color, **history_kwargs)
        axes[1].hist(data, label=histogram_label,
                     orientation='horizontal',
                     bins=bins, density=density,
                     color=color, alpha=alpha,
                     weights=weight,
                     )

    def __getattr__(self, name):

        if name in self.__dict__:
            return self.__dict__[name]

        # The ensemble is reached through __dict__ rather than as self.Ensemble,
        # which would be an attribute lookup of its own and, on an instance that
        # has yet to be given one, a miss --- so this method would call itself
        # until the stack ran out.  copy.copy and copy.deepcopy alike reconstruct
        # an empty instance and ask it for __setstate__, which is exactly that
        # lookup, so copying an ordinary, fully
        # populated Blocking raised RecursionError.
        ensemble = self.__dict__.get('Ensemble')
        if ensemble is None:
            raise AttributeError(
                f'{type(self).__name__} has no {name!r}; it has no ensemble.')

        if name in supervillain.observables:
            forward = getattr(ensemble, name)
            self.__dict__[name] = self._block(forward)

            return self.__dict__[name]

        if name in ensemble.__dict__:
            return ensemble.__dict__[name]

        if name in ('plot_history', 'autocorrelation_time'):
            return getattr(ensemble, name)

        raise AttributeError

