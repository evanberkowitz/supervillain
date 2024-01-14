
import numpy as np
import h5py as h5

from supervillain import _no_op
import supervillain
from supervillain.h5 import H5able
from supervillain.performance import Timer
from supervillain.analysis import autocorrelation_time
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

        self.configuration = configurations

        return self

    def generate(self, steps, generator, start='cold', progress=_no_op, starting_index=0, index_stride=1):
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
                Defaults to no progress reporting.  Must accept a `desc` keyword argument.
            starting_index: int
                An ensemble has a ``.index`` which is an array of regularly-spaced integers labeling the configurations; this sets the lower value.
            index_stride: int
                The increment of the ``.index`` for each call of the generator.

        Returns
        -------
            the ensemble itself, so that one can do ``ensemble = GrandCanonical(action).generate(...)``.

        .. _tqdm.tqdm: https://pypi.org/project/tqdm/
        .. _tqdm.notebook: https://tqdm.github.io/docs/notebook/
        '''

        self.configuration = self.Action.configurations(steps)
        self.index_stride = index_stride
        self.index = starting_index + self.index_stride * np.arange(steps)
        self.weight = np.ones(steps)

        if start == 'cold':
            seed = self.Action.configurations(1)[0]
        elif type(start) is dict:
            seed = start
        else:
            raise ValueError('Not sure how to transform a {type(start)} into a starting configuration.')

        with Timer(logger.info, f'Generation of {steps} configurations', per=steps):

            self.configuration[0] = generator.step(seed)

            for mcmc_step in progress(range(1,steps), desc='Generation'):
                self.configuration[mcmc_step] = generator.step(self.configuration[mcmc_step-1])

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

            progress:
                As in :py:meth:`~.generate`.

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
            last      = e.configuration[-1]
            index     = e.index[-1] + e.index_stride
        except:
            raise ValueError('The ensemble must provide a generator, an Action, and at least one configuration.')

        return Ensemble(action).generate(steps, generator, last, progress=progress, starting_index=index, index_stride=e.index_stride)

    def __len__(self):
        return len(self.configuration)

    def measure(self, observables=None):
        r'''
        If ``observables`` is None, measure every known primary observable on this ensemble.
        Otherwise measure only those observables named.
        If an observable is already computed, no new computation occurs.

        Parameters
        ----------
        observables: ``None`` or iterable of strings naming observables.
            Observables to compute on this ensemble.

        Returns
        -------
        dict:
            Keys are observable names, values are the measurements.
        '''

        if observables is None:
            observables = supervillain.observables.keys()

        return {o: getattr(self, o) for o in observables}

    @property
    def measured(self):
        r'''
        A set of strings naming measured observables.
        '''

        return self.__dict__.keys() & supervillain.observables.keys()

    def autocorrelation_time(self, observables=None, every=False):
        r'''
        Compute the autocorrelation time for the ensemble's measurements.
        However, the autocorrelation time for any observable is only computed if that observable's
        :py:meth:`~.Observable.autocorrelation` is true for this ensemble.
        
        .. note ::
            This does *not* trigger the measurement of any observable unless explicitly asked for in the ``observables`` parameter.

        Parameters
        ----------
        observables: ``None`` or iterable of strings naming observables.
            Which observables to consider.  If ``None``, consider all previously-measured observables.

        every: boolean
            If ``True`` returns a dictionary with keys given by observable names and values the computed autocorrelation times.
        '''

        if observables is None:
            observables = self.measured

        auto = {
                name: autocorrelation_time(getattr(self, name))
                for name in observables
                if supervillain.observables[name].autocorrelation(self)
                }
        if every:
            return auto
        else:
            return max(auto.values())

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
        e = Ensemble(self.Action).from_configurations(self.configuration[start:])
        e.index = self.index[start:]
        e.index_stride = self.index_stride
        e.weight = self.weight[start:]

        for o in self.measured:
            setattr(e, o, getattr(self, o)[start:])

        e.generator = self.generator

        return e

    def every(self, stride):
        r'''
        Good for decorrelation.

        The generator is wrapped in :class:`~.KeepEvery` so that :py:meth:`~.continue_from` produces a strided follow-on ensemble.

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

        e = Ensemble(self.Action).from_configurations(self.configuration[::stride])
        e.index = self.index[::stride]
        e.index_stride = self.index_stride * stride
        e.weight = self.weight[::stride]

        for o in self.measured:
            setattr(e, o, getattr(self, o)[::stride])

        e.generator = supervillain.generator.combining.KeepEvery(stride, self.generator)

        return e

    def plot_history(self, axes, observable, label=None,
                     history_label=None,
                     histogram_label=None,
                     bins=31, density=True,
                     alpha=0.5, color=None,
                     ):

        if history_label is None:
            history_label=label
        if histogram_label is None:
            histogram_label=label

        data = getattr(self, observable)
        axes[0].plot(self.index, data, color=color, label=history_label)
        axes[1].hist(data, label=histogram_label,
                     orientation='horizontal',
                     bins=bins, density=density,
                     color=color, alpha=alpha,
                     )
 
    def __getattr__(self, name):
        # It is particularly useful to expose fields as ensemble attributes
        # because that helps unify the Observable's application to both
        # fields and other primary observables.
        try:
            return getattr(self.configuration, name)
        except Exception as e:
            raise e
