
import numpy as np
import h5py as h5

from supervillain import _no_op
import supervillain
from supervillain.h5 import Extendable
from supervillain.performance import Timer
from supervillain.analysis.autocorrelation import sample_autocorrelation_time
from supervillain.batch import Batch
import supervillain.h5

import logging
logger = logging.getLogger(__name__)


class Ensemble(Extendable):
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

        .. note ::
            An ensemble assembled this way has no Markov history, but it still gets
            a default :attr:`index` and :attr:`index_stride`, because
            :meth:`~.Ensemble.cut`, :meth:`~.Ensemble.every`, and
            :class:`~.Blocking` all rely on them.  :meth:`~.Ensemble.generate`
            overwrites both with the real chain's labelling.
        '''

        self.configuration = configurations
        self.index = Batch(np.arange(len(configurations)))
        self.index_stride = 1

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
        self.configuration |= generator.inline_observables(steps)
        self.index_stride = index_stride
        self.index = Batch(starting_index + self.index_stride * np.arange(steps))

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

        for line in generator.report().split('\n'):
            logger.info(line)

        return self

    @classmethod
    def continue_from(cls, ensemble, steps, progress=_no_op):
        r'''
        Use the last configuration and generator of ``ensemble`` to produce a new ensemble of ``steps`` configurations.
        
        .. note ::
            Any importance weights ride along automatically.  They are
            ``logWeight_`` fields of the configuration rather than a separate
            array, so the continuation's generator emits its own and
            :meth:`~.Extendable.extend_h5` grows the column like any other, and
            :attr:`weight` normalizes across the grown chain when it is asked
            for.

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

        result = dict()
        for o in observables:
            try:
                result[o] = getattr(self, o)
            except NotImplementedError:
                logger.info(f'{o} is not implemented for {self.Action}')

        return result

    @property
    def measured(self):
        r'''
        A set of strings naming measured observables.
        '''

        return self.__dict__.keys() & supervillain.observables.keys()

    @property
    def weight(self):
        r'''
        Per-configuration importance weight, **derived on access** from any
        ``logWeight_*`` columns the generators emitted (default: all ones).

        A reweighting generator emits its own log-weight contribution as an
        inline observable named ``logWeight_<name>``; the total importance weight is the
        product over contributions, or the exponential of the *summed*
        log-weights.  Working in logs and summing keeps the accumulation stable
        even when individual factors are astronomically small.

                    '''
        # The global ``max`` subtraction is numerical conditioning only --- it
        # cancels in :class:`~.Bootstrap`'s :math:`\langle Ow\rangle/\langle
        # w\rangle` ratio --- and is retaken over whatever configurations are
        # present.  Nothing normalized is persisted, so :meth:`cut`,
        # :meth:`every`, and :meth:`~.Extendable.continue_from` stay
        # self-consistent with no on-disk rewrite.
        cols = sorted(k for k in self.configuration.fields if k.startswith('logWeight_'))
        if not cols:
            return Batch(np.ones(len(self)))
        lw = sum(np.asarray(Batch.as_array(self.configuration.fields[k])) for k in cols)
        if not np.isfinite(lw.max()):
            # Every configuration weighs zero, so <Ow>/<w> is 0/0 and the max
            # subtraction is -inf minus -inf.  Silent nan is the worst outcome.
            raise ValueError(
                'every configuration has zero importance weight, so no weighted '
                'expectation value exists; the reweighting has no overlap with '
                'what it is meant to sample.')
        return Batch(np.exp(lw - lw.max()))

    def autocorrelation_time(self, observables=None, every=False):
        r'''
        Compute the autocorrelation time for the ensemble's measurements.
        However, the autocorrelation time for any observable is only computed if that observable's
        :py:meth:`~.Observable.autocorrelation` is true for this ensemble.
        
        However, if no measurements have been made so that :py:attr:`~.measured` is empty, try every observable with a true :py:meth:`~.Observable.autocorrelation`.
        This may trigger measurement, and is usually what you want; after generation you want to thermalize or decorrelate.

        .. note ::
            The measurement of some observables,
            particularly those for which :py:meth:`~.Observable.autocorrelation` is false for this ensemble,
            is not triggered automatically, unless it is a prerequisite for an observable for :py:meth:`~.Observable.autocorrelation` is true.

        Parameters
        ----------
        observables: ``None`` or iterable of strings naming observables.
            Which observables to consider.  If ``None``, consider all previously-measured observables.

        every: boolean
            If ``True`` returns a dictionary with keys given by observable names and values the computed autocorrelation times.
        '''

        return sample_autocorrelation_time(self, observables=observables, every=every)

    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        r'''
        Read an ensemble back.

        An ensemble stored before :attr:`weight` was derived carries a ``weight``
        alongside its configurations.  It is dropped rather than kept: the property
        shadows it, so it would sit there unread and be written out again by
        :meth:`~.ReadWriteable.to_h5`, propagating a column nothing consults.  Every
        weight the library ever stored was one, so there is nothing to preserve.
        '''
        o = super().from_h5(group, strict=strict, _top=_top)
        o.__dict__.pop('weight', None)
        return o

    def timeseries(self, name):
        r'''
        The measurement of observable ``name`` on every configuration, as a plain
        array.  An :class:`~.Ensemble` holds its configurations, so this is just
        the measurement itself; something that streams its configurations has to
        work harder.
        '''
        return Batch.as_array(getattr(self, name))

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
        # .weight is derived from the (now-sliced) logWeight_* configuration columns.

        for o in self.measured:
            setattr(e, o, getattr(self, o)[start:])

        # A hand-assembled ensemble (from_configurations) has no generator; there
        # is then nothing to carry forward and continue_from is simply unavailable.
        try:
            e.generator = self.generator
        except AttributeError:
            pass

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
        # .weight is derived from the (now-strided) logWeight_* configuration columns.

        for o in self.measured:
            setattr(e, o, getattr(self, o)[::stride])

        # As in cut: no generator to wrap when the ensemble was hand-assembled.
        try:
            e.generator = supervillain.generator.combining.KeepEvery(stride, self.generator, blocked_inline=False)
        except AttributeError:
            pass

        return e

    def plot_history(self, axes, observable, label=None,
                     histogram_label=None,
                     bins=31, density=True,
                     alpha=0.5, color=None,
                     history_kwargs=dict(),
                     ):
        r'''
        .. seealso ::
            :py:meth:`Blocking.plot_history <~.Blocking.plot_history>`.
        '''

        if 'label' not in history_kwargs:
            history_kwargs['label']=label

        if histogram_label is None:
            histogram_label=label

        data = Batch.as_array(getattr(self, observable))
        axes[0].plot(Batch.as_array(self.index), data, color=color, **history_kwargs)
        # The trajectory (left) is the raw chain; the histogram (right) is the
        # observable's DISTRIBUTION, so on a reweighted ensemble it is weighted by
        # .weight (a no-op for the default unit weights) to show the physical, not
        # the merely-sampled, distribution.
        axes[1].hist(data, label=histogram_label,
                     orientation='horizontal',
                     bins=bins, density=density,
                     color=color, alpha=alpha,
                     weights=Batch.as_array(self.weight),
                     )

    def __getattr__(self, name):
        # It is particularly useful to expose fields as ensemble attributes
        # because that helps unify the Observable's application to both
        # fields and other primary observables.
        try:
            return getattr(self.configuration, name)
        except Exception as e:
            raise e from None
