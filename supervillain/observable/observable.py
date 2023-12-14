#!/usr/bin/env python

import numpy as np
from functools import partial
import inspect

import supervillain.ensemble
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)

registry=dict()

class Observable:

    def __init_subclass__(cls, intermediate=False):
        # This registers every subclass that inherits from Observable.
        # Upon registration, Ensemble gets an attribute with the appropriate name.

        name = cls.__name__

        registry[name] = cls

        cls._logger = (logger.debug if name[0] == '_' else logger.info)
        cls._debug  = logger.debug
        cls._logger(f'Observable registered: {name}')

        setattr(supervillain.ensemble.Ensemble, name, cls())

    def __get__(self, obj, objtype=None):
        # The __get__ method is the workhorse of the Descriptor protocol.
        name = self.__class__.__name__
        # Cache:
        if name in obj.__dict__:
            # What's nice about this is that the cache is in the object's dictionary itself,
            # rather than associated with the observable class.  This avoids the issue of a
            # class level cache discussed in https://github.com/evanberkowitz/two-dimensional-gasses/issues/12
            # in that there's no extra reference to the object at all with this strategy.
            # So, when it goes out of scope with no reference, it will be deleted.
            self._debug(f'{name} already cached.')
            return obj.__dict__[name]

        # Just call the measurement and cache the result.
        class_name = obj.Action.__class__.__name__
        try:
            # Observables can have action-dependent implementations
            # and a fall-back default which is convenient for observables which
            # depend simply on others.  For example, a density might not
            # need the field variables but the global charge (or vice-versa).
            try:
                measure = getattr(self, class_name)
            except AttributeError as e:
                if hasattr(self, 'default'):
                    measure = getattr(self, 'default')
                else:
                    raise e from None

            # All observables must take the action as the first argument.
            measure = partial(measure, obj.Action)

            # Observables can depend on field variables and other Observables.
            # We look up the arguments as attributes of the ensemble.
            with Timer(self._logger, f'Measurement of {name}', per=len(obj)):
                obj.__dict__[name]= np.array([
                    measure(*obs)
                    for obs in zip(*[getattr(obj, o) for o in inspect.getfullargspec(measure).args])
                    ])
            return obj.__dict__[name]
        except Exception as exception:
            raise NotImplementedError(f'{name} not implemented for {class_name}') from exception

        raise NotImplementedError()

    def __set__(self, obj, value):
        obj.__dict__[self.__class__.__name__] = value

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        Deciding whether an observable is included in an ensemble's :py:meth:`~.Ensemble.autocorrelation_time` computation is ensemble-dependent.

        For example, if $W=1$ then certain vortex observables are independent of configuration and thus look like
        they have an infinite autocorrelation time.  However, that's expected and not an ergodicity problem.  That's
        real physics!

        So, to decide whether an observable should be included in the ensemble's autocorrelation time requires in general
        evaluating a function on the observable itself and the ensemble.

        By default observables just return ``False`` but observables can override this function to make more clever decisions.
        '''
        return False

class Scalar:

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        Scalars are simple to understand and can be included in the autocorrelation computation.

        Returns ``True``.
        '''
        return True

class Constrained:

    @classmethod
    def autocorrelation(cls, ensemble):
        r'''
        If $W=1$ the observable should not be included in the autocorrelation computation.

        If $W\neq 1$ then use all other considerations to decide.
        '''
        return (ensemble.Action.W != 1) and super().autocorrelation(ensemble)
