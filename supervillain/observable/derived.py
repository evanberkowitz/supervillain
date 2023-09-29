#!/usr/bin/env python

import numpy as np
from functools import partial
import inspect

import supervillain.analysis
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)

class DerivedQuantity:

    def __init_subclass__(cls, intermediate=False):
        # This registers every subclass that inherits from DerivedQuantity.
        # Upon registration, Bootstrap gets an attribute with the appropriate name.

        name = cls.__name__

        cls._logger = (logger.debug if name[0] == '_' else logger.info)
        cls._debug  = logger.debug
        cls._logger(f'DerivedQuantity registered: {name}')

        setattr(supervillain.analysis.Bootstrap, name, cls())

    def __get__(self, obj, objtype=None):
        # The __get__ method is the workhorse of the Descriptor protocol.
        name = self.__class__.__name__
        # Cache:
        if name in obj.__dict__:
            # What's nice about this is that the cache is in the object's dictionary itself,
            # rather than associated with the DerivedQuantity class.  This avoids the issue of a
            # class level cache discussed in https://github.com/evanberkowitz/two-dimensional-gasses/issues/12
            # in that there's no extra reference to the object at all with this strategy.
            # So, when it goes out of scope with no reference, it will be deleted.
            self._debug(f'{name} already cached.')
            return obj.__dict__[name]

        # Just call the measurement and cache the result.
        class_name = obj.Ensemble.Action.__class__.__name__
        try:
            # DQs can have action-dependent implementations
            # and a fall-back default which is convenient when dqs depend
            # depend simply on observables or other dqs.  For example, a global
            # charge might just sum up a density, regardless of formulation.
            try:
                measure = getattr(self, class_name)
            except AttributeError as e:
                if hasattr(self, 'default'):
                    measure = getattr(self, 'default')
                else:
                    raise e from None
            
            # All dqs must take the action as the first argument.
            measure = partial(measure, obj.Ensemble.Action)

            # DQs can depend on other DQs and expectation values of Observables.
            # We look up the arguments as attributes of the bootstrap.
            # Since primary Observables are automatically bootstrapped by the Bootstrap
            # object, at this point the loop is over expectation values.
            with Timer(self._logger, f'Bootstrapping of {name}', per=len(obj)):
                obj.__dict__[name]= np.array([
                    measure(*expectation)
                    for expectation in zip(*[getattr(obj, o) for o in inspect.getfullargspec(measure).args])
                ])
            return obj.__dict__[name]
        except:
            raise NotImplementedError(f'Needs an implementation of {name} for {class_name} action.')

        raise NotImplementedError()

    def __set__(self, obj, value):
        setattr(obj, self.name, value)
