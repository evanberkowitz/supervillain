#!/usr/bin/env python

import numpy as np

import supervillain.ensemble
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)


class Observable:

    def __init_subclass__(cls, intermediate=False):
        # This registers every subclass that inherits from Observable.
        # Upon registration, Ensemble gets an attribute with the appropriate name.

        name = cls.__name__

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
            with Timer(self._logger, f'Measurement of {name}', per=len(obj)):
                obj.__dict__[name]= np.array([getattr(self, class_name)(obj.Action, **o) for o in obj.configurations])
            return obj.__dict__[name]
        except:
            raise NotImplemented(f'{name} not implemented for {class_name}')

        raise NotImplemented()

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

