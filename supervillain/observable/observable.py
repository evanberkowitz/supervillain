#!/usr/bin/env python

import numpy as np

import supervillain.ensemble
from supervillain.performance import Timer

import logging
logger = logging.getLogger(__name__)


class Observable:

    def __init_subclass__(cls, name='', intermediate=False):
        # This registers every subclass that inherits from Observable.
        # Upon registration, Ensemble gets an attribute with the appropriate name.

        if name == '':
            cls.name = cls.__name__
        else:
            cls.name = name

        cls._logger = (logger.debug if cls.name[0] == '_' else logger.info)
        cls._debug  = logger.debug
        cls._logger(f'Observable registered: {cls.name}')

        setattr(supervillain.ensemble.Ensemble, cls.name, cls())

    def __set_name__(self, owner, name):
        self.name  = name

    def __get__(self, obj, objtype=None):
        # The __get__ method is the workhorse of the Descriptor protocol.

        # Cache:
        if self.name in obj.__dict__:
            # What's nice about this is that the cache is in the object's dictionary itself,
            # rather than associated with the observable class.  This avoids the issue of a
            # class level cache discussed in https://github.com/evanberkowitz/two-dimensional-gasses/issues/12
            # in that there's no extra reference to the object at all with this strategy.
            # So, when it goes out of scope with no reference, it will be deleted.
            self._debug(f'{self.name} already cached.')
            return obj.__dict__[self.name]

        # Just call the measurement and cache the result.
        class_name = obj.Action.__class__.__name__
        try:
            with Timer(self._logger, f'Measurement of {self.name}', per=len(obj)):
                obj.__dict__[self.name]= np.array([getattr(self, class_name)(obj.Action, **o) for o in obj.configurations])
            return obj.__dict__[self.name]
        except:
            raise NotImplemented(f'{self.name} not implemented for {class_name}')

        raise NotImplemented()

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

