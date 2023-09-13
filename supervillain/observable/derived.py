#!/usr/bin/env python

import numpy as np
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
            measure = getattr(self, class_name)
            with Timer(self._logger, f'Bootstrapping of {name}', per=len(obj)):
                # The main difference between a DerivedQuantity and an Observable is that a DQ needs
                # expectation values---attributes of the Bootstrap, not just the field variables in
                # configurations.  Therefore, the arguments might other than eg. phi and n (Villain) or m (Worldline)
                #
                # A DerivedQuantity method always gets the action as the first parameter (so that if the lattice is needed for
                # correlation, or if κ is needed, or whatever, it is available) and subsequent parameters can be 
                #   Observables         (which already get forwarded to the Ensemble observable by Bootstrap.__getattr__) or 
                #   DerivedQuantities   (which if not already evaluated will be evaluated now).
                #
                #
                # So, dumb example DerivedQuantities might look like
                #
                # class TwiceInternalEnergyDensity(DerivedQuantity):
                #
                #     @staticmethod
                #     def Villain(Action, InternalEnergyDensity):
                #         return 2*InternalEnergyDensity
                #
                # class ThriceInternalEnergyDensity(DerivedQuantity):
                #
                #     @staticmethod
                #     def Villain(Action, InternalEnergyDensity, TwiceInternalEnergyDensity):
                #         return InternalEnergyDensity + TwiceInternalEnergyDensity
                #
                # where (aside from the fact that we could always just multiply by 2 or 3) we compute 3*u in a funny way,
                # by summing u + 2u, just to show that one DerivedQuantity can depend on another DerivedQuantity.
                # If you construct some horrible loop that cannot be resolved, god save you.
                #
                # The user would NOT have to evaluate TwiceInternalEnergyDensity themselves before evaluating Thrice.
                # The inspect module is used to look at the arguments and evaluate those attributes of the bootstrap,
                # so TwiceInternalEnergyDensity will be evaluated and cached under the hood.
                #
                # NB this requires the parameters' names to EXACTLY MATCH the corresponding Observables or DerivedQuantities!
                obj.__dict__[name]= np.array([
                    measure(obj.Ensemble.Action, *obs)
                    for obs in zip(*[getattr(obj, o) for o in inspect.getfullargspec(measure).args[1:]])
                ])
            return obj.__dict__[name]
        except:
            raise NotImplementedError(f'Needs an implementation of {name} for {class_name} action.')

        raise NotImplementedError()

    def __set__(self, obj, value):
        setattr(obj, self.name, value)
