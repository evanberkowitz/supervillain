import numpy as np

from supervillain.h5.strategy.np import ndarray as base_strategy
from supervillain.h5 import Data, H5able

import logging
logger = logging.getLogger(__name__)

class array(np.ndarray):

    def __new__(cls, input_array):

        obj = np.asarray(input_array).view(cls)

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return


class strategy(base_strategy, name='extendable'):

    @staticmethod
    def applies(value):
        # Annoyingly scalars reduced from extendable arrays might themselves be extendable.
        # Therefore, we also check that they have dimension (and are indeed an array).
        # TODO: Upon the resolution of https://github.com/numpy/numpy/issues/25606
        # we can either adapt the array class or keep this workaround.

        return isinstance(value, array) and value.shape

    @staticmethod
    def write(group, key, value):
        shape = (None, )+ value.shape[1:]
        result = group.create_dataset(key, data=value, shape=value.shape, maxshape=shape, dtype=value.dtype)
        # Because it is never used through the usual Data.write, we've got to apply the
        # approprite metadata ourselves.
        Data._mark_strategy(result, 'extendable')
        Data._mark_metadata(result, strategy)

        return result
    
    @staticmethod
    def read(group, strict):
        return array(super(strategy, strategy).read(group, strict))

    @staticmethod
    def extend(group, key, value):
        if key not in group:
            raise KeyError(f'{key} not found in {group}')

        shape = group[key].shape
        extension = value.shape[0]
        shape = (shape[0] + extension,) + shape[1:]

        logger.debug(f"Extending {group.name}/{key} by {extension}.")
        group[key].resize(shape)
        group[key][-extension:] = value
        return group[key]

class H5Extendable(H5able):

    def extend_h5(self, group, _top=True):
        logger.info(f'Extending h5 {group.name}.')

        for attr, value in self.__dict__.items():
            if isinstance(value, H5Extendable):
                value.extend_h5(group[attr])
            elif isinstance(value, array):
                strategy.extend(group, attr, value)
