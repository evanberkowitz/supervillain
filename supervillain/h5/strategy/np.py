from supervillain.h5 import Data

import numpy as np

class ndarray(Data, name='numpy'):

    metadata = {
        'version': np.__version__,
    }

    @staticmethod
    def applies(value):
        return isinstance(value, np.ndarray)

    @staticmethod
    def read(group, strict):
        return group[()]

    @staticmethod
    def write(group, key, value):
        group[key] = value
        return group[key]


