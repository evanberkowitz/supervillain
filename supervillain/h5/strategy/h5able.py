from supervillain.h5 import Data

import supervillain.h5
import numpy as np
import pickle

# A default strategy that just using pickling.
class H5able(Data, name='h5able'):

    metadata = {}

    @staticmethod
    def applies(value):
        return isinstance(value, supervillain.h5.H5able)

    @staticmethod
    def read(group, strict):
        cls = pickle.loads(group.attrs['H5able_class'][()])
        return cls.from_h5(group, strict, _top=False)

    def write(group, key, value):
        g = group.create_group(key)
        g.attrs['H5able_class'] = np.void(pickle.dumps(type(value)))
        value.to_h5(g, _top=False)
        return g


