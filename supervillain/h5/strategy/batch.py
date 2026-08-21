import pickle

import numpy as np

from supervillain.h5 import Data
import supervillain.h5.extendable as extendable
from supervillain.h5.extendable import strategy as extendable_strategy
from supervillain.batch import Batch, resolve_batch_cls


class BatchStrategy(Data, name='batch'):

    metadata = {}

    @staticmethod
    def applies(value):
        return isinstance(value, Batch)

    @staticmethod
    def write(group, key, value):
        g = group.create_group(key)
        extendable_strategy.write(g, 'data', value.array)
        tag = getattr(value.cls, '__batch_tag__', '') if value.cls is not None else ''
        g.attrs['H5Batch_cls'] = tag
        g.attrs['H5Batch_dtype'] = str(value.dtype)
        g.attrs['H5Batch_item_kwargs'] = np.void(pickle.dumps(value._item_kwargs))
        return g

    @staticmethod
    def read(group, strict):
        tag = group.attrs.get('H5Batch_cls', '')
        if isinstance(tag, bytes):
            tag = tag.decode()
        cls = resolve_batch_cls(tag) if tag else None
        data = extendable.array(extendable_strategy.read(group['data'], strict))
        item_kwargs = pickle.loads(group.attrs['H5Batch_item_kwargs'].tobytes())
        return Batch(data, cls=cls, dtype=data.dtype, **item_kwargs)
