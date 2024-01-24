from supervillain.h5 import Data

# A strategy for a python tuple
class Tuple(Data, name='tuple'):

    @staticmethod
    def applies(value):
        return isinstance(value, tuple)

    @staticmethod
    def read(group, strict):
        return tuple(Data.read(group[str(i)], strict) for i in range(group.attrs['len']))

    @staticmethod
    def write(group, key, value):
        g = group.create_group(key)
        g.attrs['len'] = len(value)
        for i, v in enumerate(value):
            Data.write(g, str(i), v)
        return g

