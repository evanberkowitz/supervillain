from supervillain.h5 import Data

# A strategy for a python list.
class List(Data, name='list'):

    @staticmethod
    def applies(value):
        return isinstance(value, list)

    @staticmethod
    def read(group, strict):
        return [Data.read(group[str(i)], strict) for i in range(group.attrs['len'])]

    @staticmethod
    def write(group, key, value):
        g = group.create_group(key)
        g.attrs['len'] = len(value)
        for i, v in enumerate(value):
            Data.write(g, str(i), v)
        return g

