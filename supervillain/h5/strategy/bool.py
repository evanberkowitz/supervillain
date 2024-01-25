from supervillain.h5 import Data

class Boolean(Data, name='bool'):

    @staticmethod
    def applies(value):
        return isinstance(value, bool)

    @staticmethod
    def read(group, strict):
        return bool(group[()])

    @staticmethod
    def write(group, key, value):
        group[key] = value
        return group[key]


