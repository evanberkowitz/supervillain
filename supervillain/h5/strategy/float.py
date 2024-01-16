from supervillain.h5 import Data

class Float(Data, name='float'):

    @staticmethod
    def applies(value):
        return isinstance(value, float)

    @staticmethod
    def read(group, strict):
        return float(group[()])

    @staticmethod
    def write(group, key, value):
        group[key] = value
        return group[key]


