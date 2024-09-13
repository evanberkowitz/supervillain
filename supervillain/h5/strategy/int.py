from supervillain.h5 import Data

class Integer(Data, name='integer'):

    @staticmethod
    def applies(value):
        return isinstance(value, int)

    @staticmethod
    def read(group, strict):
        return int(group[()])

    @staticmethod
    def write(group, key, value):
        group[key] = value
        return group[key]


