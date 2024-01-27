from supervillain.h5 import Data

class Complex(Data, name='complex'):

    @staticmethod
    def applies(value):
        return isinstance(value, complex)

    @staticmethod
    def read(group, strict):
        return complex(group[()])

    @staticmethod
    def write(group, key, value):
        group[key] = value
        return group[key]


