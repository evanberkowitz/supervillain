from supervillain.h5 import Data

# A strategy for a python dictionary.
class Dict(Data, name='dict'):

    @staticmethod
    def applies(value):
        return isinstance(value, dict)

    @staticmethod
    def read(group, strict):
        return {key: Data.read(group[key], strict) for key in group}

    @staticmethod
    def write(group, key, value):
        g = group.create_group(key)
        for k, v in value.items():
            if isinstance(k, tuple):
                raise ValueError('With the current Dict tuple-valued keys are not allowed.  See issue #65, https://github.com/evanberkowitz/supervillain/issues/65')
            Data.write(g, k, v)
        return g

