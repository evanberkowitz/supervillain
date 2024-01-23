import h5py as h5
from supervillain.h5 import Data

import logging
logger = logging.getLogger(__name__)

####
#### ReadWriteable
####

# A class user-classes should inherit from.
# Those classes will get the .to_h5 and .from_h5 methods and automatically
# be treated with the ReadWriteable strategy.
class ReadWriteable:

    # Each instance gets a to_h5 method that stores the object's __dict__
    # Therefore, cached properties might be saved.
    # Fields whose names start with _ are considered private and hidden one
    # level below in a group called _
    def to_h5(self, group, _top=True):
        r'''
        Write the object as an HDF5 `group`_.
        Member data will be stored as groups or datasets inside ``group``,
        with the same name as the property itself.

        .. note::
            `PEP8`_ considers ``_single_leading_underscores`` as weakly marked for internal use.
            All of these properties will be stored in a single group named ``_``.

        .. _group: https://docs.hdfgroup.org/hdf5/develop/_h5_d_m__u_g.html#subsubsec_data_model_abstract_group
        .. _PEP8: https://peps.python.org/pep-0008/#naming-conventions
        '''

        # If we're at the ``_top`` emit an info to the log, otherwise emit a debug line.
        (logger.info if _top else logger.debug)(f'Saving to_h5 as {group.name}.')

        for attr, value in self.__dict__.items():
            if attr[0] == '_':
                if '_' not in group:
                    private_group = group.create_group('_')
                else:
                    private_group = group['_']
                Data.write(private_group, attr[1:], value)
            else:
                Data.write(group, attr, value)

    # To construct an object from the h5 data, however, we can't start with an object
    # (since we don't know the data to initialize it with).  Instead we need a classmethod
    # and to construct the __dict__ out of the saved data.
    @classmethod
    def from_h5(cls, group, strict=True, _top=True):
        '''
        Construct a fresh object from the HDF5 `group`_.

        .. warning::
            If there is no known strategy for writing data to HDF5, objects will be pickled.

            **Loading pickled data received from untrusted sources can be unsafe.**

            See: https://docs.python.org/3/library/pickle.html for more.

        .. _group: https://docs.hdfgroup.org/hdf5/develop/_h5_d_m__u_g.html#subsubsec_data_model_abstract_group
        '''

        # If we're at the ``_top`` emit an info to the log, otherwise emit a debug line.
        (logger.info if _top else logger.debug)(f'Reading from_h5 {group.name} {"strictly" if strict else "leniently"}.')

        o = cls.__new__(cls)
        for field in group:
            if field == '_':
                for private in group['_']:
                    read = Data.read(group['_'][private], strict)
                    key = f'_{private}'
                    o.__dict__[key] = read
            else:
                o.__dict__[field] = Data.read(group[field], strict)
        return o




def _example_readwrite(cls, filename, original):
    with h5.File(filename, 'w') as f:
        original.to_h5(f.create_group('object'))
        from_disk = cls.from_h5(f['object'])

    return from_disk


def _test():

    import numpy as np

    from pathlib import Path
    test_file = Path(f'{__file__}').parent/'readwriteable.h5'

    class C(ReadWriteable):
        def __init__(self):
            self.string = 'foo'
            self.integer = 17
            self.lst = [1.2, 2.3, 3.4]
            self.tpl = (4.5, 5.6, 6.7)
            self.numpy = np.arange(10)

        def __eq__(self, other):
            if not (self.string == other.string): return False
            if not (self.integer == other.integer): return False
            if not (self.lst == other.lst): return False
            if not (self.tpl == other.tpl): return False
            if not (self.numpy== other.numpy).all(): return False
            return True

    original = C()
    from_disk = _example_readwrite(C, test_file, original)
    test_file.unlink()

    # To fail, try any of these:

    #from_disk.string += 'bar'
    #from_disk.integer += 1
    #from_disk.lst +=[7.8,]
    #from_disk.lst[0] = 7.8
    #from_disk.tpl+=(7.8,)
    #from_disk.numpy *= 2

    if original == from_disk:
        logger.info(f'PASSED')
        return 0
    else:
        logger.error(f'FAILED')
        return 1

if __name__ == '__main__':
    exit(_test())
