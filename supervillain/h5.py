import numpy as np
import torch
import io
import pickle
import h5py as h5

import logging
logger = logging.getLogger(__name__)

####
####
####

class H5Data:
    # H5Data provides an extensible interface for writing and reading to H5.
    # The H5able class (below) uses H5Data.{read,write}.
    # No object instance is needed, so all methods are @staticmethod

    # However, we need a class-level registry to store strategies.
    # A strategy is a way to read and write objects to hdf5.
    # Different strategies can be used for numpy, torch, or other objects.
    _strategies = {}
    metadata = {}

    # Before reading data, we need to be able to decide the right strategy.
    # So, we need a way to write the strategy into the HDF5 metadata
    @staticmethod
    def _mark_strategy(group, name):
        group.attrs['H5Data_strategy'] = name
    # and read that metadata back.
    @staticmethod
    def _get_strategy(group):
        return group.attrs['H5Data_strategy']

    # Different strategies might have their own metadata, which can be used for some
    # amount of reproducibility and data provenance.
    @staticmethod
    def _mark_metadata(group, strategy):
        # That metadata, however, cannot be its own H5Data (because we wouldn't know the
        # correct strategy to read it).  Therefore we just pickle it up.
        group.attrs['H5Data_metadata'] = np.void(pickle.dumps(strategy.metadata))
    # When we read, we check the written metadata to the strategy's current metadata,
    # so we know if something has changed.
    #
    # We default to a strict behavior, which raises an exception if the metadata differs.
    @staticmethod
    def _check_metadata(group, strategy, strict=True):
        metadata = pickle.loads(group.attrs['H5Data_metadata'])
        for key, value in metadata.items():
            try:
                current = strategy.metadata[key]
                if value != current:
                    message = f"Version mismatch for {group.name}.  Stored with '{value}' but currently use '{current}'"
                    if strict:
                        raise ValueError(message)
                    else:
                        logger.warn(message)
            except KeyError:
                pass
    
    # Specific strategies will inherit from this class.
    # When they're declared they should be added to the registry.
    def __init_subclass__(cls, name):
        H5Data._strategies[name] = cls

    @staticmethod
    def write(group, key, value):
        for name, strategy in H5Data._strategies.items():
            try:
                if strategy.applies(value):
                    logger.debug(f"Writing {group.name}/{key} as {name}.")
                    result = strategy.write(group, key, value)
                    H5Data._mark_strategy(result, name)
                    H5Data._mark_metadata(result, strategy)
                    break
            except Exception as e:
                logger.error(str(e))
        else: # Wow, a real-life instance of for/else!
            logger.debug(f"Writing {group.name}/{key} by pickling.")
            group[key] = np.void(pickle.dumps(value))
            H5Data._mark_metadata(group[key], H5Data)

    @staticmethod
    def read(group, strict=True):
        try:
            name = H5Data._get_strategy(group)
            logger.debug(f"Reading {group.name} as {name}.")
            strategy = H5Data._strategies[name]
            H5Data._check_metadata(group, strategy, strict)
            return strategy.read(group, strict)
        except KeyError:
            logger.debug(f"Reading {group.name} by unpickling.")
            return pickle.loads(group[()])

####
#### Specific Strategies
####

# A default strategy that just using pickling.
class H5ableStrategy(H5Data, name='h5able'):

    metadata = {}

    @staticmethod
    def applies(value):
        return isinstance(value, H5able)

    @staticmethod
    def read(group, strict):
        cls = pickle.loads(group.attrs['H5able_class'][()])
        return cls.from_h5(group, strict, _top=False)

    def write(group, key, value):
        g = group.create_group(key)
        g.attrs['H5able_class'] = np.void(pickle.dumps(type(value)))
        value.to_h5(g, _top=False)
        return g

class IntegerStrategy(H5Data, name='integer'):

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

class FloatStrategy(H5Data, name='float'):

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

# A strategy for numpy data.
class NumpyStrategy(H5Data, name='numpy'):

    metadata = {
        'version': np.__version__,
    }

    @staticmethod
    def applies(value):
        return isinstance(value, np.ndarray)

    @staticmethod
    def read(group, strict):
        return group[()]

    @staticmethod
    def write(group, key, value):
        group[key] = value
        return group[key]

# A strategy for torch data.
# The computational graph is severed!
class TorchStrategy(H5Data, name='torch'):

    metadata = {
        'version': torch.__version__.split('+')[0],
    }

    @staticmethod
    def applies(value):
        return isinstance(value, torch.Tensor)

    @staticmethod
    def read(group, strict):
        data = group[()]
        # We would like to read directly onto the default device,
        # or, if there is a device context manager,
        #   https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html
        # the correct device.  Even though there is a torch.set_default_device
        #   https://pytorch.org/docs/stable/generated/torch.set_default_device.html
        # there is no corresponding .get_default_device
        # Instead we infer it
        device = torch.tensor(0).device
        # and ship the data to the device.
        # TODO: Make the device detection as elegant as torch allows.
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        return torch.tensor(data).to(device)

    @staticmethod
    def write(group, key, value):
        # Move the data to the cpu to prevent pickling of GPU tensors
        # and subsequent incompatibility with CPU-only machines.
        group[key] = value.cpu().clone().detach().numpy()
        return group[key]

class ObservableStrategy(TorchStrategy, name='observable'):
    r'''
    This strategy is never used by default.
    It is used by the tdg.ensemble.GrandCanonical to create, extend, and read only segments of configurations and observables.
    '''

    @staticmethod
    def applies(value):
        return False

    @staticmethod
    def write(group, key, value):
        logger.debug(f"Writing {group.name}/{key} as observable.")
        # Markov chains are of unknown length and might be extended, so we should use
        # a resizable dataset, https://docs.h5py.org/en/stable/high/dataset.html#resizable-datasets
        # of unknown/unlimited length.
        shape = (None, )+ value.shape[1:]
        value = value.clone().detach().cpu().numpy()
        result = group.create_dataset(key, data=value, shape=value.shape, maxshape=shape, dtype=value.dtype)
        # Because it is never used through the usual H5Data.write, we've got to apply the
        # approprite metadata ourselves.
        H5Data._mark_strategy(result, 'observable')
        H5Data._mark_metadata(result, ObservableStrategy)

        return result

    @staticmethod
    def extend(group, key, value):
        if key not in group:
            return ObservableStrategy.write(group, key, value)

        shape = group[key].shape
        extension = value.shape[0]
        shape = (shape[0] + extension,) + shape[1:]

        logger.debug(f"Extending {group.name}/{key} observable by {extension}.")
        group[key].resize(shape)
        group[key][-extension:] = value.clone().detach().cpu().numpy()
        return group[key]

    @staticmethod
    def read_only(selection, group, strict):

        logger.debug(f"Reading {selection} of {group.name}.")
        # Rather than all data, just read the selection.
        data = group[selection]

        # Then, just as with the torch strategy:

        # We would like to read directly onto the default device,
        # or, if there is a device context manager,
        #   https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html
        # the correct device.  Even though there is a torch.set_default_device
        #   https://pytorch.org/docs/stable/generated/torch.set_default_device.html
        # there is no corresponding .get_default_device
        # Instead we infer it
        device = torch.tensor(0).device
        # and ship the data to the device.
        # TODO: Make the device detection as elegant as torch allows.
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        return torch.tensor(data).to(device)

class TorchSizeStrategy(H5Data, name='torch.Size'):

    metadata = {
        'version': torch.__version__.split('+')[0],
    }

    @staticmethod
    def applies(value):
        return isinstance(value, torch.Size)

    @staticmethod
    def read(group, strict):
        return torch.Size(group[()])

    @staticmethod
    def write(group, key, value):
        group[key] = value
        return group[key]

class TorchObjectStrategy(H5Data, name='torch.object'):

    metadata = {
        'version': torch.__version__.split('+')[0],
    }

    @staticmethod
    def applies(value):
        return any(
                isinstance(value, torchType)
                for torchType in
                (
                    # Things we'd otherwise want to read and write with torch.save:
                    torch.distributions.Distribution,
                )
                )

    @staticmethod
    def read(group, strict):
        device = torch.tensor(0).device
        return torch.load(io.BytesIO(group[()]), map_location=device)

    @staticmethod
    def write(group, key, value):
        f = io.BytesIO()
        torch.save(value, f)
        group[key] = f.getbuffer()
        return group[key]

# A strategy for a python dictionary.
class DictionaryStrategy(H5Data, name='dict'):

    @staticmethod
    def applies(value):
        return isinstance(value, dict)

    @staticmethod
    def read(group, strict):
        return {key: H5Data.read(group[key], strict) for key in group}

    @staticmethod
    def write(group, key, value):
        g = group.create_group(key)
        for k, v in value.items():
            H5Data.write(g, k, v)
        return g

# A strategy for a python list.
class ListStrategy(H5Data, name='list'):

    @staticmethod
    def applies(value):
        return isinstance(value, list)

    @staticmethod
    def read(group, strict):
        return [H5Data.read(group[str(i)], strict) for i in range(group.attrs['len'])]

    @staticmethod
    def write(group, key, value):
        g = group.create_group(key)
        g.attrs['len'] = len(value)
        for i, v in enumerate(value):
            H5Data.write(g, str(i), v)
        return g

####
#### H5able
####

# A class user-classes should inherit from.
# Those classes will get the .to_h5 and .from_h5 methods and automatically
# be treated with the H5ableData strategy.
class H5able:

    def __init__(self):
        super().__init__()

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
                H5Data.write(private_group, attr[1:], value)
            else:
                H5Data.write(group, attr, value)

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
                    read = H5Data.read(group['_'][private], strict)
                    key = f'_{private}'
                    o.__dict__[key] = read
            else:
                o.__dict__[field] = H5Data.read(group[field], strict)
        return o
