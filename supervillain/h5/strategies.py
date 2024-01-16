import numpy as np
#import torch
import io
import pickle
import h5py as h5

from supervillain.h5 import Data
from supervillain.h5 import H5able

import logging
logger = logging.getLogger(__name__)

####
####
####

####
#### Specific Strategies
####

# A default strategy that just using pickling.
class H5ableStrategy(Data, name='h5able'):

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

class IntegerStrategy(Data, name='integer'):

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

class FloatStrategy(Data, name='float'):

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
class NumpyStrategy(Data, name='numpy'):

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

#### Below removed to avoid torch dependency
####
#
## A strategy for torch data.
## The computational graph is severed!
#class TorchStrategy(Data, name='torch'):
#
#    metadata = {
#        'version': torch.__version__.split('+')[0],
#    }
#
#    @staticmethod
#    def applies(value):
#        return isinstance(value, torch.Tensor)
#
#    @staticmethod
#    def read(group, strict):
#        data = group[()]
#        # We would like to read directly onto the default device,
#        # or, if there is a device context manager,
#        #   https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html
#        # the correct device.  Even though there is a torch.set_default_device
#        #   https://pytorch.org/docs/stable/generated/torch.set_default_device.html
#        # there is no corresponding .get_default_device
#        # Instead we infer it
#        device = torch.tensor(0).device
#        # and ship the data to the device.
#        # TODO: Make the device detection as elegant as torch allows.
#        if isinstance(data, np.ndarray):
#            return torch.from_numpy(data).to(device)
#        return torch.tensor(data).to(device)
#
#    @staticmethod
#    def write(group, key, value):
#        # Move the data to the cpu to prevent pickling of GPU tensors
#        # and subsequent incompatibility with CPU-only machines.
#        group[key] = value.cpu().clone().detach().numpy()
#        return group[key]
#
#class ObservableStrategy(TorchStrategy, name='observable'):
#    r'''
#    This strategy is never used by default.
#    It is used by the tdg.ensemble.GrandCanonical to create, extend, and read only segments of configurations and observables.
#    '''
#
#    @staticmethod
#    def applies(value):
#        return False
#
#    @staticmethod
#    def write(group, key, value):
#        logger.debug(f"Writing {group.name}/{key} as observable.")
#        # Markov chains are of unknown length and might be extended, so we should use
#        # a resizable dataset, https://docs.h5py.org/en/stable/high/dataset.html#resizable-datasets
#        # of unknown/unlimited length.
#        shape = (None, )+ value.shape[1:]
#        value = value.clone().detach().cpu().numpy()
#        result = group.create_dataset(key, data=value, shape=value.shape, maxshape=shape, dtype=value.dtype)
#        # Because it is never used through the usual Data.write, we've got to apply the
#        # approprite metadata ourselves.
#        Data._mark_strategy(result, 'observable')
#        Data._mark_metadata(result, ObservableStrategy)
#
#        return result
#
#    @staticmethod
#    def extend(group, key, value):
#        if key not in group:
#            return ObservableStrategy.write(group, key, value)
#
#        shape = group[key].shape
#        extension = value.shape[0]
#        shape = (shape[0] + extension,) + shape[1:]
#
#        logger.debug(f"Extending {group.name}/{key} observable by {extension}.")
#        group[key].resize(shape)
#        group[key][-extension:] = value.clone().detach().cpu().numpy()
#        return group[key]
#
#    @staticmethod
#    def read_only(selection, group, strict):
#
#        logger.debug(f"Reading {selection} of {group.name}.")
#        # Rather than all data, just read the selection.
#        data = group[selection]
#
#        # Then, just as with the torch strategy:
#
#        # We would like to read directly onto the default device,
#        # or, if there is a device context manager,
#        #   https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html
#        # the correct device.  Even though there is a torch.set_default_device
#        #   https://pytorch.org/docs/stable/generated/torch.set_default_device.html
#        # there is no corresponding .get_default_device
#        # Instead we infer it
#        device = torch.tensor(0).device
#        # and ship the data to the device.
#        # TODO: Make the device detection as elegant as torch allows.
#        if isinstance(data, np.ndarray):
#            return torch.from_numpy(data).to(device)
#        return torch.tensor(data).to(device)
#
#class TorchSizeStrategy(Data, name='torch.Size'):
#
#    metadata = {
#        'version': torch.__version__.split('+')[0],
#    }
#
#    @staticmethod
#    def applies(value):
#        return isinstance(value, torch.Size)
#
#    @staticmethod
#    def read(group, strict):
#        return torch.Size(group[()])
#
#    @staticmethod
#    def write(group, key, value):
#        group[key] = value
#        return group[key]
#
#class TorchObjectStrategy(Data, name='torch.object'):
#
#    metadata = {
#        'version': torch.__version__.split('+')[0],
#    }
#
#    @staticmethod
#    def applies(value):
#        return any(
#                isinstance(value, torchType)
#                for torchType in
#                (
#                    # Things we'd otherwise want to read and write with torch.save:
#                    torch.distributions.Distribution,
#                )
#                )
#
#    @staticmethod
#    def read(group, strict):
#        device = torch.tensor(0).device
#        return torch.load(io.BytesIO(group[()]), map_location=device)
#
#    @staticmethod
#    def write(group, key, value):
#        f = io.BytesIO()
#        torch.save(value, f)
#        group[key] = f.getbuffer()
#        return group[key]
#
####
#### Above removed to avoid torch dependency.

# A strategy for a python dictionary.
class DictionaryStrategy(Data, name='dict'):

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
                raise ValueError('With the current DictionaryStrategy tuple-valued keys are not allowed.  See issue #65, https://github.com/evanberkowitz/supervillain/issues/65')
            Data.write(g, k, v)
        return g

# A strategy for a python list.
class ListStrategy(Data, name='list'):

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


