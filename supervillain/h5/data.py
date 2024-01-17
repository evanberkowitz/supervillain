import h5py as h5

import pickle
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Data:
    # Data provides an extensible interface for writing and reading to H5.
    # The H5able class (below) uses Data.{read,write}.
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
        # That metadata, however, cannot be its own Data (because we wouldn't know the
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
        Data._strategies[name] = cls

    @staticmethod
    def write(group, key, value):
        for name, strategy in reversed(Data._strategies.items()):
            try:
                if strategy.applies(value):
                    logger.debug(f"Writing {group.name}/{key} as {name}.")
                    result = strategy.write(group, key, value)
                    Data._mark_strategy(result, name)
                    Data._mark_metadata(result, strategy)
                    break
            except Exception as e:
                logger.error(str(e))
        else: # Wow, a real-life instance of for/else!
            logger.debug(f"Writing {group.name}/{key} by pickling.")
            group[key] = np.void(pickle.dumps(value))
            Data._mark_metadata(group[key], Data)

    @staticmethod
    def read(group, strict=True):
        try:
            name = Data._get_strategy(group)
            logger.debug(f"Reading {group.name} as {name}.")
            strategy = Data._strategies[name]
            Data._check_metadata(group, strategy, strict)
            return strategy.read(group, strict)
        except KeyError:
            logger.debug(f"Reading {group.name} by unpickling.")
            return pickle.loads(group[()])


