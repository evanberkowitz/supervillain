#!/usr/bin/env python

from supervillain.h5 import ReadWriteable, Extendable
import supervillain.h5.extendable as extendable

import logging
logger = logging.getLogger(__name__)

class Configurations(Extendable, ReadWriteable):
    r'''
    A group of configurations has fields (which you can access by doing ``cfgs.field``) and other auxiliary information (one per configuration).

    However, you can also use ``cfgs[step]`` to get a dictionary with keys that correspond to the names of the fields (and the auxiliary information) and associated values.

    If you like you can think of a set of Configurations as a very lightweight barely-featured `pandas DataFrame`_.

    Parameters
    ----------
    dictionary: dict
        A dictionary with ``{key: value}`` pairs, where each value is an array whose first dimension is one per configuration.


    .. _pandas DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    '''
    def __init__(self, dictionary):
        self.fields = dictionary
        
    def __str__(self):
        return str(self.fields)

    def __contains__(self, name):
        return (name in self.fields)

    def __getitem__(self, index):
        r'''
        Parameters
        ----------
        index: fancy indexing
            A subset of numpy `fancy indexing`_ is supported; this selection is used for selecting configurations based on their location in the dataset, rather than their ``index``.
            Some valid choices are ``7``, ``[1,2,3]``, ``slice(1,4)``, ``slice(1,10,2)``.

        Returns
        -------
        one or many configurations:
            If index is an integer, returns a dictionary with key/value pairs for the requested configuration.
            If the index is fancier, return another set of :class:`Configurations`.


        .. _fancy indexing: https://docs.h5py.org/en/stable/high/dataset.html#fancy-indexing
        '''
        t = type(index)
        if t is int:
            return {key: value[index] for key, value in self.fields.items()}
        if t is slice or list:
            return Configurations({key: value[index] for key, value in self.fields.items()})

        raise ValueError(f'Not sure how to select configurations given a {type(index)}.')
    
    def __setitem__(self, index, new):
        r'''
        Parameters
        ----------
        index: fancy indexing
            Index or indices to overwrite.
        new: dictionary or Configurations
            Data to write.
        '''
        for key, value in new.items():
            self.fields[key][index] = value
            

    def __len__(self):
        L = None
        for k, v in self.items():
            try:
                l = len(v)
            except TypeError as e:
                continue

            if L is None:
                L = l
            elif L == l:
                pass
            else:
                raise ValueError("Configurations have no consistent length")

        return L

    def items(self):
        r'''
        Like a dictionary's ``.items()``, iterates over the fields and auxiliary information.
        '''
        return self.fields.items()

    def __getattr__(self, name):
        try:
            return self.fields[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == 'fields':
            self.__dict__['fields'] = value
        elif name in self.fields:
            self.fields[name] = value
        else:
            self.__dict__[name] = value

    def extend_h5(self, group, _top=True):
        logger.info(f'Extending h5 {group.name}.')

        for attr, value in self.items():
            if isinstance(value, Extendable):
                value.extend_h5(group['fields'][attr])
            elif isinstance(value, extendable.array):
                extendable.strategy.extend(group['fields'], attr, value)

    def __ior__(self, value):
        self.fields |= value
        return self

    def copy(self):
        return Configurations(self.fields.copy())
