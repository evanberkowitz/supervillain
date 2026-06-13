#!/usr/bin/env python

import numbers

import numpy as np

import supervillain.h5.extendable as extendable

import logging
logger = logging.getLogger(__name__)


def resolve_batch_cls(tag):
    r'''
    Look up the element class for a stored :class:`Batch` column.

    Parameters
    ----------
    tag: str
        Value of ``cls.__batch_tag__`` written to HDF5.  An empty string means
        a plain ndarray column (``cls is None``).

    Returns
    -------
    type or None
        The element class, or ``None`` when ``tag`` is empty.

    Raises
    ------
    ValueError
        If ``tag`` is not registered.
    '''
    if not tag:
        return None
    if tag == 'Form':
        from supervillain.lattice.compact import Form
        return Form
    raise ValueError(f'Unknown Batch element class tag {tag!r}')


class Batch:
    r'''
    A column of MCMC draws stored as :class:`~supervillain.h5.extendable.array`
    with shape ``(draw, …)``.

    ``extendable`` is for storage only; computation uses ``batch[i]`` (a scalar,
    ndarray slice, or ``cls``-wrapped element such as :class:`~supervillain.lattice.Form`).
    For whole-column ndarray operations use :attr:`array`.

    Parameters
    ----------
    draws_or_data:
        If an ``int``, allocate a new zeroed column of that many draws.
        Otherwise wrap existing batched data (draw axis must be 0).
    cls:
        Optional element class (:class:`~supervillain.lattice.Form`, etc.).
        ``None`` for plain ndarray / scalar columns.
    shape:
        Spatial shape when ``cls`` is ``None`` and ``draws_or_data`` is an ``int``.
    dtype:
        Column dtype (default ``float``).
    item_kwargs:
        Column-constant keyword arguments passed to ``cls`` on each draw
        (e.g. ``degree``, ``lattice`` for :class:`~supervillain.lattice.Form`).
    '''

    def __init__(self, draws_or_data, *, cls=None, shape=None, dtype=float, **item_kwargs):
        if isinstance(draws_or_data, numbers.Integral) and not isinstance(draws_or_data, bool):
            draws = int(draws_or_data)
            if cls is not None:
                spatial = cls.spatial_shape(**item_kwargs)
            elif shape is None:
                raise ValueError('Batch(draws, …) requires shape= when cls is None.')
            else:
                spatial = shape
            arr = np.zeros((draws,) + spatial, dtype=dtype)
        else:
            arr = np.asarray(draws_or_data, dtype=dtype)

        self._data = self._as_extendable(arr)
        self.cls = cls
        r'''The element class used to wrap each draw, or ``None`` for plain arrays.'''
        self.dtype = self._data.dtype
        r'''The column dtype.'''
        self._item_kwargs = item_kwargs
        r'''Column-constant keyword arguments passed to ``cls`` when indexing a single draw.'''

    @staticmethod
    def _as_extendable(arr):
        r'''
        Ensure ``arr`` is stored as :class:`~supervillain.h5.extendable.array`.
        '''
        if isinstance(arr, extendable.array):
            return arr
        return extendable.array(arr)

    @classmethod
    def from_data(cls, data, *, dtype=None, **kwargs):
        r'''
        Construct a :class:`Batch` from existing batched data.

        Parameters
        ----------
        data: array_like
            Data whose zeroth axis is the draw index.
        dtype:
            Optional dtype override when wrapping ``data``.
        kwargs:
            Forwarded to :meth:`__init__` (e.g. ``cls``, ``degree``, ``lattice``).

        Returns
        -------
        Batch
            A batch wrapping ``data``.
        '''
        return cls(data, dtype=dtype, **kwargs)

    @property
    def array(self):
        r'''
        The resizable storage column (``extendable.array``, shape ``(draw, …)``).

        Use when you already have a :class:`Batch`.  Prefer ``batch[i]`` for one
        draw; use :meth:`as_array` when a value might still be a legacy column.
        '''
        return self._data

    @staticmethod
    def as_array(column):
        r'''
        Unwrap a column for NumPy analysis.

        :class:`Batch` → :attr:`~Batch.array`; anything else passes through
        (legacy ``extendable.array``, plain ndarray).  Use at boundaries where
        the static type is unknown — not on attributes known to be ``Batch``.
        '''
        return column.array if isinstance(column, Batch) else column

    @property
    def shape(self):
        r'''
        Returns
        -------
        tuple
            Shape of the underlying storage array, ``(draw, …)``.
        '''
        return self._data.shape

    def __len__(self):
        r'''
        Returns
        -------
        int
            Number of draws (length of axis 0).
        '''
        return len(self._data)

    def __getitem__(self, index):
        r'''
        Parameters
        ----------
        index: int, slice, or tuple
            If an ``int``, return one draw (a scalar, ndarray slice, or ``cls`` instance).
            If a ``slice``, return a new :class:`Batch` sharing metadata.
            Otherwise delegate fancy indexing to the underlying storage array.

        Returns
        -------
        scalar, ndarray, Form, or Batch
            One element, a sub-batch, or a indexed view of the storage array.
        '''
        if type(index) is int:
            sliced = self._data[index]
            if self.cls is None:
                return sliced
            return self.cls(sliced, dtype=self.dtype, **self._item_kwargs)
        if type(index) is slice:
            return Batch(
                self._data[index],
                cls=self.cls,
                dtype=self.dtype,
                **self._item_kwargs,
            )
        return self._data[index]

    def __setitem__(self, index, item):
        r'''
        Parameters
        ----------
        index:
            Draw index or indices to overwrite (numpy indexing).
        item:
            Value to store, coerced to :attr:`dtype`.
        '''
        self._data[index] = self._coerce_item(item)

    def _coerce_item(self, item):
        r'''
        Coerce a generator return value to the column :attr:`dtype` for storage.
        '''
        return np.asarray(item, dtype=self.dtype)

    def __iter__(self):
        r'''
        Yield one draw per step (same objects as ``batch[i]`` for integer ``i``).
        '''
        for i in range(len(self)):
            yield self[i]

    def extend_h5(self, group):
        r'''
        Append this batch's draws to an on-disk column.

        The ``group`` must be an HDF5 group produced by the ``batch`` write
        strategy (it must contain a resizable ``data`` dataset).

        Parameters
        ----------
        group: h5py.Group
            Group storing the batch column to extend.
        '''
        from supervillain.h5.extendable import strategy as extendable_strategy

        extendable_strategy.extend(group, 'data', self._data)

    def __repr__(self):
        r'''
        Returns
        -------
        str
            A short summary of shape, element class, and dtype.
        '''
        cls_name = self.cls.__name__ if self.cls is not None else 'ndarray'
        return f'Batch(shape={self.shape}, cls={cls_name}, dtype={self.dtype})'
