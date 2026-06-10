#!/usr/bin/env python
r"""
Bridge :class:`~supervillain.lattice.Lattice2D` field layouts and compact :class:`~supervillain.compact.Form`.

Production Villain uses ``(d\phi - 2\pi n)^2``.  Compact uses ``(d\phi + 2\pi n)^2`` with the same
physics under the field rename ``n_{\mathrm{compact}} = -n_{\mathrm{production}}``.
"""

import numpy as np

from supervillain.lattice.compact import Form, Lattice


def compact_lattice(lattice2d):
    r"""Square ``Lattice2D`` → compact ``Lattice(D=2, N=n)``."""
    if lattice2d.nx != lattice2d.nt:
        raise ValueError('compact_lattice requires a square Lattice2D.')
    return Lattice(D=2, N=lattice2d.nx)


def to_form(field, *, degree, lattice2d, dtype=None):
    r"""
    Convert a production-layout field to a compact :class:`Form`.
    """
    L = compact_lattice(lattice2d)
    data = np.asarray(field, dtype=dtype)

    if degree == 0:
        if data.shape != lattice2d.dims:
            raise ValueError(f'0-form shape {data.shape} != {lattice2d.dims}')
        data = data.reshape(1, *lattice2d.dims)
    elif degree == 1:
        if data.shape != (lattice2d.dim,) + lattice2d.dims:
            raise ValueError(f'1-form shape {data.shape} != {(lattice2d.dim,) + lattice2d.dims}')
        data = -data
    else:
        raise ValueError(f'Unsupported degree {degree} for Villain layout bridge.')

    return Form(data, degree=degree, lattice=L, dtype=data.dtype)


def from_form(form):
    r"""Compact :class:`Form` → production-layout ndarray."""
    data = np.asarray(form)
    if form.degree == 0:
        return data[0]
    if form.degree == 1:
        return -data
    raise ValueError(f'Unsupported degree {form.degree} for Villain layout bridge.')
