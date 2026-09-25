from supervillain.h5 import Data

import numpy as np

# Standalone barycubic forms.
#
# A BarycubicForm is an ndarray subclass, so without this strategy the numpy strategy would
# capture it and read it back as a plain array, losing its degree, its primal/dual kind, and
# its lattice.  Data.write tries strategies in reverse registration order, and this module is
# imported after supervillain.h5.strategy.np, so it is consulted first.
#
# The lattice module is imported lazily: supervillain.lattice itself depends on supervillain.h5,
# so importing it here at module load would be circular.  applies() therefore recognizes the
# forms by their batch tag rather than by isinstance.

_TAGS = ('BarycubicForm', 'BarycubicDualForm')


class BarycubicFormStrategy(Data, name='barycubic_form'):
    r'''
    HDF5 strategy for a standalone :class:`~supervillain.lattice.barycubic.BarycubicForm` or
    :class:`~supervillain.lattice.barycubic.BarycubicDualForm`.

    The form is written as one dataset holding its data, with attributes recording the class
    tag, the ``degree``, the ``dual`` flag, the primal degree, and the lattice's ``D`` and ``N``.
    Reading rebuilds the lattice and returns the same class.  Stored metadata that contradicts
    itself (a dual flag that does not match the class, a primal degree that does not match the
    degree, a shape that does not match the degree and lattice) raises ``ValueError`` rather than
    returning a mislabelled form.

    The strategy is registered after the NumPy strategy, so ``Data.write`` tries it first;
    otherwise a form, being an ndarray, would be written as a plain array and read back without
    its degree or lattice.
    '''

    metadata = {}

    @staticmethod
    def applies(value):
        # Only a whole form: a slice or reshape of a form is still an instance of the class but
        # has no well-defined degree, and falls through to the NumPy strategy as a plain array,
        # as views of hypercubic Forms do.
        if not isinstance(value, np.ndarray) or getattr(type(value), '__batch_tag__', None) not in _TAGS:
            return False
        lattice, degree = getattr(value, 'lattice', None), getattr(value, 'degree', None)
        if lattice is None or degree is None:
            return False
        try:
            return value.shape == type(value).spatial_shape(degree=degree, lattice=lattice)
        except (ValueError, TypeError, AttributeError, KeyError):
            return False

    @staticmethod
    def write(group, key, value):
        # Everything is computed and checked before anything is created, so a failure cannot
        # leave a half-written dataset behind (Data.write would then try the next strategy).
        if not BarycubicFormStrategy.applies(value):
            raise ValueError(f'{type(value).__name__} is not a whole barycubic form')
        attributes = {
            'barycubic_class': type(value).__batch_tag__,
            'degree': int(value.degree),
            'dual': bool(value.dual),
            'primal_degree': int(value.lattice.D - value.degree if value.dual else value.degree),
            'D': int(value.lattice.D),
            'N': int(value.lattice.N),
        }
        data = np.ascontiguousarray(value).view(np.ndarray)
        dataset = group.create_dataset(key, data=data)
        for name, attribute in attributes.items():
            dataset.attrs[name] = attribute
        return dataset

    @staticmethod
    def read(group, strict):
        from supervillain.lattice.barycubic import BarycubicLattice, BarycubicForm, BarycubicDualForm

        # Data.read treats a KeyError as "no strategy" and falls back to unpickling, so missing
        # metadata must surface as a ValueError instead.
        try:
            tag = group.attrs['barycubic_class']
            degree = int(group.attrs['degree'])
            dual = bool(group.attrs['dual'])
            primal_degree = int(group.attrs['primal_degree'])
            D = int(group.attrs['D'])
            N = int(group.attrs['N'])
        except KeyError as missing:
            raise ValueError(f'{group.name} lacks barycubic form metadata {missing}') from None
        if isinstance(tag, bytes):
            tag = tag.decode()

        cls = {'BarycubicForm': BarycubicForm, 'BarycubicDualForm': BarycubicDualForm}.get(tag)
        if cls is None:
            raise ValueError(f'{group.name}: unknown barycubic form class {tag!r}')
        if dual != cls.dual:
            raise ValueError(f'{group.name}: class {tag} is inconsistent with dual={dual}')
        if primal_degree != (D - degree if dual else degree):
            raise ValueError(f'{group.name}: degree {degree} and primal degree {primal_degree} '
                             f'are inconsistent for a {tag} in D={D}')

        lattice = BarycubicLattice(D, N)
        # The constructor checks the stored shape against (degree, lattice).
        return cls(group[()], degree=degree, lattice=lattice)
