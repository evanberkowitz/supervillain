#!/usr/bin/env python

r"""A strategy that raises AFTER creating its group (e.g. the Dict strategy on
tuple-valued keys, issue #65) must not wedge ``Data.write``: the partial write
is cleaned up and the value falls back to pickling, so a ``ReadWriteable``
carrying such an attribute (the ``SurfaceWormGas``'s ``wgroups`` is the
production case) still round-trips.
"""

import h5py as h5
import numpy as np
import pytest

from supervillain.h5 import ReadWriteable


class CarriesTupleKeyedDict(ReadWriteable):
    def __init__(self):
        # the Dict strategy applies, then raises on the tuple keys after
        # creating the group -- the partial write that used to collide with
        # the pickle fallback
        self.wgroups = [{(0, 0, 1, 0): [(1, (0, 0, 0, 0), -1)],
                         (1, 2, 3): 'x'}]
        self.plain = np.arange(4)


def test_partial_strategy_write_falls_back_to_pickle(tmp_path):
    original = CarriesTupleKeyedDict()
    with h5.File(tmp_path / 'fallback.h5', 'w') as f:
        original.to_h5(f.create_group('object'))
        loaded = CarriesTupleKeyedDict.from_h5(f['object'])
    assert loaded.wgroups == original.wgroups
    assert (loaded.plain == original.plain).all()


if __name__ == '__main__':
    exit(pytest.main([__file__, '-v']))
