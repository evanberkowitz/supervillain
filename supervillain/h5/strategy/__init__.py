from .readwriteable import ReadWriteable

from .bool import Boolean as bool
from .int import Integer as int
from .float import Float as float
from .complex import Complex as complex

from .list import List as list
from .tuple import Tuple as tuple
# TODO: set, range, slice
# TODO: string
# TODO: bytes
# TODO: full stdlib coverage? https://docs.python.org/3/library/stdtypes.html
from .dict import Dict as dict

import supervillain.h5.strategy.np as np
