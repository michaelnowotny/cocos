import numpy as np
import typing as tp

import cocos.numerics as cn
from cocos.numerics.data_types import NumericArray


def find_length_of_state_vectors(state_vectors: tp.Iterable[NumericArray]) \
        -> int:
    R = max([state_argument.size
         for state_argument
         in state_vectors
         if isinstance(state_argument, (np.ndarray, cn.ndarray))] + [1])

    return R
