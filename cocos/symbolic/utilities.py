import numpy as np
import typing as tp

import cocos.numerics as cn
from cocos.numerics.data_types import NumericArray


def find_length_of_state_vectors(state_vectors: tp.Iterable[NumericArray]) \
        -> int:
    """
    Returns the length of the length of the state vectors of a set of state.

    Args:
        state_vectors: (todo): write your description
        tp: (todo): write your description
        Iterable: (todo): write your description
        NumericArray: (int): write your description
    """
    R = max([state_argument.size
         for state_argument
         in state_vectors
         if isinstance(state_argument, (np.ndarray, cn.ndarray))] + [1])

    return R
