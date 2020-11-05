import collections
import numpy as np
from types import ModuleType
import typing as tp

from cocos.numerics.data_types import NumericArray
import cocos.numerics as cn


def select_num_pack(gpu: bool = True) -> ModuleType:
    """
    Selects the number of ints from the nearest integer.

    Args:
        gpu: (todo): write your description
    """
    if gpu:
        return cn
    else:
        return np


def select_num_pack_by_dtype(array: NumericArray) -> ModuleType:
    """
    Selects the number of the given dtype.

    Args:
        array: (array): write your description
    """
    gpu, num_pack = get_gpu_and_num_pack_by_dtype(array)
    return num_pack


def get_gpu_and_num_pack_by_dtype(array: NumericArray) \
        -> tp.Tuple[bool, ModuleType]:
    """
    Get numpy array and dtype and numpy arrays.

    Args:
        array: (array): write your description
    """
    if isinstance(array, cn.ndarray):
        return True, cn
    else:
        return False, np


def select_num_pack_by_dtype_from_iterable(arrays: tp.Iterable[NumericArray]) \
        -> ModuleType:
    """
    Returns the number of dtype dtype that have the given dtype.

    Args:
        arrays: (array): write your description
        tp: (todo): write your description
        Iterable: (todo): write your description
        NumericArray: (int): write your description
    """
    gpu, num_pack = get_gpu_and_num_pack_by_dtype_from_iterable(arrays)
    return num_pack


def get_gpu_and_num_pack_by_dtype_from_iterable(
        arrays: tp.Iterable[NumericArray]) \
        -> tp.Tuple[bool, ModuleType]:
    """
    Get numpy array and num_by_dtype and num_dtype.

    Args:
        arrays: (array): write your description
        tp: (str): write your description
        Iterable: (todo): write your description
        NumericArray: (int): write your description
    """
    if any([isinstance(array, cn.ndarray) for array in arrays]):
        return True, cn
    else:
        return False, np
