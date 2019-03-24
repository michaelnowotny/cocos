import collections
import numpy as np
from types import ModuleType
import typing as tp

from cocos.numerics.data_types import NumericArray
import cocos.numerics as cn


def select_num_pack(gpu: bool = True) -> ModuleType:
    if gpu:
        return cn
    else:
        return np


def select_num_pack_by_dtype(array: NumericArray) -> ModuleType:
    gpu, num_pack = get_gpu_and_num_pack_by_dtype(array)
    return num_pack


def get_gpu_and_num_pack_by_dtype(array: NumericArray) \
        -> tp.Tuple[bool, ModuleType]:
    if isinstance(array, cn.ndarray):
        return True, cn
    else:
        return False, np


def select_num_pack_by_dtype_from_iterable(arrays: tp.Iterable[NumericArray]) \
        -> ModuleType:
    gpu, num_pack = get_gpu_and_num_pack_by_dtype_from_iterable(arrays)
    return num_pack


def get_gpu_and_num_pack_by_dtype_from_iterable(
        arrays: tp.Iterable[NumericArray]) \
        -> tp.Tuple[bool, ModuleType]:
    if any([isinstance(array, cn.ndarray) for array in arrays]):
        return True, cn
    else:
        return False, np
