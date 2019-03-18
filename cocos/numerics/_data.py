import arrayfire as af
import numpy as np
import typing as tp

from ._array import full, ndarray
from ._conversion import convert_numpy_to_af_type
from ._utilities import _pad_shape_tuple_none, _pad_shape_tuple_one


def empty(shape: tp.Tuple[int, ...],
          dtype: np.generic = np.float32) -> ndarray:
    """
    Return a new array of given shape and type, initializing entries to zero.
    """

    return zeros(shape, dtype)


def zeros(shape: tp.Tuple[int, ...],
          dtype: np.generic = np.float32) -> ndarray:
    """
    Return a new array of given shape and type, filled with zeros.
    """

    return full(shape, 0.0, dtype)


def ones(shape: tp.Tuple[int, ...],
         dtype: np.generic = np.float32) -> ndarray:
    """
    Return a new array of given shape and type, filled with ones.
    """

    return full(shape, 1.0, dtype)


def empty_like(a,
               dtype: tp.Optional[np.generic] = None,
               order: str = 'K',
               subok: bool = True) -> ndarray:
    """
    Return a new array with the same shape and type as a given array.
    """

    return zeros_like(a, dtype, order, subok)


def zeros_like(a,
               dtype: tp.Optional[np.generic] = None,
               order: str = 'K',
               subok: bool = True) -> ndarray:
    """
    Return an array of zeros with the same shape and type as a given array.
    """

    if not dtype:
        dtype = a.dtype

    return zeros(a.shape, dtype=dtype)


def ones_like(a,
              dtype: tp.Optional[np.generic] = None,
              order: str = 'K',
              subok: bool = True) -> ndarray:
    """
    Return an array of ones with the same shape and type as a given array.
    """

    if not dtype:
        dtype = a.dtype

    return ones(a.shape, dtype=dtype)


def eye(N: int,
        M: tp.Optional[int] = None,
        k: int = 0,
        dtype: np.generic = np.float32) -> ndarray:
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.
    """

    af_type = convert_numpy_to_af_type(dtype)
    if not M:
        M = N

    if k != 0:
        raise ValueError("k not zero is not supported")
    af_array = af.data.identity(N, M, dtype=af_type)

    return ndarray(af_array)


def identity(n: int,
             dtype: np.generic = np.float32) -> ndarray:
    """
    Return the identity array.
    """

    af_type = convert_numpy_to_af_type(dtype)
    af_array = af.data.identity(n, n, dtype=af_type)

    return ndarray(af_array)


def diag(v: ndarray,
         k: int = 0) -> ndarray:
    """
    Extract a diagonal or construct a diagonal array.
    """

    if v.ndim == 1:
        af_array = af.data.diag(v._af_array, num=k, extract=False)
    elif v.ndim == 2:
        af_array = af.data.diag(v._af_array, num=k, extract=True)
    else:
        raise ValueError("v must be a 1d or a 2d ndarray")

    return ndarray(af_array)


def diagflat(v: ndarray,
             k: int = 0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.
    """

    if v.ndim != 1:
        raise ValueError(f"input must be a flat array but has {v.ndim} axes")

    return diag(v, k)


def concatenate(arrays: tp.Tuple[ndarray, ...],
                axis: int = 0) -> ndarray:
    if len(arrays) < 2 or len(arrays) > 4:
        raise ValueError("the number of arrays to be concatenated is supposed "
                         "to be between two and four")

    first = arrays[0]._af_array
    second = arrays[1]._af_array
    third = None
    fourth = None

    if len(arrays) > 2:
        third = arrays[2]._af_array
        if len(arrays) > 3:
            fourth = arrays[3]._af_array

    af_array = af.join(axis, first, second, third, fourth)

    return ndarray(af_array)


def vstack(tup:tp.Tuple[ndarray, ...]) -> ndarray:
    return concatenate(tup, 0)


def hstack(tup: tp.Tuple[ndarray, ...]) -> ndarray:
    return concatenate(tup, 1)


def dstack(tup: tp.Tuple[ndarray, ...]) -> ndarray:
    return concatenate(tup, 2)


def roll(a: ndarray,
         shift: int,
         axis: tp.Optional[int] = None) -> ndarray:
    if not axis:
        (d0, d1, d2, d3) = _pad_shape_tuple_one(a.shape)
        flat_af_array = af.data.flat(a._af_array)
        af_array_flat_shift = af.data.shift(flat_af_array, shift)
        af_array = af.data.moddims(af_array_flat_shift, d0, d1, d2, d3)
    else:
        shift_dims = [0, 0, 0, 0]
        shift_dims[axis] = shift
        d0, d1, d2, d3 = shift_dims

        af_array = af.data.shift(a._af_array, d0, d1, d2, d3)

    return ndarray(af_array)


def flip(m: ndarray,
         axis: int) -> ndarray:
    return ndarray(af.data.flip(m._af_array, axis))


def fliplr(m: ndarray) -> ndarray:
    return ndarray(af.data.flip(m._af_array, 1))


def flipud(m: ndarray) -> ndarray:
    return ndarray(af.data.flip(m._af_array, 0))


def tril(m: ndarray,
         k: int = 0) -> ndarray:
    """
    Lower triangle of an array.
    """
    af_array = af.data.lower(m._af_array, is_unit_diag=False)

    return ndarray(af_array)


def triu(m: ndarray,
         k: int = 0) -> ndarray:
    """
    Upper triangle of an array.
    """
    af_array = af.data.upper(m._af_array, is_unit_diag=False)

    return ndarray(af_array)
