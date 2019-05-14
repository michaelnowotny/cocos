import arrayfire as af
import typing as tp

from ._array import ndarray, _wrap_af_array


def count_nonzero(a: ndarray,
                  axis: tp.Optional[int] = None) \
        -> tp.Union[int, ndarray]:
    return _wrap_af_array(af.count(a._af_array, dim=axis))


def diff(a: ndarray,
         n: int = 1,
         axis: int = -1) -> ndarray:
    """Calculate the n-th discrete difference along given axis."""

    if axis == -1:
        # use last axis
        axis = a.ndim - 1

    if 0 <= axis <= 3:
        if axis >= a.ndim:
            raise ValueError("axis exceeds array dimension")

        if n >= a.shape[axis]:
            raise ValueError(f"input array has length {a.shape[axis]} in "
                             f"dimension {axis} and therefore cannot be "
                             f"differentiated more than {a.shape[axis] - 1} "
                             f"times")

        if n == 0:
            return a.copy()
        elif n == 1:
            new_array = af.diff1(a._af_array, dim=axis)
        elif n == 2:
            new_array = af.diff2(a._af_array, dim=axis)
        elif n > 2:
            output = a
            while n >= 2:
                n -= 2
                output = ndarray(af.diff2(output._af_array, dim=axis))

            if n == 1:
                output = ndarray(af.diff1(output._af_array, dim=axis))

            return output
        else:
            raise ValueError(f"n must be positive but is {n}")
    else:
        raise ValueError("Axis must be between 0 and 3")

    return ndarray(new_array)


def flatnonzero(a: ndarray) -> ndarray:
    return ndarray(af.where(a._af_array))


def sort_by_keys(keys: ndarray,
                 values: ndarray,
                 axis: int = -1,
                 ascending: bool = True) -> tp.Tuple[ndarray, ndarray]:
    if keys.shape != values.shape:
        raise ValueError("Keys and values must have the same dimensions.")
    elif axis is None:
        keys = keys.flatten()
        values = values.flatten()
    elif axis == -1:
        axis = keys.ndim - 1
    elif axis >= keys.ndim:
        raise ValueError(f"Parameter axis must be between -1 and "
                         f"{keys.ndim - 1}")

    ordered_values, ordered_keys \
        = af.sort_by_key(values._af_array,
                         keys._af_array,
                         is_ascending=ascending)

    return ndarray(ordered_keys), ndarray(ordered_values)


def unique(ar: ndarray,
           return_index: bool = False,
           return_inverse: bool = False,
           return_counts: bool = False) -> ndarray:
    if return_index:
        raise ValueError("return_index=True is not supported")
    if return_inverse:
        raise ValueError("return_inverse=True is not supported")
    if return_counts:
        raise ValueError("return_counts=True is not supported")

    unsorted_unique_set_af_array = af.set_unique(ar._af_array,
                                                 is_sorted=False)

    sorted_unique_set_af_array = af.sort(unsorted_unique_set_af_array,
                                         dim=0,
                                         is_ascending=True)

    return ndarray(sorted_unique_set_af_array)


def union1d(ar1: ndarray, ar2: ndarray) -> ndarray:
    new_af_array = af.set_union(ar1._af_array,
                                ar2._af_array,
                                is_unique=False)

    return ndarray(new_af_array)


def intersect1d(ar1: ndarray, ar2: ndarray) -> ndarray:
    new_af_array = af.set_intersect(ar1._af_array,
                                    ar2._af_array,
                                    is_unique=False)

    return ndarray(new_af_array)
