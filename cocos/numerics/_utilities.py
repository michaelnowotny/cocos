from arrayfire.library import backend, c_char_ptr_t
from arrayfire.util import safe_call, to_str, c_pointer
import math
import typing as tp


def _pad_shape_tuple_default(shape: tp.Union[tp.Sequence, int, float],
                             d0: tp.Optional[int],
                             d1: tp.Optional[int],
                             d2: tp.Optional[int],
                             d3: tp.Optional[int]) \
        -> tp.Tuple[int,
                    tp.Optional[int],
                    tp.Optional[int],
                    tp.Optional[int]]:
    if isinstance(shape, list) or isinstance(shape, int):
        shape = tuple(shape)

    ndim = len(shape)
    if ndim < 1 or ndim > 4:
        raise ValueError("the array must have between 1 and 4 axes")
    else:
        d0 = shape[0]
        if ndim > 1:
            d1 = shape[1]
            if ndim > 2:
                d2 = shape[2]
                if ndim > 3:
                    d3 = shape[3]

    return d0, d1, d2, d3


def _pad_shape_tuple_none(shape: tp.Union[tp.Sequence, int, float]) \
        -> tp.Tuple[int,
                    tp.Optional[int],
                    tp.Optional[int],
                    tp.Optional[int]]:
    return _pad_shape_tuple_default(shape=shape,
                                    d0=None,
                                    d1=None,
                                    d2=None,
                                    d3=None)


def _pad_shape_tuple_axis(axes: tp.Sequence[int]) \
        -> tp.Tuple[int, int, int, int]:
    output_shape = list(axes)
    for i in range(len(axes), 4):
        output_shape.append(i)

    return tuple(output_shape)


def _pad_shape_tuple_one(shape: tp.Tuple[int]) -> tp.Tuple[int, int, int, int]:
    d0 = 1
    d1 = 1
    d2 = 1
    d3 = 1
    ndim = len(shape)
    if ndim < 1 or ndim > 4:
        raise ValueError("the array must have between 1 and 4 axes")
    else:
        d0 = shape[0]
        if ndim > 1:
            d1 = shape[1]
            if ndim > 2:
                d2 = shape[2]
                if ndim > 3:
                    d3 = shape[3]

    return d0, d1, d2, d3


def _as_str(self, dims: bool = True):
    arr_str = c_char_ptr_t(0)
    be = backend.get()
    safe_call(be.af_array_to_string(c_pointer(arr_str), "", self.arr, 4, dims))
    py_str = to_str(arr_str)
    safe_call(be.af_free_host(arr_str))
    return py_str


def _compute_slice_length(index: slice, length: int) -> int:
    """
    Given a Python slice such as 1:8:2 and the length of a source sequence
    (e.g. a numpy array) this function computes the resulting number of elements
    when using the slice to index the source sequence.

    :param index: a Python slice
    :param length: length of the source sequence
    :return: the number of elements resulting from the indexing
    """

    start, stop, step = index.indices(length)
    # print(f'start = {start}, stop = {stop}, step = {step}')
    computed_length = math.floor((stop - start - 1) / step) + 1
    # print(f'computed length = {computed_length}')
    return computed_length


def _remove_trailing_ones(shape: tp.Tuple[int, ...]) -> tp.Tuple[int, ...]:
    while len(shape) > 1 and shape[-1] == 1:
        shape = shape[:-1]

    return shape
