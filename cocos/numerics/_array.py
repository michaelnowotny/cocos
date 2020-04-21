import array as pyarray
import arrayfire as af
import numbers
import numpy as np
import typing as tp

from cocos.options import \
    GPUOptions, \
    MixedComputationErrorLevel

from ._conversion import \
    convert_numpy_to_af_type, \
    convert_af_to_numpy_type

from ._utilities import \
    _as_str, \
    _pad_shape_tuple_none, \
    _pad_shape_tuple_axis, \
    _pad_shape_tuple_one, \
    _compute_slice_length, \
    _remove_trailing_ones


def _verify_operand(other):
    if  isinstance(other, np.ndarray):
        if GPUOptions.mixed_computation_error_level == \
                MixedComputationErrorLevel.ERROR:
            raise TypeError("Combining a gpu operation with a numpy array")
        elif GPUOptions.mixed_computation_error_level == \
                MixedComputationErrorLevel.WARNING:
            print("Warning: Combining a gpu operation with a numpy array "
                  "results in a conversion which is slow.")


def _binary_function(lhs, rhs, func) -> 'ndarray':
    _verify_operand(lhs)
    _verify_operand(rhs)

    if isinstance(lhs, ndarray):
        lhs = lhs._af_array

    if isinstance(rhs, ndarray):
        rhs = rhs._af_array

    new_af_array = func(lhs, rhs)
    return ndarray(new_af_array)


def _binary_method(rhs,
                   func,
                   broadcast: bool = False) -> 'ndarray':
    _verify_operand(rhs)
    argument = None
    if isinstance(rhs, ndarray):
        argument = rhs._af_array
    else:
        argument = rhs

    if broadcast:
        new_af_array = af.broadcast(func, argument)
    else:
        new_af_array = func(argument)
    return ndarray(new_af_array)


def _unary_function(x, af_func: tp.Callable, np_func: tp.Callable) -> 'ndarray':
    _verify_operand(x)
    # print("unary function argument is a {}".format(type(x)))
    if isinstance(x, ndarray):
        x = x._af_array
        return ndarray(af_func(x))
    elif isinstance(x, numbers.Number):
        return np_func(x)
    else:
        return af_func(x)


def is_boolean(a):
    return a.dtype == np.bool or a.dtype == np.bool_


def _translate_index_key(item, input_shape: tp.Union[int, tp.Tuple[int, ...]]) \
        -> tp.Tuple[tp.Any, tp.Tuple[int, ...]]:
    if isinstance(item, tuple):
        # item is a tuple of items
        af_item = []
        required_shape = []

        for key, current_shape in zip(item, input_shape):
            translated_index_for_key, required_shape_for_key \
                = _translate_index_key(key, current_shape)
            af_item.append(translated_index_for_key)
            if isinstance(required_shape_for_key, tuple):
                required_shape_for_key = list(required_shape_for_key)
                required_shape += required_shape_for_key
            else:
                required_shape.append(required_shape_for_key)

        af_item = tuple(af_item)
        required_shape = tuple(required_shape)
    elif isinstance(item, ndarray):
        # item is a cocos array
        af_item = item._af_array
        if is_boolean(item):
            required_shape = (int(item.sum()), )
        else:
            required_shape = item.shape
    elif isinstance(item, np.ndarray):
        # item is a numpy array
        af_item = array(item)._af_array
        if is_boolean(item):
            required_shape = (int(item.sum()), )
        else:
            required_shape = item.shape
    elif isinstance(item, numbers.Number):
        af_item = item
        required_shape = (1, )
    elif isinstance(item, slice):
        af_item = item
        if isinstance(input_shape, tuple):
            input_shape = input_shape[0]
        required_shape = (_compute_slice_length(item, input_shape), )
    else:
        af_item = item
        required_shape = (-1,)

    return af_item, required_shape


# def _translate_index_key(item):
#     if isinstance(item, tuple):
#         af_item = []
#         for key in item:
#             if isinstance(key, ndarray):
#                 af_item.append(key._af_array)
#             else:
#                 af_item.append(key)
#
#         af_item = tuple(af_item)
#     elif isinstance(item, ndarray):
#         af_item = item._af_array
#     elif isinstance(item, np.ndarray):
#         af_item = array(item)
#     else:
#         af_item = item
#
#     return af_item


class ndarray(object):
    __array_priority__ = 35

    def __init__(self,
                 af_array: af.Array,
                 shape: tp.Optional[tp.Tuple[int, ...]] = None):
        self._af_array = af_array
        self._label = 'array'
        self._shape = shape

    def __del__(self):
        del self._af_array

    def __getstate__(self) -> tp.Tuple:
        return np.array(self._af_array), self._label, self._shape

    def __setstate__(self, state: tp.Tuple):
        numpy_array, label, shape = state
        af_array = af.to_array(numpy_array)
        self._af_array = af_array
        self._label = label
        self._shape = shape

    def __array__(self) -> np.ndarray:

        return self._af_array.__array__()

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def dtype(self) -> np.generic:
        """
        Data-type of the array’s elements.
        """

        return convert_af_to_numpy_type(self._af_array.dtype())

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        """
        Tuple of array dimensions.
        """

        if self._shape is not None:
            return self._shape
        else:
            return self._af_array.dims()

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions.
        """

        return len(self.shape)

    @property
    def size(self) -> int:
        """
        Number of elements in the array.
        """

        return self._af_array.elements()

    @property
    def nbytes(self) -> int:
        """
        Total bytes consumed by the elements of the array.
        """

        return self._af_array.allocated()

    @property
    def itemsize(self) -> int:
        """
        Length of one array element in bytes.
        """

        return int(self.nbytes / self.size)

    @property
    def strides(self):
        """
        Tuple of bytes to step in each dimension when traversing an array.
        """

        return self._af_array.strides()

    @property
    def T(self):
        """
        Same as self.transpose(), except that self is returned if self.ndim < 2.
        """

        if self.ndim < 2:
            return self
        else:
            return ndarray(af.transpose(self._af_array, False))

    @property
    def H(self):
        """
        Conjugate transpose.
        """

        return ndarray(af.transpose(self._af_array, True))

    @property
    def imag(self) -> 'ndarray':
        """
        The imaginary part of the array.
        """

        return imag(self)

    @property
    def real(self) -> 'ndarray':
        """
        The real part of the array.
        """

        return real(self)

    def astype(self,
               dtype: np.generic,
               order: str = 'K',
               casting: str = 'unsafe',
               subok: bool = True,
               copy: bool = True) \
            -> 'ndarray':
        """
        Copy of the array, cast to a specified type.
        """

        af_type = convert_numpy_to_af_type(dtype)
        af_array = self._af_array.as_type(af_type)
        return ndarray(af_array)

    def copy(self) -> 'ndarray':
        """
        Return a copy of the array.
        """

        af_array = self._af_array.copy()
        return ndarray(af_array)

    def __str__(self) -> str:
        if GPUOptions.str_via_numpy:
            np_array = np.array(self)
            return np_array.__str__()
        else:
            # return self._af_array._as_str()
            return _as_str(self._af_array, dims=True)

    def __repr__(self) -> str:
        np_array = np.array(self)
        return np_array.__repr__()

    def __len__(self) -> int:
        return self.shape[0]

    def __abs__(self):
        new_af_array = af.abs(self._af_array)
        return ndarray(new_af_array)

    @af.broadcast
    def __add__(self, other):
        return _binary_method(other, func=self._af_array.__add__)

    @af.broadcast
    def __iadd__(self, other):
        return _binary_method(other, func=self._af_array.__iadd__)

    @af.broadcast
    def __radd__(self, other):
        return _binary_method(other, func=self._af_array.__radd__)

    @af.broadcast
    def __sub__(self, other):
        return _binary_method(other, func=self._af_array.__sub__)

    @af.broadcast
    def __isub__(self, other):
        return _binary_method(other, func=self._af_array.__isub__)

    @af.broadcast
    def __rsub__(self, other):
        return _binary_method(other, func=self._af_array.__rsub__)

    @af.broadcast
    def __mul__(self, other):
        return _binary_method(other, func=self._af_array.__mul__)

    @af.broadcast
    def __imul__(self, other):
        return _binary_method(other, func=self._af_array.__imul__)

    @af.broadcast
    def __rmul__(self, other):
        return _binary_method(other, func=self._af_array.__rmul__)

    @af.broadcast
    def __truediv__(self, other):
        return _binary_method(other, func=self._af_array.__truediv__)

    @af.broadcast
    def __itruediv__(self, other):
        return _binary_method(other, func=self._af_array.__itruediv__)

    @af.broadcast
    def __rtruediv__(self, other):
        return _binary_method(other, func=self._af_array.__rtruediv__)

    @af.broadcast
    def __idiv__(self, other):
        return _binary_method(other, func=self._af_array.__idiv__)

    @af.broadcast
    def __rdiv__(self, other):
        return _binary_method(other, func=self._af_array.__rdiv__)

    @af.broadcast
    def __mod__(self, other):
        return _binary_method(other, func=self._af_array.__mod__)

    @af.broadcast
    def __imod__(self, other):
        return _binary_method(other, func=self._af_array.__imod__)

    @af.broadcast
    def __rmod__(self, other):
        return _binary_method(other, func=self._af_array.__rmod__)

    @af.broadcast
    def __pow__(self, other):
        return _binary_method(other, func=self._af_array.__pow__)

    @af.broadcast
    def __ipow__(self, other):
        return _binary_method(other, func=self._af_array.__ipow__)

    @af.broadcast
    def __rpow__(self, other):
        return _binary_method(other, func=self._af_array.__rpow__)

    @af.broadcast
    def __lt__(self, other):
        return _binary_method(other, func=self._af_array.__lt__)

    @af.broadcast
    def __gt__(self, other):
        return _binary_method(other, func=self._af_array.__gt__)

    @af.broadcast
    def __le__(self, other):
        return _binary_method(other, func=self._af_array.__le__)

    @af.broadcast
    def __ge__(self, other):
        return _binary_method(other, func=self._af_array.__ge__)

    @af.broadcast
    def __eq__(self, other):
        return _binary_method(other, func=self._af_array.__eq__)

    @af.broadcast
    def __ne__(self, other):
        return _binary_method(other, func=self._af_array.__ne__)

    @af.broadcast
    def __and__(self, other):
        return _binary_method(other, func=self._af_array.__and__)

    @af.broadcast
    def __iand__(self, other):
        return _binary_method(other, func=self._af_array.__iand__)

    @af.broadcast
    def __or__(self, other):
        return _binary_method(other, func=self._af_array.__or__)

    @af.broadcast
    def __ior__(self, other):
        return _binary_method(other, func=self._af_array.__ior__)

    @af.broadcast
    def __xor__(self, other):
        return _binary_method(other, func=self._af_array.__xor__)

    @af.broadcast
    def __ixor__(self, other):
        return _binary_method(other, func=self._af_array.__ixor__)

    @af.broadcast
    def __lshift__(self, other):
        return _binary_method(other, func=self._af_array.__lshift__)

    @af.broadcast
    def __ilshift__(self, other):
        return _binary_method(other, func=self._af_array.__ilshift__)

    @af.broadcast
    def __rshift__(self, other):
        return _binary_method(other, func=self._af_array.__rshift__)

    @af.broadcast
    def __irshift__(self, other):
        return _binary_method(other, func=self._af_array.__irshift__)

    def __neg__(self):
        return ndarray(self._af_array.__neg__())

    def __pos__(self):
        return ndarray(self._af_array.__pos__())

    def __invert__(self):
        return ndarray(self._af_array.__invert__())

    def __matmul__(self, other):
        _verify_operand(other)
        return dot(self, other)

    def __getitem__(self, item) -> 'ndarray':
        af_item, required_shape = _translate_index_key(item, self.shape)
        # from IPython.core.debugger import set_trace
        # set_trace()
        new_af_array = self._af_array.__getitem__(af_item)
        new_cocos_array = ndarray(new_af_array)

        if _remove_trailing_ones(new_cocos_array.shape) != \
                _remove_trailing_ones(required_shape):
            new_cocos_array = reshape_without_reorder(new_cocos_array,
                                                      newshape=required_shape)

        return new_cocos_array

    def __setitem__(self, key, value) -> 'ndarray':
        from cocos.options import GPUOptions, MixedComputationErrorLevel
        af_key, required_shape = _translate_index_key(key, self.shape)
        if isinstance(value, np.ndarray):
            if GPUOptions.mixed_computation_error_level == MixedComputationErrorLevel.ERROR:
                raise TypeError('Trying to set elements of a cocos array with a numpy array.')
            elif GPUOptions.mixed_computation_error_level == MixedComputationErrorLevel.WARNING:
                print('Trying to set elements of a cocos array with a numpy array might be slow. ')

            value = array(value)
        if isinstance(value, ndarray):
            value = value._af_array
        new_af_array = self._af_array.__setitem__(af_key, value)

        return ndarray(new_af_array)

    def __sizeof__(self):
        return self.size

    def dot(self,
            b: 'ndarray') -> 'ndarray':
        """
        Dot product of two arrays.
        """

        return dot(self, b)

    def _get_index_from_args(self, *args) \
            -> tp.Union[int, tp.Tuple[int, ...]]:
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]

        if len(args) == 1:
            if self.ndim == 1:
                index = args[0]
            else:
                index = np.unravel_index(args[0], self.shape)
        else:
            index = args

        return index

    def item(self, *args):
        """
        Copy an element of an array to a standard Python scalar and return it.
        """

        index = self._get_index_from_args(*args)
        af_item, required_shape = _translate_index_key(index, self.shape)
        return (np.array(self._af_array[af_item])).item()
        # return self._af_array[af_item].scalar()

    def itemset(self, *args):
        """
        Insert scalar into an array
        (scalar is cast to array’s dtype, if possible)
        """

        value = args[-1]
        index = self._get_index_from_args(*args[:-1])
        self[index] = value

    def conj(self) -> 'ndarray':
        """Complex-conjugate all elements."""

        return conj(self)

    def conjugate(self) -> 'ndarray':
        """
        Return the complex conjugate, element-wise.
        """

        return conj(self)

    def flatten(self) -> 'ndarray':
        """
        Return a copy of the array collapsed into one dimension.
        """

        af_array = af.data.flat(self._af_array)
        return ndarray(af_array)

    def reshape(self, shape: tp.Tuple[int, ...]) -> 'ndarray':
        """
        Returns an array containing the same data with a new shape.
        """

        return reshape(self, shape)

    def squeeze(self,
                axis: tp.Optional[int] = None) -> 'ndarray':
        """
        Remove single-dimensional entries from the shape of a.
        """

        return squeeze(self, axis)

    def transpose(self):
        """
        Returns a new array with axes transposed.
        """

        return ndarray(af.transpose(self._af_array, False))

    def clip(self, min, max) -> 'ndarray':
        """
        Return an array whose values are limited to [min, max].
        """

        return ndarray(af.clamp(self._af_array, min, max))

    def max(self, axis: tp.Optional[int] = None):
        """
        Return the maximum along a given axis.
        """

        new_array = af.max(self._af_array, dim=axis)
        if isinstance(new_array, af.Array):
            return ndarray(new_array)
        else:
            return new_array

    def mean(self,
             axis: tp.Optional[int] = None,
             dtype: tp.Optional[np.generic] = None):
        """
        Returns the average of the array elements along given axis.
        """

        return mean(self, axis=axis)

    def min(self,
            axis: tp.Optional[int] = None):
        """
        Returns the minimum along given axis.
        """

        new_array = af.min(self._af_array, dim=axis)
        if isinstance(new_array, af.Array):
            return ndarray(new_array)
        else:
            return new_array

    def nonzero(self) -> tp.Tuple:
        """
        Return the indices of the elements that are non-zero.
        """

        return nonzero(self)

    def sort(self,
             axis: int = -1,
             ascending: bool = True):
        """
        Sort an array, in-place.
        """

        self._af_array = _sort_internal(self, axis, ascending)

    def all(self,
            axis: tp.Optional[int] = None) \
            -> tp.Union[bool, 'ndarray']:
        """
        Returns True if all elements evaluate to True.
        """

        return all(self, axis)

    def any(self,
            axis: tp.Optional[int] = None,
            out: tp.Optional = None,
            keepdims: bool = False) \
            -> tp.Union[bool, 'ndarray']:
        """
        Returns True if any of the elements of a evaluate to True.
        """

        return any(self, axis)

    def argmax(self,
               axis: tp.Optional[int] = None,
               out: tp.Optional['ndarray'] = None):
        """
        Return indices of the maximum values along the given axis.
        """

        val, idx = af.imax(self._af_array, dim=axis)
        return _wrap_af_array(idx)

    def argmin(self,
               axis: tp.Optional[int] = None,
               out: tp.Optional['ndarray'] = None):
        """
        Return indices of the minimum values along the given axis of a.
        """

        val, idx = af.imin(self._af_array, dim=axis)
        return _wrap_af_array(idx)

    def argsort(self,
                axis: int = -1,
                ascending: bool = True) \
            -> 'ndarray':
        """
        Returns the indices that would sort this array.
        """

        out, indices = sort_argsort(self, axis, ascending)
        return indices

    def cumsum(self,
               axis: tp.Optional[int] = None) \
            -> tp.Union[float, 'ndarray']:
        """
        Return the cumulative sum of the elements along the given axis.
        """

        return cumsum(self, axis)

    def diagonal(self,
                 offset: int = 0):
        """
        Return specified diagonals.
        """

        return ndarray(af.data.diag(self._af_array,
                                    num=offset,
                                    extract=True))

    def fill(self, value):
        """
        Fill the array with a scalar value.
        """

        shape = self.shape
        dtype = self.dtype
        del self._af_array
        self._af_array = _full_internal(shape, value, dtype)

    def prod(self,
             axis: tp.Optional[int] = None) \
            -> tp.Union[float, 'ndarray']:
        """
        Return the product of the array elements over the given axis
        """

        return prod(self)

    def ptp(self,
            axis: tp.Optional[int] = None) \
            -> 'ndarray':
        """
        Peak to peak (maximum - minimum) value along a given axis.
        """

        return ptp(self, axis)

    def repeat(self,
               repeats, axis: tp.Optional[int] = None) \
            -> 'ndarray':
        """
        Repeat elements of an array.
        """

        if axis is None:
            return tile(self.flatten(), repeats)
        else:
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"axis must be between 0 and {self.ndim-1}")
            new_shape = list(self.shape)
            new_shape[axis] *= repeats
            return tile(self.flatten(), repeats).reshape(tuple(new_shape))
            # reps = [1, 1, 1, 1]
            # reps[axis] = repeats
            # return tile(self, tuple(reps))

    def round(self) -> 'ndarray':
        return round(self)

    def std(self, axis: tp.Optional[int] = None) \
            -> tp.Union['ndarray', numbers.Number]:
        """
        Returns the standard deviation of the array elements along given axis.
        """

        return std(self, axis)

    def sum(self,
            axis: tp.Optional[int] = None) \
            -> tp.Union[numbers.Number, 'ndarray']:
        """
        Return the sum of the array elements over the given axis.
        """

        return sum(self, axis)

    def trace(self, offset: int = 0) -> numbers.Number:
        """
        Return the sum along diagonals of the array.
        """

        return trace(self, offset)

    def var(self,
            axis: tp.Optional[int] = None,
            ddof: int = 0) \
            -> tp.Union[numbers.Number, 'ndarray']:
        """
        Returns the variance of the array elements, along given axis.
        """

        return var(self, axis=axis, ddof=ddof)


def asscalar(a: ndarray) \
        -> numbers.Number:
    if a.size > 1:
        raise ValueError(f"a must be an array size 1 but has {a.size} elements")

    return (np.array(a)).item()


def matmul(a: ndarray,
           b: ndarray) \
        -> ndarray:
    """
    Matrix product of two arrays.
    """

    return ndarray(af.matmul(a._af_array, b._af_array))


def copy(a: ndarray) -> ndarray:
    """
    Return an array copy of the given object.
    """

    return a.copy()


def ptp(a: ndarray,
        axis: tp.Optional[int] = None) -> ndarray:
    """
    Range of values (maximum - minimum) along an axis.
    """

    if axis is None:
        tmp = a.flatten()
    else:
        tmp = a

    return tmp.max(axis) - tmp.min(axis)


def array(object: tp.Union[tp.Sequence,
                           np.ndarray,
                           pyarray.array],
          dtype: tp.Optional[np.generic] = None,
          copy: bool = True,
          order: tp.Optional = None,
          subok: bool = False,
          ndmin: int = 0) \
        -> ndarray:
    """
    Create an array.
    """

    if order:
        raise ValueError('order != None is not supported')

    # if dtype is not None:
    #     raise ValueError("parameter dtype != None is not supported")

    af_array = None
    shape = None
    if isinstance(object, np.ndarray):
        if object.ndim == 0:
            # object = [np.asscalar(object)]
            object = np.reshape(object, (1,))
        af_array = af.to_array(object)
        shape = object.shape
    elif isinstance(object, list):
        np_array = np.array(object, dtype=dtype)
        af_array = af.to_array(np_array)
        shape = np_array.shape
    else:
        af_array = af.Array(object)

    result = ndarray(af_array, shape)
    if dtype is not None and result.dtype != dtype:
        result = result.astype(dtype)
    return result


def transpose(a: ndarray,
              axes: tp.Optional = None):
    """
    Permute the dimensions of an array.
    """

    if axes:
        raise ValueError('axes != None is not supported')

    return a.transpose()


def real(val: ndarray):
    """
    Return the real part of the complex argument.
    """

    return _unary_function(val, af_func=af.real, np_func=np.real)


def imag(a: ndarray):
    """
    Return the imaginary part of the complex argument.
    """

    return _unary_function(a, af_func=af.imag, np_func=np.imag)


def isempty(num: ndarray) -> bool:
    return num._af_array.is_empty()


def isscalar(num: ndarray) -> bool:
    return num._af_array.is_scalar()


def isrow(num: ndarray) -> bool:
    return num._af_array.is_row()


def iscolumn(num: ndarray) -> bool:
    return num._af_array.is_column()


def isvector(num: ndarray) -> bool:
    return num._af_array.is_vector()


def isrealobj(x: ndarray) -> bool:
    return x._af_array.is_real()


def iscomplexobj(x: ndarray) -> bool:
    return x._af_array.is_complex()


def isdouble(num: ndarray) -> bool:
    return num._af_array.is_double()


def issingle(num: ndarray) -> bool:
    return num._af_array.is_single()


def isrealfloating(num: ndarray) -> bool:
    return num._af_array.is_real_floating()


def isfloating(num: ndarray) -> bool:
    return num._af_array.is_floating()


def isinteger(num: ndarray) -> bool:
    return num._af_array.is_integer()


def isbool(num: ndarray) -> bool:
    return num._af_array.is_integer()


def isfortran(a: ndarray) -> bool:
    if isinstance(a, ndarray):
        return True
    elif isinstance(a, np.ndarray):
        return np.isfortran(a)
    else:
        raise TypeError("argument must be an ndarray")


def display(array: ndarray):
    if isinstance(array, ndarray):
        af.display(array._af_array)
    else:
        print(array)


def _complete_shape(a: ndarray, shape: tp.Sequence[int]) \
        -> tp.Tuple[int, ...]:
    shape_np = np.array(shape)
    placeholder_indices = np.where(shape_np == -1)[0]
    number_of_placeholders = len(placeholder_indices)

    assert number_of_placeholders >= 0
    if number_of_placeholders == 0:
        return tuple(shape)
    elif number_of_placeholders > 1:
        raise ValueError('can only specify one unknown dimension')
    elif number_of_placeholders == 1:
        position_of_placeholder = placeholder_indices.item()
        dimensions_without_placeholder = shape_np[np.where(shape_np != -1)]
        size_of_all_other_dimensions = np.prod(dimensions_without_placeholder)
        remainder = a.size % size_of_all_other_dimensions
        if remainder != 0:
            raise ValueError(
                f'cannot reshape array of size {a.size} into shape {shape}')
        dimension_of_placeholder = a.size / size_of_all_other_dimensions
        shape_np[position_of_placeholder] = dimension_of_placeholder
        return tuple(shape_np)
    else:
        ValueError('This cannot happen')


def reorder(a: ndarray, new_order: tp.Tuple[int, ...]):
    if len(new_order) > 4:
        raise ValueError(f'Cocos does not support arrays with more than 4 axes.')

    d0, d1, d2, d3 = _pad_shape_tuple_axis(new_order)
    # print(f'd0={d0}, d1={d1}, d2={d2}, d3={d3}')

    output_array = af.reorder(a._af_array, d0, d1, d2, d3)
    return ndarray(output_array)


def reshape_without_reorder(a: ndarray,
                            newshape: tp.Tuple[int, ...]) -> ndarray:
    """
    Gives a new shape to an array without changing its data.
    """

    newshape = _complete_shape(a, newshape)
    input_array = a._af_array
    d0, d1, d2, d3 = _pad_shape_tuple_none(newshape)
    output_array = af.data.moddims(input_array, d0, d1, d2, d3)
    return ndarray(output_array)


def reshape(a, newshape):
    newshape = _complete_shape(a, newshape)
    if a.ndim < 1 or a.ndim > 4:
        raise ValueError('a must have between 1 and 4 axes')
    elif a.ndim == 1:
        tmp = a
    else:
        tmp = reorder(a, tuple(range(a.ndim))[::-1]).flatten()
    #     elif a.ndim == 2:
    #         tmp = cn.reorder(a_gpu, (1, 0)).flatten()
    #     elif a.ndim == 3:
    #         tmp = cn.reorder(a_gpu, (2, 1, 0)).flatten()

    tmp = reshape_without_reorder(tmp, newshape=newshape[::-1])
    return reorder(tmp, tuple(range(len(newshape)))[::-1])


def squeeze(a: ndarray,
            axis: tp.Optional[int] = None) -> ndarray:
    """
    Remove single-dimensional entries from the shape of an array.
    """

    shape = a.shape
    if axis is None:
        newshape = []
        for i in range(a.ndim):
            if shape[i] != 1:
                newshape.append(shape[i])
    else:
        if axis < 0 or axis >= a.ndim:
            raise ValueError(f"axis must be between 0 and {a.ndim-1}")

        if shape[axis] == 1:
            newshape = list(shape)
            del newshape[axis]
        else:
            raise ValueError(f"axis {axis} must have dimension 1 but has "
                             f"dimension {shape[axis]}")

    # print(f"newshape = {newshape}")
    newshape = tuple(newshape)
    if shape == newshape:
        return a
    else:
        return reshape(a, newshape)


def _dot_internal(a: ndarray, b: ndarray) -> ndarray:
    if (isrow(a) and isrow(b)) or (iscolumn(a) and iscolumn(b)):
        return _binary_function(a, b, af.dot)
    elif isrow(a) and iscolumn(b):
        # inner product
        return _binary_function(a, b, af.matmul)
    else:
        # outer product
        return _binary_function(a, b, af.matmul)


def dot(a: ndarray, b: ndarray) -> ndarray:
    """
    Dot product of two arrays.
    """

    if isvector(a) and isvector(b):
        return _dot_internal(a, b)
    else:
        result = _binary_function(a, b, af.matmul)
        if iscolumn(b):
            result = reshape(result, (result.size, 1))
        return result


def vdot(a: ndarray, b: ndarray) -> ndarray:
    """
    Return the dot product of two vectors.
    """

    if iscomplexobj(a):
        a = a.conj()

    if isvector(a) and isvector(b):
        return _dot_internal(a, b)
    elif a.size == b.size:
        if isvector(a):
            return _dot_internal(a, b.flatten())
        elif isvector(b):
            return _dot_internal(a.flatten(), b)
        else:
            return _dot_internal(a.flatten(), b.flatten())
    else:
        raise ValueError("a and b must have the same number of elements")


def multiply(x1, x2):
    """
    Multiply arguments element-wise.
    """

    _verify_operand(x1)
    _verify_operand(x2)

    return x1 * x2


def divide(x1, x2):
    """
    Divide arguments element-wise.
    """

    _verify_operand(x1)
    _verify_operand(x2)

    return x1 / x2


def average(a: ndarray,
            axis: tp.Optional[int] = None,
            weights: tp.Optional[ndarray] = None) \
        -> tp.Union[numbers.Number, ndarray]:
    """
    Compute the weighted average along the specified axis.
    """

    if weights is None:
        af_weights = None
    else:
        if axis is not None:
            if axis == 0:
                af_weights = weights.reshape((-1, ))._af_array
            elif axis == 1:
                af_weights = weights.reshape((1, -1))._af_array
            elif axis == 2:
                af_weights = weights.reshape((1, 1, -1))._af_array
            elif axis == 3:
                af_weights = weights.reshape((1, 1, 1, -1))._af_array
            else:
                raise ValueError('axis must be between 0 and 3')
        else:
            af_weights = weights._af_array

    new_af_array = af.mean(a._af_array,
                           weights=af_weights,
                           dim=axis)

    if isinstance(new_af_array, af.Array):
        return ndarray(new_af_array)
    else:
        return new_af_array


def mean(a: ndarray,
         axis: tp.Optional[int] = None) \
        -> tp.Union[numbers.Number, ndarray]:
    """
    Compute the arithmetic mean along the specified axis.
    """

    return average(a, axis=axis)


def median(a: ndarray,
           axis: tp.Optional[int] = None) \
        -> tp.Union[numbers.Number, ndarray]:
    """
    Compute the median along the specified axis.
    """

    new_af_array = af.median(a._af_array, dim=axis)
    if isinstance(new_af_array, af.Array):
        return ndarray(new_af_array)
    else:
        return new_af_array


def std(a: ndarray, axis: tp.Optional[int] = None, ddof: int = 0) \
        -> tp.Union[ndarray, numbers.Number]:
    """
    Compute the standard deviation along the specified axis.
    """

    new_af_array: tp.Optional[tp.Union[ndarray, numbers.Number]] = None
    if ddof or ddof == 0:
        new_af_array = af.stdev(a._af_array, dim=axis)
    elif ddof == 1:
        new_af_array = af.sqrt(af.var(a._af_array, isbiased=False, dim=axis))
    if isinstance(new_af_array, af.Array):
        return ndarray(new_af_array)
    else:
        return new_af_array


def var(a: ndarray,
        axis: tp.Optional[int] = None,
        dtype: tp.Optional[np.generic] = None,
        out: tp.Optional[ndarray] = None,
        ddof: int = 0,
        keepdims: bool = False) -> tp.Union[float, ndarray]:
    """
    Compute the variance along the specified axis.
    """

    isbiased = True
    if ddof is not None:
        if ddof == 1:
            isbiased = False
        elif ddof == 0:
            pass
        else:
            raise ValueError(f"ddof must be 0 or 1, ddof={ddof} is not "
                             f"supported")

    new_af_array \
        = af.var(a._af_array, isbiased=isbiased, weights=None, dim=axis)

    if isinstance(new_af_array, af.Array):
        return ndarray(new_af_array)
    else:
        return new_af_array


def cov(m: ndarray,
        y: tp.Optional[ndarray] = None,
        rowvar: bool = True,
        bias: bool = False,
        ddof: tp.Optional[int] = None,
        fweights: tp.Optional = None,
        aweights: tp.Optional = None):
    """
    Estimate a covariance matrix, given data and weights.
    """

    isbiased = bias
    if ddof is not None and ddof == 0:
        isbiased = True

    new_af_array = af.cov(m._af_array, isbiased=isbiased, dim=0)
    if isinstance(new_af_array, af.Array):
        return ndarray(new_af_array)
    else:
        return new_af_array


def corrcoef(x,
             y: tp.Optional = None,
             rowvar: int = 1,
             bias: bool = False,
             ddof: tp.Optional[int] = None):
    """
    Return Pearson product-moment correlation coefficients.
    """

    isbiased = bias
    if ddof and ddof == 0:
        isbiased = True

    cov_array = af.cov(x._af_array, isbiased=isbiased, dim=0)
    std_array = af.stdev(x._af_array, dim=0)

    if isinstance(std_array, af.Array) and isinstance(cov_array, af.Array):
        new_array = cov_array / af.matmulNT(std_array, std_array)
        return ndarray(new_array)
    elif (isinstance(std_array, af.Array) and
          not isinstance(cov_array, af.Array)) or \
            (not isinstance(std_array, af.Array) and
             isinstance(cov_array, af.Array)):
        raise ValueError("error: either both std_array and cov_array must be "
                         "arrayfire arrays or none of them")
    else:
        return cov_array / std_array * std_array


def _division_with_remainder(dividend: ndarray,
                             divisor_int: int) \
        -> tp.Tuple[ndarray, ndarray]:

    # divisor_array = array(divisor_int)
    whole_part = dividend / divisor_int
    remainder = dividend % divisor_int
    return whole_part, remainder


def nonzero(a: ndarray) -> tp.Tuple:
    index_af_array = af.where(a._af_array)

    tmp_array = ndarray(index_af_array)

    if a.ndim == 1:
        return (tmp_array, )
        # return tmp_array
    elif a.ndim == 2:
        return _division_with_remainder(tmp_array, a.shape[0])
    elif a.ndim == 3:
        tmp_array2, dimension2 = _division_with_remainder(tmp_array, a.shape[1])
        dimension0, dimension1 \
            = _division_with_remainder(tmp_array2, a.shape[0])

        return dimension0, dimension1, dimension2
    else:
        raise ValueError("function nonzero is not implemented for arrays with "
                         "more than three axes")


def trace(a: ndarray,
          offset: int=0) \
        -> tp.Union[int, float]:

    return af.sum(af.diag(a._af_array, num=offset))


def reciprocal(x: ndarray) -> ndarray:
    """
    Return the reciprocal of the argument, element-wise.
    """

    return 1.0 / x


def negative(x: ndarray) -> ndarray:
    """
    Numerical negative, element-wise.
    """

    return -x


def _wrap_af_array(af_array) -> ndarray:
    if isinstance(af_array, af.Array):
        return ndarray(af_array)
    else:
        return af_array


def _sort_internal(a: ndarray,
                   axis: int = -1,
                   ascending: bool = True) \
        -> af.Array:
    if axis is None:
        a = a.flatten()
        axis = 0
    elif axis == -1:
        axis = a.ndim - 1
    elif axis >= a.ndim:
        raise ValueError(f"Parameter axis must be between -1 and {a.ndim - 1}")

    return af.sort(a._af_array, dim=axis, is_ascending=ascending)


def all(a: ndarray,
        axis: tp.Optional[int] = None,
        out: tp.Optional[ndarray] = None,
        keepdims: bool = False) \
        -> tp.Union[bool, ndarray]:
    """
    Returns True if all elements evaluate to True.
    """

    if out:
        raise ValueError('out != None is not supported')

    return _wrap_af_array(af.all_true(a._af_array, dim=axis))


def any(a: ndarray,
        axis: tp.Optional[int] = None,
        out: tp.Optional[ndarray] = None,
        keepdims: bool = False) \
        -> tp.Union[bool, ndarray]:
    return _wrap_af_array(af.any_true(a._af_array, dim=axis))


def argmax(a: ndarray,
           axis: tp.Optional[int] = None,
           out: tp.Optional[ndarray] = None) -> ndarray:
    if out:
        raise ValueError('out != None is not supported')

    val, idx = af.imax(a._af_array, dim=axis)
    return _wrap_af_array(idx)


def argmin(a: ndarray,
           axis: tp.Optional[int] = None,
           out: tp.Optional[ndarray] = None) -> ndarray:
    if out:
        raise ValueError('out != None is not supported')

    val, idx = af.imin(a._af_array, dim=axis)
    return _wrap_af_array(idx)


def argsort(a: ndarray,
            axis: int = -1,
            ascending: bool = True) -> ndarray:
    out, indices = sort_argsort(a, axis, ascending)
    return indices


def conj(x: ndarray):
    """
    Return the complex conjugate, element-wise.
    """

    return _unary_function(x, af_func=af.conjg, np_func=np.conj)


def cumsum(a: ndarray,
           axis: tp.Optional[int] = None,
           dtype: tp.Optional[np.generic] = None,
           out: tp.Optional[ndarray] = None) \
        -> tp.Union[float, ndarray]:
    """
    Return the cumulative sum of the elements along a given axis.
    """

    if axis is None:
        flat_array = a.flatten()
        new_af_array = af.accum(flat_array._af_array)
    else:
        new_af_array = af.accum(a._af_array, dim=axis)

    return ndarray(new_af_array)


def prod(a: ndarray, axis: tp.Optional[int] = None) -> tp.Union[float, ndarray]:
    """
    Return the product of array elements over a given axis.
    """

    return _wrap_af_array(af.product(a._af_array, dim=axis))


def sort(a: ndarray, axis: int = -1, ascending: bool = True) -> ndarray:
    return ndarray(_sort_internal(a, axis, ascending))


def sort_argsort(a: ndarray, axis: int = -1, ascending: bool = True) \
        -> tp.Tuple[ndarray, ndarray]:
    if axis is None:
        a = a.flatten()
        axis = 0
    elif axis == -1:
        axis = a.ndim - 1
    elif axis >= a.ndim:
        raise ValueError(f"Parameter axis must be between -1 and {a.ndim - 1}")

    af_out_array, af_idx_array \
        = af.sort_index(a._af_array, dim=axis, is_ascending=ascending)

    return ndarray(af_out_array), ndarray(af_idx_array)


def sum(a: ndarray, axis: tp.Optional[int] = None) \
        -> tp.Union[numbers.Number, ndarray]:
    """
    Sum of array elements over a given axis.
    """

    return _wrap_af_array(af.sum(a._af_array, dim=axis))


# from ._data
def _full_internal(shape: tp.Tuple[int, ...],
                   value,
                   dtype: np.generic = np.float32) -> af.Array:
    af_type = convert_numpy_to_af_type(dtype)

    d0, d1, d2, d3 = _pad_shape_tuple_none(shape)
    return af.data.constant(value, d0=d0, d1=d1, d2=d2, d3=d3, dtype=af_type)


def full(shape: tp.Tuple[int, ...], fill_value, dtype: np.generic = np.float32) \
        -> ndarray:
    """
    Return a new array of given shape and type, filled with fill_value.
    """

    return ndarray(_full_internal(shape, fill_value, dtype))


def full_like(a, fill_value, dtype: tp.Optional[np.generic] = None) -> ndarray:
    """
    Return a full array with the same shape and type as a given array.
    """

    if dtype is None:
        dtype = a.dtype
    return full(a.shape, fill_value, dtype)


def tile(A: ndarray, reps: tp.Union[int, tp.Tuple[int, ...]]) -> ndarray:
    af_array = None
    if isinstance(reps, tuple):
        d0, d1, d2, d3 = _pad_shape_tuple_one(reps)
        af_array = af.data.tile(A._af_array, d0, d1, d2, d3)
    elif isinstance(reps, int):
        if not reps > 0:
            raise ValueError("reps must be strictly positive")
        af_array = af.data.tile(A._af_array, reps)
    else:
        raise TypeError("types supported for reps are int an Tuple")

    return ndarray(af_array)


def repeat(A: ndarray, repeats: int, axis: tp.Optional[int] = None):
    if axis is None:
        raise ValueError('axis=None is not supported')

    reps = A.ndim * [1]
    reps[axis] = repeats

    return tile(A=A, reps=tuple(reps))


# from ._arith
def round(a: ndarray):
    """
    Return a with each element rounded to the given number of decimals.
    """

    return _unary_function(a, af_func=af.round, np_func=np.round)


def array_equal(a1: ndarray, a2: ndarray) -> bool:
    return all(a1 == a2)


def greater(x1: ndarray, x2: ndarray) -> ndarray:
    return x1 > x2


def greater_equal(x1: ndarray, x2: ndarray) -> ndarray:
    return x1 >= x2


def less(x1: ndarray, x2: ndarray) -> ndarray:
    return x1 < x2


def less_equal(x1: ndarray, x2: ndarray) -> ndarray:
    return x1 <= x2


def equal(x1: ndarray, x2: ndarray) -> ndarray:
    return x1 == x2


def not_equal(x1: ndarray, x2: ndarray) -> ndarray:
    return x1 != x2


def add(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Add arguments element-wise.
    """

    return x1 + x2


def subtract(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Subtract arguments, element-wise.
    """

    return x1 - x2


def true_divide(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Returns a true division of the inputs, element-wise.
    """

    return x1 / x2


def floor(a: ndarray):
    """
    Return the floor of the input, element-wise.
    """

    return _unary_function(a, af_func=af.floor, np_func=np.floor)


def floor_divide(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Return the largest integer smaller or equal to the division of the inputs.
    """

    return floor(x1 / x2)


def clip(a: ndarray, a_min, a_max) -> 'ndarray':
    """
    Clip (limit) the values in an array.
    """

    return a.clip(a_min, a_max)
