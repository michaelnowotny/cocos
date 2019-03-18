import arrayfire as af
from arrayfire.arith import _arith_unary_func, _arith_binary_func
from arrayfire.array import _binary_func
from arrayfire.library import backend
import numbers
import numpy as np
import typing as tp

from ._array import ndarray
from ._array import _binary_function, _unary_function
from ._array import ndarray, isinteger
from ._data import full, zeros


def absolute(x: ndarray) -> ndarray:
    """
    Calculate the absolute value element-wise.
    """

    return _unary_function(x, af.abs)


@af.broadcast
def minimum(x1: tp.Union[float, ndarray],
            x2: tp.Union[float, ndarray]) \
        -> ndarray:
    """
    Element-wise minimum of array elements.
    """

    return _binary_function(x1, x2, af.minof)


@af.broadcast
def maximum(x1: tp.Union[float, ndarray],
            x2: tp.Union[float, ndarray]) \
        -> ndarray:
    """
    Element-wise maximum of array elements.
    """

    return _binary_function(x1, x2, af.maxof)


def sqrt(x: tp.Union[float, ndarray]) \
        -> ndarray:
    """
    Return the positive square-root of an array, element-wise.
    """

    return _unary_function(x, af.sqrt)


def square(x: ndarray) -> ndarray:
    """
    Return the element-wise square of the input.
    """

    return x * x


def exp(x: tp.Union[numbers.Number, ndarray]) -> ndarray:
    """
    Calculate the exponential of all elements in the input array.
    """

    return _unary_function(x, af.exp)


def log(x: tp.Union[float, ndarray]) -> ndarray:
    """
    Natural logarithm, element-wise.
    """

    return _unary_function(x, af.log)


def iszero(x: ndarray) -> ndarray:
    return _unary_function(x, af.iszero)


def isinf(x: ndarray) -> ndarray:
    return _unary_function(x, af.isinf)


def isnan(x: ndarray) -> ndarray:
    return _unary_function(x, af.isnan)


@af.broadcast
def logical_and(x1: ndarray,
                x2: ndarray):
    new_af_array \
        = _binary_func(x1._af_array,
                       x2._af_array,
                       backend.get().af_and)

    return ndarray(new_af_array)


@af.broadcast
def logical_or(x1: ndarray,
               x2: ndarray):
    new_af_array = _binary_func(x1._af_array,
                                x2._af_array,
                                backend.get().af_or)

    return ndarray(new_af_array)


def logical_not(x: ndarray):
    new_af_array = _arith_unary_func(x._af_array,
                                     backend.get().af_not)

    return ndarray(new_af_array)


@af.broadcast
def logical_xor(x1: ndarray,
                x2: ndarray):
    return bitwise_xor(x1.astype(np.bool8), x2.astype(np.bool8))
    # return logical_and(logical_or(a,b),(logical_not(logical_and(a, b))))


@af.broadcast
def bitwise_and(x1: ndarray,
                x2: ndarray):
    new_af_array \
        = _binary_func(x1._af_array, x2._af_array, backend.get().af_bitand)
    return ndarray(new_af_array)


@af.broadcast
def bitwise_or(x1: ndarray,
               x2: ndarray):
    new_af_array \
        = _binary_func(x1._af_array, x2._af_array, backend.get().af_bitor)
    return ndarray(new_af_array)


@af.broadcast
def invert(x: ndarray):
    if not isinteger(x):
        raise TypeError("invert only supportd integers arguments")
    af_array = x._af_array
    return ndarray(_binary_func(af_array, -1, backend.get().af_bitxor))


@af.broadcast
def bitwise_xor(x1: ndarray,
                x2: ndarray):
    new_af_array \
        = _binary_func(x1._af_array, x2._af_array, backend.get().af_bitxor)
    return ndarray(new_af_array)
    # return a.__xor__(b)


@af.broadcast
def left_shift(x1: ndarray,
               x2: ndarray):
    return x1 << x2


@af.broadcast
def right_shift(x1: ndarray,
                x2: ndarray):
    return x1 >> x2


@af.broadcast
def remainder(x1: ndarray,
              x2: ndarray):
    """
    Return element-wise remainder of division.
    """

    if isinstance(x1, ndarray):
        x1 = x1._af_array
    if isinstance(x2, ndarray):
        x2 = x2._af_array

    return ndarray(af.rem(x1, x2))


@af.broadcast
def mod(x1: ndarray,
        x2: ndarray):
    """
    Return element-wise remainder of division.
    """

    # return _binary_function(a, b, af.rem)
    new_af_array \
        = _arith_binary_func(x1._af_array, x2._af_array, backend.get().af_mod)

    return ndarray(new_af_array)


# def absolute(source: ndarray):
#     return abs(source)


def angle(z: ndarray):
    """
    Return the angle of the complex argument.
    """

    return _unary_function(z, af.arg)


def sign(x: ndarray):
    """
    Returns an element-wise indication of the sign of a number.
    """

    intermediate = _unary_function(x, af.sign)
    z = zeros(x.shape, dtype=x.dtype)
    result = (x > z) - intermediate

    return result


def trunc(a: ndarray):
    """
    Return the truncated value of the input, element-wise.
    """

    return _unary_function(a, af.trunc)


def ceil(a: ndarray):
    """
    Return the ceiling of the input, element-wise.
    """

    return _unary_function(a, af.ceil)


@af.broadcast
def hypot(x1: ndarray,
          x2: ndarray):
    """
    Given the “legs” of a right triangle, return its hypotenuse.
    """

    return _binary_function(x1, x2, af.hypot)


def sin(x: ndarray):
    """
    Trigonometric sine, element-wise.
    """

    return _unary_function(x, af.sin)


def cos(x: ndarray):
    """
    Cosine element-wise.
    """

    return _unary_function(x, af.cos)


def tan(x: ndarray):
    """
    Compute tangent element-wise.
    """

    return _unary_function(x, af.tan)


def arcsin(x: ndarray):
    """
    Inverse sine, element-wise.
    """

    return _unary_function(x, af.asin)


def arccos(x: ndarray):
    """
    Trigonometric inverse cosine, element-wise.
    """

    return _unary_function(x, af.acos)


def arctan(x: ndarray):
    """
    Trigonometric inverse tangent, element-wise.
    """

    return _unary_function(x, af.atan)


@af.broadcast
def arctan2(a: ndarray,
            b: ndarray):
    """
    Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
    """

    return _binary_function(a, b, af.atan2)


@af.broadcast
def cplx(a: ndarray, b: tp.Optional[ndarray] = None):
    return _binary_function(a, b, af.cplx)


def sinh(a: ndarray):
    """
    Hyperbolic sine, element-wise.
    """

    return _unary_function(a, af.sinh)


def cosh(a: ndarray):
    """
    Hyperbolic cosine, element-wise.
    """

    return _unary_function(a, af.cosh)


def tanh(a: ndarray):
    """
    Compute hyperbolic tangent element-wise.
    """

    return _unary_function(a, af.tanh)


def arcsinh(a: ndarray):
    """
    Inverse hyperbolic sine element-wise.
    """

    return _unary_function(a, af.asinh)


def arccosh(a: ndarray):
    """
    Inverse hyperbolic cosine, element-wise.
    """

    return _unary_function(a, af.acosh)


def arctanh(a: ndarray):
    """
    Inverse hyperbolic tangent element-wise.
    """

    return _unary_function(a, af.atanh)


@af.broadcast
def root(a: ndarray,
         b: ndarray):
    return _binary_function(a, b, af.root)


@af.broadcast
def power(x1: ndarray,
          x2: ndarray):
    """
    First array elements raised to powers from second array, element-wise.
    """

    return _binary_function(x1, x2, af.pow)


def power2(a: ndarray):
    return _unary_function(a, af.pow2)


def expit(a: ndarray):
    return _unary_function(a, af.sigmoid)


def expm1(x: ndarray):
    """
    Calculate exp(x) - 1 for all elements in the array.
    """

    return _unary_function(x, af.expm1)


def erf(a: ndarray):
    return _unary_function(a, af.erf)


def erfc(a: ndarray):
    return _unary_function(a, af.erfc)


def log1p(a: ndarray):
    """
    Return the natural logarithm of one plus the input array,
    element-wise.
    """

    return _unary_function(a, af.log1p)


def log10(a: ndarray):
    """
    Return the base 10 logarithm of the input array, element-wise.
    """

    return _unary_function(a, af.log10)


def log2(a: ndarray):
    """
    Base-2 logarithm of x.
    """

    return _unary_function(a, af.log2)


def cbrt(x: ndarray):
    """
    Return the cube-root of an array, element-wise.
    """

    return _unary_function(x, af.cbrt)


def factorial(a: ndarray):
    return _unary_function(a, af.factorial)


def gamma(a: ndarray):
    return _unary_function(a, af.tgamma)


def gammaln(a: ndarray):
    return _unary_function(a, af.lgamma)


def isfinite(a: ndarray):
    inf = isinf(a)
    nan = isnan(a)
    return logical_not(logical_or(inf, nan))


# @af.broadcast
def isneginf(a: ndarray):
    neginf = full((1,), np.NINF, a.dtype)
    return a == neginf


# @af.broadcast
def isposinf(a: ndarray):
    posinf = full((1,), np.inf, a.dtype)
    return a == posinf
