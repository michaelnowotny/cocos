import numpy as np

from cocos.options import GPUOptions
import cocos.numerics.random as random
import cocos.numerics.linalg as linalg

if GPUOptions.use_gpu:
    # from arrayfire/algorithm.py
    from ._algorithm import \
        count_nonzero, \
        diff, \
        flatnonzero, \
        intersect1d, \
        sort_by_keys, \
        union1d, \
        unique

    from ._array import \
        add, \
        all, \
        any, \
        argmax, \
        argmin, \
        argsort, \
        divide, \
        cumsum, \
        floor_divide, \
        multiply, \
        prod, \
        sort, \
        sort_argsort, \
        subtract, \
        sum, \
        true_divide

    # from arrayfire/_array.py
    from ._array import \
        ndarray, \
        array, \
        transpose, \
        display, \
        reorder, \
        reshape, \
        reshape_without_reorder, \
        squeeze, \
        nonzero, \
        trace, \
        reciprocal
    from cocos.device import init, info, sync, is_dbl_supported, selected_backend

    from ._array import \
        isempty, \
        isscalar, \
        isrow, \
        iscolumn, \
        isvector, \
        isrealobj, \
        iscomplexobj, \
        isdouble, \
        issingle, \
        isrealfloating, \
        isfloating, \
        isinteger, \
        isbool
    from ._array import \
        array_equal, \
        greater, \
        greater_equal, \
        less, \
        less_equal, \
        equal, \
        not_equal

    # from arrayfire/_arith.py
    from ._arith import \
        abs, \
        absolute, \
        minimum, \
        maximum, \
        sqrt, \
        square, \
        exp, \
        log, \
        iszero, \
        isinf, \
        isnan, \
        logical_and, \
        logical_or, \
        logical_not, \
        logical_xor, \
        bitwise_and, \
        bitwise_or, \
        invert, \
        bitwise_xor, \
        left_shift, \
        right_shift, \
        remainder, \
        mod, \
        angle, \
        sign, \
        trunc, \
        ceil, \
        hypot, \
        sin, \
        cos, \
        tan, \
        arcsin, \
        arccos, \
        arctan, \
        arctan2, \
        cplx, \
        sinh, \
        cosh, \
        tanh, \
        arcsinh, \
        arccosh, \
        arctanh, \
        root, \
        power, \
        power2, \
        expit, \
        expm1, \
        erf, \
        erfc, \
        log1p, \
        log1p, \
        log10, \
        log2, \
        cbrt, \
        factorial, \
        gamma, \
        gammaln, \
        isfinite, \
        isneginf, \
        isposinf

    from ._array import \
        conj, \
        floor, \
        real, \
        imag, \
        round, \
        clip

    # from arrayfire/blas.py
    from ._array import dot

    # from arrayfire/data.py
    from ._array import (
        full,
        tile,
        # repeat
    )

    from ._data import \
        empty, \
        zeros, \
        ones, \
        empty_like, \
        zeros_like, \
        ones_like, \
        eye, \
        identity, \
        diag, \
        diagflat, \
        concatenate, \
        vstack, \
        hstack, \
        dstack, \
        roll, \
        fliplr, \
        flipud, \
        tril, \
        triu

    # from arrayfire/statistics.py
    from ._array import \
        average, \
        corrcoef, \
        cov, \
        median, \
        mean, \
        std, \
        var

    from . import random

    init(None)
else:
    def is_dbl_supported() -> bool:
        return True

    class ndarray(np.ndarray):
        @property
        def label(self) -> str:
            return self._label

        @label.setter
        def label(self, label):
            self._label = label

    def init(backend: str=None):
        pass

    def sync(device=None):
        pass

    def info():
        print("dummy gpu numerics emulated on cpu")

    def selected_backend() -> str:
        return "dummy numpy cpu"

    def isempty(num) -> bool:
        raise NotImplementedError()

    def isrow(num) -> bool:
        raise NotImplementedError()

    def iscolumn(num) -> bool:
        raise NotImplementedError()

    def isdouble(num) -> bool:
        raise NotImplementedError()

    def issingle(num) -> bool:
        raise NotImplementedError()

    def isrealfloating(num) -> bool:
        raise NotImplementedError()

    def isfloating(num) -> bool:
        raise NotImplementedError()

    def isinteger(num) -> bool:
        raise NotImplementedError()

    def isbool(num) -> bool:
        raise NotImplementedError()

    def iszero(num) -> bool:
        raise NotImplementedError()

    def cplx(a, b=None):
        raise NotImplementedError()

    def root(a, b):
        raise NotImplementedError()

    def power2(a):
        raise NotImplementedError()

    # from arrayfire/algorithm.py
    from numpy import \
        all, \
        any, \
        argmax, \
        argmin, \
        argsort, \
        cumsum, \
        multiply, \
        prod, \
        sort, \
        sum

    # from arrayfire/_array.py
    from numpy import \
        ndarray, \
        array, \
        transpose, \
        reshape, \
        squeeze, \
        nonzero, \
        trace, \
        reciprocal
    from numpy import \
        array_equal, \
        greater, \
        greater_equal, \
        less, \
        less_equal, \
        equal, \
        not_equal

    # from arrayfire/_arith.py

    # from arrayfire/blas.py
    from numpy import dot

    # from arrayfire/data.py

    # from arrayfire/statistics.py

    # from numpy import array, transpose, reshape, nonzero, trace, reciprocal, roll
    #
    # from numpy import absolute, minimum, maximum, sqrt, square, exp, log, isinf, isnan, logical_and, logical_or, logical_not, logical_xor, \
    #     bitwise_and, \
    #     bitwise_or, invert, bitwise_xor, left_shift, right_shift, remainder, mod, angle, sign, trunc, floor, ceil, \
    #     hypot, sin, cos, tan, arcsin, arccos, arctan, arctan2, real, imag, conj, sinh, cosh, tanh, arcsinh, arccosh, arctanh, power, \
    #     expm1, log1p, log10, log2, cbrt, isfinite

    # from numpy import dot

# ARCHIMEDES = {}
#
# ARCHIMEDES_DEFAULT = {"I": 1j}
#
# ARCHIMEDES_TRANSLATIONS = {
#     "acos": "arccos",
#     "acosh": "arccosh",
#     "arg": "angle",
#     "asin": "arcsin",
#     "asinh": "arcsinh",
#     "atan": "arctan",
#     "atan2": "arctan2",
#     "atanh": "arctanh",
#     "ceiling": "ceil",
#     "E": "e",
#     "im": "imag",
#     "ln": "log",
#     "Mod": "mod",
#     "oo": "inf",
#     "re": "real",
#     "SparseMatrix": "array",
#     "ImmutableSparseMatrix": "array",
#     "Matrix": "array",
#     "MutableDenseMatrix": "array",
#     "ImmutableMatrix": "array",
#     "ImmutableDenseMatrix": "array",
# }
#
# ARCHIMEDES_SYMPY = (ARCHIMEDES, ARCHIMEDES_DEFAULT, ARCHIMEDES_TRANSLATIONS, ("import_module('archimedes.numerics')",))
