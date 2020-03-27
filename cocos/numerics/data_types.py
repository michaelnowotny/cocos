from numbers import Number
import numpy as np
import typing as tp

import cocos.numerics as cn

NumericArray = tp.TypeVar('NumericArray',
                          np.ndarray,
                          cn.ndarray)

IntOrNumericArray = tp.TypeVar('IntOrNumericArray',
                               int,
                               np.ndarray,
                               cn.ndarray)

FloatOrNumericArray \
    = tp.TypeVar('FloatOrNumericArray',
                 float,
                 np.ndarray,
                 cn.ndarray)

ComplexOrNumericArray \
    = tp.TypeVar('ComplexOrNumericArray',
                 complex,
                 np.ndarray,
                 cn.ndarray)

ComplexOrFloatOrNumericArray \
    = tp.TypeVar('ComplexOrFloatOrNumericArray',
                 complex,
                 float,
                 np.ndarray,
                 cn.ndarray)

NumericArrayOrScalar = tp.TypeVar('NumericArrayOrScalar',
                                  np.ndarray,
                                  cn.ndarray, Number)
