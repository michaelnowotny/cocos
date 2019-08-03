import numpy as np
from numpy.core.multiarray import array

from cocos.numerics._arith import \
    absolute, \
    arccos, \
    arccosh, \
    angle, \
    arcsin, \
    arcsinh, \
    arctan, \
    arctan2, \
    arctanh, \
    ceil, \
    cbrt, \
    cos, \
    cosh, \
    exp, \
    expm1, \
    erf, \
    erfc, \
    factorial, \
    gamma, \
    hypot, \
    log, \
    log10, \
    gammaln, \
    mod, \
    power, \
    root, \
    sign, \
    sin, \
    sinh, \
    sqrt, \
    tan, \
    tanh

from cocos.numerics._array import \
    conj, \
    floor, \
    imag, \
    real

# from cocos.numerics import \
#     arccos, \
#     arccosh, \
#     angle, \
#     arcsin, \
#     arcsinh, \
#     arctan, \
#     arctan2, \
#     arctanh, \
#     ceil, \
#     cbrt, \
#     conj, \
#     cos, \
#     cosh, \
#     exp, \
#     expm1, \
#     erf, \
#     erfc, \
#     factorial, \
#     floor, \
#     gamma, \
#     hypot, \
#     imag, \
#     log, \
#     log10, \
#     gammaln, \
#     mod, \
#     power, \
#     real, \
#     root, \
#     sign, \
#     sin, \
#     sinh, \
#     sqrt, \
#     tan, \
#     tanh

inf = np.inf
COCOS_TRANSLATIONS = {
    "abs": absolute,
    "acos": arccos,
    "acosh": arccosh,
    "arg": angle,
    "asin": arcsin,
    "asinh": arcsinh,
    "atan": arctan,
    "atan2": arctan2,
    "atanh": arctanh,
    "ceiling": ceil,
    "cbrt": cbrt,
    "conjugate": conj,
    "cos": cos,
    "cosh": cosh,
    "exp": exp,
    "E": exp,
    "expm1": expm1,
    "erf": erf,
    "erfc": erfc,
    "fac": factorial,
    "factorial": factorial,
    "floor": floor,
    "gamma": gamma,
    "hypot": hypot,
    "I": 1j,
    "im": imag,
    "ln": log,
    "log": log,
    "log10": log10,
    "loggamma": gammaln,
    "Mod": mod,
    "oo": inf,
    "power": power,
    "re": real,
    "root": root,
    "sign": sign,
    "sin": sin,
    "sinh": sinh,
    "sqrt": sqrt,
    "tan": tan,
    "tanh": tanh,
    "SparseMatrix": array,
    "ImmutableSparseMatrix": array,
    "Matrix": array,
    "MutableDenseMatrix": array,
    "ImmutableMatrix": array,
    "ImmutableDenseMatrix": array,
}
