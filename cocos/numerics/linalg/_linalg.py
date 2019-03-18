import arrayfire as af

from numbers import Number
import numpy as np
import typing as tp

from cocos.numerics._array import ndarray, isvector
from cocos.numerics._array import _unary_function
from cocos.numerics._data import zeros
from cocos.scientific.linalg import cholesky as chol_sc
from cocos.scientific.linalg import qr as qr_sc
from cocos.scientific.linalg import svd as svd_sc
from cocos.utilities import convert_trans_to_af_matprop


def cholesky(a: ndarray) -> ndarray:
    """
    Cholesky decomposition.
    """

    return chol_sc(a)


def qr(a: ndarray) -> tp.Union[tp.Tuple[ndarray, ndarray, ndarray], ndarray]:
    """
    Compute the qr factorization of a matrix."""

    return qr_sc(a)


def svd(a: ndarray,
        full_matrices: bool=True,
        compute_uv: bool=True) \
        -> tp.Union[tp.Tuple[ndarray, ndarray, ndarray], ndarray]:
    """
    Singular Value Decomposition.
    """

    return svd_sc(a, full_matrices, compute_uv)


def solve(a: ndarray,
          b: ndarray,
          trans: int = 0) -> ndarray:
    """
    Solve a linear matrix equation, or system of linear scalar equations.
    """

    options = convert_trans_to_af_matprop(trans)
    return ndarray(af.solve(a._af_array, b._af_array, options=options))


def inv(a: ndarray) -> ndarray:
    """
    Compute the (multiplicative) inverse of a matrix.
    """

    return _unary_function(a, af.inverse)


def matrix_rank(M: ndarray,
                tol: tp.Optional[float]=None) -> int:
    """
    Return matrix rank of array using SVD method
    """

    if tol is None:
        tol = 1e-5
    elif not isinstance(tol, Number):
        raise TypeError("tol must be numeric")

    return af.rank(M._af_array, tol)


def det(a: ndarray) -> float:
    """
    Compute the determinant of an array.
    """

    return af.det(a._af_array)


def norm(x: ndarray,
         ord: tp.Optional[int]=None,
         axis: tp.Optional[int]=None,
         keepdims: bool = False) -> float:
    """
    Matrix or vector norm.
    """

    if axis is not None:
        raise ValueError("axis != None is not supported")

    p = 1.0
    q = 1.0

    if ord == 2:
        S = svd(x, compute_uv=False)
        return S.max()
    else:
        if ord is None:
            norm_type = af.NORM.VECTOR_2
        elif isinstance(ord, Number):
            if ord == float("inf") or ord == np.inf:
                norm_type = af.NORM.MATRIX_INF
            elif ord == 1:
                norm_type = af.NORM.MATRIX_1
            else:
                norm_type = af.NORM.VECTOR_P
                p = ord
        else:
            raise ValueError(f"ord = {ord} is not supported")

        return af.norm(x._af_array, norm_type=norm_type, p=p, q=q)
