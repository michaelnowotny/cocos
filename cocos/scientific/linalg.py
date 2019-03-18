import arrayfire as af
import numpy as np
import typing as tp

from cocos.numerics._array import ndarray, isvector
from cocos.numerics._data import zeros
from cocos.utilities import convert_trans_to_af_matprop


def cholesky(a: ndarray,
             overwrite: bool = False,
             is_upper: bool = False) -> ndarray:
    """
    Compute the Cholesky decomposition of a matrix.
    """

    if overwrite:
        info = af.cholesky_inplace(a._af_array, is_upper=is_upper)
        if info == 0:
            return a
        else:
            raise np.linalg.LinAlgError(f"cholesky decomposition failed at "
                                        f"rank {info}")
    else:
        R, info = af.cholesky(a._af_array, is_upper=is_upper)
        if info == 0:
            return ndarray(R)
        else:
            raise np.linalg.LinAlgError(f"cholesky decomposition failed at "
                                        f"rank {info}")


def qr(a: ndarray,
       mode: str = 'reduced',
       overwrite: bool = False) \
        -> tp.Union[tp.Tuple[ndarray, ndarray, ndarray], ndarray]:
    """
    Compute QR decomposition of a matrix.
    """

    if overwrite:
        return ndarray(af.qr_inplace(a._af_array))
    else:
        (Q, R, T) = af.qr(a._af_array)
        return ndarray(Q), ndarray(R), ndarray(T)


def svd(a: ndarray,
        full_matrices: bool = True,
        compute_uv: bool = True,
        overwrite: bool = False) \
        -> tp.Union[tp.Tuple[ndarray, ndarray, ndarray], ndarray]:
    """
    Singular Value Decomposition.
    """

    if not full_matrices:
        raise ValueError("full_matrices must be True")

    if overwrite:
        U, S, Vt = af.svd_inplace(a._af_array)
    else:
        U, S, Vt = af.svd(a._af_array)

    if compute_uv:
        return ndarray(U), ndarray(S), ndarray(Vt)
    else:
        return ndarray(S)


def _convert_pivot_matrix(pivot: ndarray):
    if not isvector(pivot):
        raise ValueError("pivot must be a vector")

    n = pivot.size
    out = zeros((n, n), dtype=np.int32)

    pivot_np = np.array(pivot)

    for i in range(n):
        out[pivot_np[i], i] = 1

    return out


def lu(a: np.ndarray,
       permute_l: bool = False,
       overwrite_a: bool = False,
       check_finite: bool = True,
       is_lapack_piv: bool = True):
    """
    Compute pivoted LU decomposition of a matrix.
    """

    if overwrite_a:
        pivot, LU = _lu_internal(a,
                                 permute_l=permute_l,
                                 overwrite_a=overwrite_a,
                                 is_lapack_piv=is_lapack_piv)

        return _convert_pivot_matrix(pivot), LU
    else:
        pivot, lower, upper = _lu_internal(a,
                                           permute_l=permute_l,
                                           overwrite_a=overwrite_a)

        return _convert_pivot_matrix(pivot), lower, upper


def lu_factor(a: ndarray,
              overwrite_a: bool = False,
              check_finite: bool = True) -> tp.Tuple[ndarray, ndarray]:
    """
    Compute pivoted LU decomposition of a matrix.
    """

    if not overwrite_a:
        raise ValueError("overwrite_a != True is not supported")

    pivot, LU = _lu_internal(a,
                             permute_l=True,
                             overwrite_a=overwrite_a,
                             is_lapack_piv=True)

    return LU, pivot


def _lu_internal(a: ndarray,
                 permute_l: bool = False,
                 overwrite_a: bool = False,
                 is_lapack_piv: bool = True) \
        -> tp.Union[tp.Tuple[ndarray, ndarray],
                    tp.Tuple[ndarray, ndarray, ndarray]]:
    if overwrite_a:
        if permute_l:
            if is_lapack_piv:
                pivot = "lapack"
            else:
                pivot = "full"

            P = af.lu_inplace(a._af_array, pivot=pivot)
            return ndarray(P), a
        else:
            raise ValueError("cannot overwrite a without permuting l")
    else:
        if permute_l:
            raise ValueError("cannot permute l without overwriting a")
        else:
            (L, U, P) = af.lu(a._af_array)
            return ndarray(P), ndarray(L), ndarray(U)


def lu_solve(lu_and_piv: tp.Tuple[ndarray, ndarray],
             b: ndarray,
             trans: int = 0,
             overwrite_b: bool = False,
             check_finite: bool = True) -> ndarray:
    """
    Solve an equation system, a x = b, given the LU factorization of a
    """

    if len(lu_and_piv) != 2:
        raise ValueError("lu_and_piv must be a two-dimensional tuple")

    lu, piv = lu_and_piv
    options = convert_trans_to_af_matprop(trans)

    return ndarray(af.solve_lu(lu._af_array,
                               piv._af_array,
                               b._af_array,
                               options=options))
