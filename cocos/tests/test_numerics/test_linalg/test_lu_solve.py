import cocos.device
import cocos.numerics as cn
import cocos.scientific.linalg as csl
import numpy as np
import pytest
import scipy.linalg as scipy_linalg

test_data = [(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]],
                       dtype=np.float64),
             np.array([[100, 1000], [200, 2000], [300, 3000]],
                      dtype=np.float64))]


@pytest.mark.parametrize("A_numpy, b_numpy", test_data)
def test_solve(A_numpy, b_numpy):
    cocos.device.init()

    if not cocos.device.is_dbl_supported():
        A_numpy = A_numpy.astype(np.float32)
        b_numpy = b_numpy.astype(np.float32)

    # using numpy
    lu, piv = scipy_linalg.lu_factor(A_numpy,
                                     overwrite_a=False)

    solution = scipy_linalg.lu_solve((lu, piv),
                                     b_numpy,
                                     trans=0,
                                     overwrite_b=False)

    # using Cocos
    b_cocos = cn.array(b_numpy)

    piv_cocos, lu_cocos = csl._lu_internal(cn.array(A_numpy),
                                           permute_l=True,
                                           overwrite_a=True)

    solution_cocos = csl.lu_solve((lu_cocos, piv_cocos),
                                  b_cocos,
                                  trans=0,
                                  overwrite_b=False)

    lu_cocos_2, piv_cocos_2 = csl.lu_factor(cn.array(A_numpy),
                                            overwrite_a=True)

    solution_cocos_2 = csl.lu_solve((lu_cocos_2, piv_cocos_2),
                                    b_cocos,
                                    trans=0,
                                    overwrite_b=False)

    # conduct tests
    assert np.allclose(solution, solution_cocos)
    assert np.allclose(solution, solution_cocos_2)
