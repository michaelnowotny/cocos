import numpy as np
import scipy.linalg as scipy_linalg

import cocos.device
import cocos.numerics as cn
import cocos.numerics.linalg


def compare_cocos_numpy(cocos_array, numpy_array):
    return np.allclose(np.array(cocos_array), numpy_array)


def test_cholesky():
    cocos.device.init()

    A_numpy = np.array([[1. + 0.j,  0. - 2.j], [0. + 2.j,  5. + 0.j]],
                       dtype=np.complex64)

    L_numpy = np.linalg.cholesky(A_numpy)
    # print("L_numpy")
    # print(L_numpy)
    # print("")

    A_cocos = cn.array(A_numpy)
    L_cocos = cn.linalg.cholesky(A_cocos)
    # print("L_numpy cocos")
    # print(np.array(L_cocos))
    # print("")

    assert compare_cocos_numpy(L_cocos, L_numpy)
