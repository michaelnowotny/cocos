import cocos.device
import cocos.numerics as cn
import cocos.scientific.linalg as csl
import numpy as np
import scipy.linalg as scipy_linalg


def compare_cocos_numpy(cocos_array, numpy_array):
    return np.allclose(np.array(cocos_array), numpy_array)


def test_lu():
    cocos.device.init()

    A_numpy = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]], dtype=np.float32)

    P, L, U = scipy_linalg.lu(A_numpy)
    # print("P")
    # print(P)
    # print("")
    #
    # print("L")
    # print(L)
    # print("")
    #
    # print("U")
    # print(U)
    # print("")

    A_cocos = cn.array(A_numpy)
    P_cocos, L_cocos, U_cocos = csl.lu(A_cocos)
    # print("P cocos")
    # print(np.array(P_cocos))
    # print("")
    #
    # print("L cocos")
    # print(np.array(L_cocos))
    # print("")
    #
    # print("U cocos")
    # print(np.array(U_cocos))
    # print("")

    assert compare_cocos_numpy(P_cocos, P)
    assert compare_cocos_numpy(L_cocos, L)
    assert compare_cocos_numpy(U_cocos, U)