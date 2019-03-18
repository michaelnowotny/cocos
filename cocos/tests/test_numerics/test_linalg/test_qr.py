import cocos.device
import cocos.numerics as cn
import cocos.numerics.linalg as linalg
import numpy as np


def compare_cocos_numpy(cocos_array, numpy_array):
    return np.allclose(np.array(cocos_array), numpy_array)


def test_qr():
    cocos.device.init()

    A_numpy = np.array([[1 + 2j, 2, 3], [4, 5 + 1j, 6], [7, 8, 20 + 7j]],
                       dtype=np.complex64)

    Q, R = np.linalg.qr(A_numpy)
    h, tau = np.linalg.qr(A_numpy, 'raw')
    print("Q numpy")
    print(Q)
    print("")

    print("R numpy")
    print(R)
    print("")

    print("tau numpy")
    print(tau)
    print("")

    A_cocos = cn.array(A_numpy)
    print(A_cocos.dtype)
    Q_cocos, R_cocos, T_cocos = cn.linalg.qr(A_cocos)
    print("Q cocos")
    print(np.array(Q_cocos))
    print("")

    print("R cocos")
    print(np.array(R_cocos))
    print("")

    print("tau cocos")
    print(np.array(T_cocos))
    print("")

    assert compare_cocos_numpy(Q_cocos, Q)
    assert compare_cocos_numpy(R_cocos, R)
    assert compare_cocos_numpy(T_cocos, tau)
