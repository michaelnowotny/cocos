import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [np.array([[1 + 2j, 2, 3],
                       [4, 5 + 1j, 6],
                       [7, 8, 20 + 7j]],
                      dtype=np.complex64),
             np.array([[0, 2, 3],
                       [4, 0, 6],
                       [7, 8, 9]],
                      dtype=np.int32)]


@pytest.mark.parametrize("A_numpy", test_data)
def test_nonzero(A_numpy):
    cocos.device.init()

    A_cocos = cn.array(A_numpy)

    # flatnonzero
    assert np.allclose(cn.flatnonzero(A_cocos), np.flatnonzero(A_numpy))

    # nonzero
    (nonzero_i, nonzero_j) = np.nonzero(A_numpy)
    (nonzero_i_self, nonzero_j_self) = A_numpy.nonzero()
    (nonzero_i_cocos, nonzero_j_cocos) = cn.nonzero(A_cocos)
    (nonzero_i_cocos_self, nonzero_j_cocos_self) = A_cocos.nonzero()
    # print("nonzero numpy i")
    # print(nonzero_i)
    # print("nonzero numpy j")
    # print(nonzero_j)
    # print("nonzero cocos i")
    # print(nonzero_i_cocos)
    # print("nonzero cocos i")
    # print(nonzero_j_cocos)
    assert np.allclose(nonzero_i_cocos, nonzero_i)
    assert np.allclose(nonzero_j_cocos, nonzero_j)
    assert np.allclose(nonzero_i_cocos_self, nonzero_i_self)
    assert np.allclose(nonzero_j_cocos_self, nonzero_j_self)

