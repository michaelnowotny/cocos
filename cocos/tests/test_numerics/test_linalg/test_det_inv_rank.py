import cocos.device
import cocos.numerics as cn
import cocos.numerics.linalg as cnl
import numpy as np
import pytest

test_data = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]],
                      dtype=np.float32),
             np.array([[1 + 2j, 2, 3], [4, 5 + 1j, 6], [7, 8, 20 + 7j]],
                      dtype=np.complex64)]


@pytest.mark.parametrize("A_numpy", test_data)
def test_det_inv_rank(A_numpy):
    cocos.device.init()

    # using numpy
    # print(A_numpy)

    det_numpy = np.linalg.det(A_numpy)
    rank_numpy = np.linalg.matrix_rank(A_numpy)
    inv_numpy = np.linalg.inv(A_numpy)

    print(f"determinant python = {det_numpy}")

    # using Cocos
    A_cocos = cn.array(A_numpy)

    det_cocos = cnl.det(A_cocos)
    rank_cocos = cnl.matrix_rank(A_cocos)
    inv_cocos = cnl.inv(A_cocos)
    print(f"determinant cocos= {det_cocos}")

    # conduct tests
    assert np.isclose(det_numpy, det_cocos)
    assert np.isclose(rank_numpy, rank_cocos)
    assert np.allclose(inv_numpy, inv_cocos)