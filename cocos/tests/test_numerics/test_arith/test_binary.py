import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data_int = [(np.array([[1, 2, -3],
                            [4, -5, 6],
                            [7, 8, 20]],
                           dtype=np.int32),
                  np.array([[2, 1, 5],
                            [4, 5, 6],
                            [7, 2, 25]],
                           dtype=np.int32))]


test_data_float = [(np.array([[1, 2, -3],
                              [4, -5, 6],
                              [7, 8, 20]],
                             dtype=np.float32),
                    np.array([[2, 1, 5],
                              [4, 5, 6],
                              [7, 2, 25]],
                             dtype=np.float32))]


@pytest.mark.parametrize("A_numpy, B_numpy", test_data_float)
def test_binary_float(A_numpy, B_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)
    B_cocos = cn.array(B_numpy)

    assert np.allclose(A_cocos + B_cocos, A_numpy + B_numpy)
    assert np.allclose(A_cocos - B_cocos, A_numpy - B_numpy)
    assert np.allclose(A_cocos * B_cocos, A_numpy * B_numpy)
    assert np.allclose(A_cocos / B_cocos, A_numpy / B_numpy)
    assert np.allclose(cn.hypot(A_cocos, B_cocos),
                       np.hypot(A_numpy, B_numpy))

    assert np.allclose(cn.arctan2(A_cocos, B_cocos),
                       np.arctan2(A_numpy, B_numpy))

    assert np.allclose(cn.cplx(A_cocos, B_cocos),
                       A_numpy + 1j * B_numpy)

    assert np.allclose(np.power(A_numpy, B_numpy),
                       cn.power(A_cocos, B_cocos))


@pytest.mark.parametrize("A_numpy, B_numpy", test_data_int)
def test_binary_int(A_numpy, B_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)
    B_coocs = cn.array(B_numpy)

    assert np.allclose(cn.bitwise_and(A_cocos, B_coocs),
                       np.bitwise_and(A_numpy, B_numpy))

    assert np.allclose(cn.bitwise_or(A_cocos, B_coocs),
                       np.bitwise_or(A_numpy, B_numpy))

    assert np.allclose(cn.bitwise_xor(A_cocos, B_coocs),
                       np.bitwise_xor(A_numpy, B_numpy))

    assert np.allclose(cn.minimum(A_cocos, B_coocs),
                       np.minimum(A_numpy, B_numpy))

    assert np.allclose(cn.maximum(A_cocos, B_coocs),
                       np.maximum(A_numpy, B_numpy))

    assert np.allclose(cn.left_shift(A_cocos, B_coocs),
                       np.left_shift(A_numpy, B_numpy))

    assert np.allclose(cn.right_shift(A_cocos, B_coocs),
                       np.right_shift(A_numpy, B_numpy))

    assert np.allclose(A_cocos + B_coocs, A_numpy + B_numpy)
    assert np.allclose(A_cocos - B_coocs, A_numpy - B_numpy)
    assert np.allclose(A_cocos * B_coocs, A_numpy * B_numpy)
