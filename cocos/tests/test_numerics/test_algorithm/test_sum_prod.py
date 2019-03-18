import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 20]],
                      dtype=np.int32),
             np.array([[0.2, 1.0, 0.5],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.2, 0.25]],
                      dtype=np.float32),
             np.array([[0.5, 2.3, 3.1],
                       [4, 5.5, 6],
                       [7 - 9j, 8 + 1j, 2 + 10j]],
                      dtype=np.complex64)]


@pytest.mark.parametrize("A_numpy", test_data)
def test_sum_prod(A_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    # # using numpy
    # mean_numpy = np.mean(A)
    #
    # # using Cocos

    # mean_cocos = cn.mean(A_cocos)

    # conduct tests

    # tests sum
    assert np.allclose(np.sum(A_numpy), cn.sum(A_cocos))
    assert np.allclose(np.sum(A_numpy, axis=0), cn.sum(A_cocos, axis=0))
    assert np.allclose(np.sum(A_numpy, axis=1), cn.sum(A_cocos, axis=1))

    # tests prod
    assert np.allclose(np.prod(A_numpy), cn.prod(A_cocos))
    assert np.allclose(np.prod(A_numpy, axis=0), cn.prod(A_cocos, axis=0))
    assert np.allclose(np.prod(A_numpy, axis=1), cn.prod(A_cocos, axis=1))

    # tests cumsum
    assert np.allclose(np.cumsum(A_numpy.transpose()), cn.cumsum(A_cocos))
    assert np.allclose(np.cumsum(A_numpy, axis=0), cn.cumsum(A_cocos, axis=0))
    assert np.allclose(np.cumsum(A_numpy, axis=1), cn.cumsum(A_cocos, axis=1))
