import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]],
                      dtype=np.int32),
             np.array([[0.2, 1.0, 0.5], [0.4, 0.5, 0.6], [0.7, 0.2, 0.25]],
                      dtype=np.float32),
             np.array([[0.5, 2.3, 3.1], [4, 5.5, 6], [7 - 9j, 8 + 1j, 2 + 10j]],
                      dtype=np.complex64)]


@pytest.mark.parametrize("A", test_data)
def test_mean(A):
    cocos.device.init()
    A_arch = cn.array(A)

    # # using numpy
    # mean_numpy = np.mean(A)
    #
    # # using Archimedes

    # mean_arch = cn.mean(A_arch)

    # conduct tests

    # tests mean
    assert np.allclose(np.mean(A), cn.mean(A_arch))
    assert np.allclose(np.mean(A, axis=0), cn.mean(A_arch, axis=0))
    assert np.allclose(np.mean(A, axis=1), cn.mean(A_arch, axis=1))
