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
def test_flip(A_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    # conduct tests
    assert np.allclose(np.fliplr(A_numpy), cn.fliplr(A_cocos))
    assert np.allclose(np.flipud(A_numpy), cn.flipud(A_cocos))
