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
                       [0.7, 0.3, 0.25]],
                      dtype=np.float32)]


@pytest.mark.parametrize("A_numpy", test_data)
def test_argmin(A_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    # tests argmin
    assert np.allclose(np.argmin(A_numpy.transpose()), cn.argmin(A_cocos))
    assert np.allclose(np.argmin(A_numpy, axis=0), cn.argmin(A_cocos, axis=0))
    assert np.allclose(np.argmin(A_numpy, axis=1), cn.argmin(A_cocos, axis=1))

    # tests argmax
    print("numpy argmax")
    print(np.argmax(A_numpy.transpose()))
    print("cocos argmax")
    print(cn.argmax(A_cocos))
    assert np.allclose(np.argmax(A_numpy.transpose()), cn.argmax(A_cocos))
    assert np.allclose(np.argmax(A_numpy, axis=0), cn.argmax(A_cocos, axis=0))
    assert np.allclose(np.argmax(A_numpy, axis=1), cn.argmax(A_cocos, axis=1))
