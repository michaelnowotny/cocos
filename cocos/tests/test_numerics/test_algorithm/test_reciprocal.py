import numpy as np
import pytest

import cocos.device
import cocos.numerics as cn

# backend = 'cpu'
# backend = 'cuda'
# backend = 'opencl'
backend = None


test_data = [np.array([[1.0, -2.0],
                       [-3.0, 4.0]],
                      dtype=np.float32),
             np.array([[2, -3.0],
                       [9, 5.0]],
                      dtype=np.float32),
             np.array([[-100, 50],
                       [1, 30]],
                      dtype=np.float32),
             np.array([[20, 50],
                       [15, 22]],
                      dtype=np.float32)]


def compare_numpy_and_cocos(cocos_array: cn.ndarray, numpy_array: np.ndarray):
    return np.allclose(np.array(cocos_array).astype(bool), numpy_array)


@pytest.mark.parametrize("x", test_data)
def test_reciprocal(x):
    cocos.device.init(backend)
    cocos.device.info()

    # using numpy
    y = np.reciprocal(x)

    # using Cocos
    x_cocos = cn.array(x)
    y_cocos = cn.reciprocal(x_cocos)

    assert np.allclose(y_cocos, y)
