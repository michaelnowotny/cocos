import array as pyarray
import numpy as np
import pytest

import cocos.device
import cocos.numerics as cn


# backend = 'cpu'
# backend = 'cuda'
# backend = 'opencl'
backend = None


test_data = [(np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32),
              np.array([[0.0, -3.0], [0.0, 5.0]], dtype=np.float32)),
             (np.array([[-100, 50], [1, 30]], dtype=np.float32),
              np.array([[20, 50], [15, 22]], dtype=np.float32))]


def compare_numpy_and_cocos(cocos_array: cn.ndarray, numpy_array: np.ndarray):
    return np.allclose(np.array(cocos_array).astype(bool), numpy_array)


@pytest.mark.parametrize("x, y", test_data)
def test_any_all(x, y):
    cocos.device.init(backend)
    cocos.device.info()

    # using numpy
    # x = np.array([[1.0, -2.0], [-3.0, 4.0]])
    # y = np.array([[0.0, -3.0], [0.0, 5.0]])
    z1 = x > y
    z2 = x < y
    z3 = x == y

    # print("numpy result")
    # print(z1)
    # print("numpy dtype = {}".format(z1.dtype))

    # using Cocos
    a = cn.array(x)
    b = cn.array(y)
    c1 = a > b
    c2 = a < b
    c3 = a == b

    # print("cocos result")
    # print(c1)
    # print("cocos dtype = {}".format(c1.dtype))

    # assert np.allclose(np.array(c1).astype(bool), z1)
    # assert np.allclose(np.array(cn.all(c1)).astype(bool), np.all(z1))
    # assert np.allclose(np.array(cn.all(c1, 0)).astype(bool), np.all(z1, 0))
    # assert np.allclose(np.array(cn.all(c1, 1)).astype(bool), np.all(z1, 1))

    list_of_comparisons = [(c1, z1), (c2, z2), (c3, z3)]

    for c, z in list_of_comparisons:
        assert compare_numpy_and_cocos(c, z)
        assert compare_numpy_and_cocos(cn.all(c), np.all(z))
        assert compare_numpy_and_cocos(cn.all(c, 0), np.all(z, 0))
        assert compare_numpy_and_cocos(cn.all(c, 1), np.all(z, 1))

        assert compare_numpy_and_cocos(cn.any(c), np.any(z))
        assert compare_numpy_and_cocos(cn.any(c, 0), np.any(z, 0))
        assert compare_numpy_and_cocos(cn.any(c, 1), np.any(z, 1))

