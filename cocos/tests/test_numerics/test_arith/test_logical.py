import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [(np.array([[True, True, False],
                        [True, False, True],
                        [False, True, True]],
                       dtype=bool),
             np.array([[True, False, True],
                       [False, False, True],
                       [True, True, True]],
                      dtype=bool))]


@pytest.mark.parametrize("A_numpy, B_numpy", test_data)
def test_logical(A_numpy, B_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)
    B_cocos = cn.array(B_numpy)

    assert np.allclose(cn.logical_not(A_cocos),
                       np.logical_not(A_numpy))

    assert np.allclose(cn.logical_and(A_cocos, B_cocos),
                       np.logical_and(A_numpy, B_numpy))

    assert np.allclose(cn.logical_or(A_cocos, B_cocos),
                       np.logical_or(A_numpy, B_numpy))

    assert np.allclose(cn.logical_xor(A_cocos, B_cocos),
                       np.logical_xor(A_numpy, B_numpy))
