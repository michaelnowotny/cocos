import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [(2, 2), (3, 3)]


@pytest.mark.parametrize("n, m", test_data)
def test_identity(n: int, m: int):
    cocos.device.init()

    # eye
    A_numpy = np.eye(n, m)
    A_cocos = cn.eye(n, m)

    assert np.allclose(A_numpy, A_cocos)

    # identity
    B_numpy = np.identity(n)
    B_cocos = cn.identity(n)

    assert np.allclose(B_numpy, B_cocos)
