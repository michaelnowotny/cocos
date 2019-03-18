import cocos.device
import cocos.numerics as cn
import cocos.numerics.linalg
import numpy as np
import pytest

test_data = [np.inf, 1, 2, None]


@pytest.mark.parametrize("ord", test_data)
def test_norm(ord):
    cocos.device.init()

    # using numpy
    A_numpy = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]], dtype=np.float32)
    print(A_numpy)

    norm = np.linalg.norm(A_numpy, ord)
    print(f"norm python = {norm}")

    # using Cocos
    A_cocos = cn.array(A_numpy)

    norm_cocos = cn.linalg.norm(A_cocos, ord)
    print(f"norm cocos = {norm_cocos}")

    # conduct tests
    assert np.isclose(norm, norm_cocos)
