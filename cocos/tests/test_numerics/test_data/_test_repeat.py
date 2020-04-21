import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest

y_numpy = np.array([[1, 2, 3], [4, 5, 6]])
y_cocos = cn.array(y_numpy)


@pytest.mark.parametrize('axis', (0, 1))
def test_repeat(axis: int):
    pass
    # result_numpy = np.repeat(y_numpy, repeats=2, axis=axis)
    # result_cocos = cn.repeat(y_cocos, repeats=2, axis=axis)
    #
    # assert np.all(result_numpy == np.array(result_cocos))
