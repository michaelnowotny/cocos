import cocos.device
import cocos.numerics as cn
import cocos.numerics.linalg as linalg
import numpy as np
import pytest

test_data = [(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]],
                       dtype=np.float64),
             np.array([[100, 1000], [200, 2000], [300, 3000]],
                      dtype=np.float64))]


@pytest.mark.parametrize("A_numpy, b_numpy", test_data)
def test_solve(A_numpy, b_numpy):
    cocos.device.init()
    if cocos.device.is_dbl_supported():
        # using numpy
        # print(A)
        solution = np.linalg.solve(A_numpy, b_numpy)

        # using Cocos
        A_cocos = cn.array(A_numpy)
        b_cocos = cn.array(b_numpy)

        solution_cocos = cn.linalg.solve(A_cocos, b_cocos)

        # conduct tests
        assert np.allclose(solution, solution_cocos)