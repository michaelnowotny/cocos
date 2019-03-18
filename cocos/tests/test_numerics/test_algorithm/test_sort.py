import cocos.numerics as cn
import numpy as np
import pytest


test_data = [np.array([[1, -2, 5, 6, 20, 50, 3, 6],
                       [-3, 4, 7, 8, 10, 60, 70, 100],
                       [-30, 40, 70, 80, 100, 600, 700, 1000],
                       [9, 3, 1, 27, 81, 84, 87, 78],
                       [90, 30, 10, 270, 810, 840, 870, 780]],
                      dtype=np.int32)]


@pytest.mark.parametrize("A_numpy", test_data)
def test_sort(A_numpy):
    cn.init()

    A_cocos = cn.array(A_numpy)

    for i in range(2):
        # argsort with axis
        argsort_numpy = np.argsort(A_numpy, axis=i)
        argsort_cocos = cn.argsort(A_cocos, axis=i)
        assert np.allclose(argsort_cocos, argsort_numpy)

        # sort with axis
        sort_numpy = np.sort(A_numpy, axis=i)
        sort_cocos = cn.sort(A_cocos, axis=i)
        assert np.allclose(sort_cocos, sort_numpy)

    # argsort without axis
    argsort_numpy = np.argsort(A_numpy, axis=None)
    argsort_cocos = cn.argsort(A_cocos.transpose(), axis=None)
    # print("argsort numpy")
    # print(argsort_numpy)
    # print("argsort cocos")
    # print(argsort_cocos)
    # assert np.allclose(argsort_cocos, argsort_numpy)

    # sort without axis
    sort_numpy = np.sort(A_numpy)
    sort_cocos = cn.sort(A_cocos)
    assert np.allclose(sort_cocos, sort_numpy)
