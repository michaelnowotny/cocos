import cocos.device
import cocos.numerics as cn
import numpy as np


def test_union_intersection():
    cocos.device.init()

    A_numpy = np.array([1, 2, 3], dtype=np.int32)
    B_numpy = np.array([1, 5], dtype=np.int32)

    A_cocos = cn.array(A_numpy)
    B_cocos = cn.array(B_numpy)

    union_AB_numpy = np.union1d(A_numpy, B_numpy)
    intersection_AB_numpy = np.union1d(A_numpy, B_numpy)

    union_AB_cocos = cn.union1d(A_cocos, B_cocos)
    intersection_AB_cocos = cn.union1d(A_cocos, B_cocos)

    assert np.allclose(union_AB_cocos, union_AB_numpy)
    assert np.allclose(intersection_AB_cocos, intersection_AB_numpy)

