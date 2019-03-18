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
                       [0.7, 0.2, 0.25]],
                      dtype=np.float32),
             np.array([[0.5, 2.3, 3.1],
                       [4, 5.5, 6],
                       [7 - 9j, 8 + 1j, 2 + 10j]],
                      dtype=np.complex64),
             np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 20]], dtype=np.int32),
             np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 20]],
                      dtype=np.int32)
            ]


@pytest.mark.parametrize("A_numpy", test_data)
def test_squeeze_reshape(A_numpy):
    cocos.device.init()
    newshapes = [(3, 1, 3), (1, 3, 3)]
    axess = [(None, 1), (None, 0)]

    for newshape, axes in zip(newshapes, axess):
        A_cocos = cn.array(A_numpy)

        # 3, 1, 3
        B_numpy = A_numpy.reshape(newshape)
        B_cocos = A_cocos.reshape(newshape)

        assert np.allclose(B_cocos, B_numpy)

        for axis in axes:
            C_numpy = B_numpy.squeeze(axis=axis)
            C_cocos = B_cocos.squeeze(axis=axis)

            assert np.allclose(C_cocos, C_numpy)


def main():
    test_squeeze_reshape(test_data[0])


if __name__ == '__main__':
    main()
