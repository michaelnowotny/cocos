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
             # np.array([[0.5, 2.3, 3.1], [4, 5.5, 6], [7 - 9j, 8 + 1j, 2 + 10j]], dtype=np.complex64),
             np.array([[[1.0, 2], [3, 4]],
                       [[5, 6], [7, 8]]],
                      dtype=np.float32)
            ]


@pytest.mark.parametrize("A_numpy", test_data)
def test_ptp(A_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    axes = [None, 0, 1]

    for axis in axes:
        if np.iscomplexobj(A_numpy):
            ptp_cocos = A_cocos.ptp(axis=axis)
            ptp_numpy = A_numpy.ptp(axis=axis)
            print(np.array(ptp_cocos))
            print(np.array(ptp_numpy))
            assert np.allclose(ptp_cocos.real, ptp_numpy.real)
            assert np.allclose(ptp_numpy.imag, ptp_cocos.imag)
        else:
            assert np.allclose(A_cocos.ptp(axis=axis), A_numpy.ptp(axis=axis))
