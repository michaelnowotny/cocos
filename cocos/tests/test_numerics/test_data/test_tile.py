import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest

test_data = [(np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 20]],
                       dtype=np.int32),
              (2, 3)),
             (np.array([[0.2, 1.0, 0.5],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.2, 0.25]],
                       dtype=np.float32),
              (2, 3)),
             (np.array([[0.5, 2.3, 3.1],
                        [4, 5.5, 6],
                        [7 - 9j, 8 + 1j, 2 + 10j]],
                       dtype=np.complex64),
              (2, 3)),
             (np.array([[[1.0, 2], [3, 4]],
                        [[5, 6], [7, 8]]],
                       dtype=np.float32),
              (2, 3, 1)),
             (np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 20]],
                       dtype=np.int32), (2, 1)),
             (np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 20]],
                       dtype=np.int32),
              (1, 3))
            ]


@pytest.mark.parametrize("A_numpy, tiles", test_data)
def test_tile(A_numpy, tiles):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    B_numpy = np.tile(A_numpy, tiles)
    B_cocos = cn.tile(A_cocos, tiles)

    assert np.allclose(B_cocos, B_numpy)
