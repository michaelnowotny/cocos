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
                      dtype=np.complex64)]


@pytest.mark.parametrize("A_numpy", test_data)
def test_diag_trace(A_numpy):
    if A_numpy.ndim != 2:
        raise ValueError("A_numpy must be a matrix")
    if A_numpy.shape[0] != A_numpy.shape[1]:
        raise ValueError("A_numpy must be a square matrix")

    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    # conduct tests
    for k in range(-1, 2):
        assert np.allclose(np.diag(A_numpy, k=k), cn.diag(A_cocos, k=k))

    # tests diag and trace
    diag_numpy = np.diag(A_numpy)
    diag_cocos = cn.diag(A_cocos)

    assert np.allclose(np.diag(diag_numpy), cn.diag(diag_cocos))
    assert np.isclose(np.trace(A_numpy), cn.trace(A_cocos))

    # tests diagonal
    diagonal_numpy = A_numpy.diagonal()
    diagonal_cocos = A_cocos.diagonal()

    assert np.allclose(diagonal_numpy, diagonal_cocos)

    d = A_numpy.shape[0]

    for offset in range(-d+1, d):
        diagonal_numpy = A_numpy.diagonal(offset=offset)
        diagonal_cocos = A_cocos.diagonal(offset=offset)
        assert np.allclose(diagonal_numpy, diagonal_cocos)

    # tests diagflat
    assert np.allclose(np.diagflat(diag_numpy), cn.diagflat(diag_cocos))
