import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest

# backend = 'cpu'
# backend = 'cuda'
# backend = 'opencl'
backend = None

test_data = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]],
                      dtype=np.int32),
             np.array([[0.2, 1.0, 0.5], [0.4, 0.5, 0.6], [0.7, 0.2, 0.25]],
                      dtype=np.float32)]


@pytest.mark.parametrize("A", test_data)
def test_var(A):
    cocos.device.init(backend)
    cocos.device.info()
    A_arch = cn.array(A)

    # tests variance
    assert np.allclose(np.var(A), cn.var(A_arch))
    assert np.allclose(np.var(A, axis=0), cn.var(A_arch, axis=0))
    assert np.allclose(np.var(A, axis=1), cn.var(A_arch, axis=1))

    # tests standard deviation
    assert np.allclose(np.std(A), cn.std(A_arch))
    assert np.allclose(np.std(A, axis=0), cn.std(A_arch, axis=0))
    assert np.allclose(np.std(A, axis=1), cn.std(A_arch, axis=1))

    # tests median
    assert np.allclose(np.median(A), cn.median(A_arch))
    assert np.allclose(np.median(A, axis=0), cn.median(A_arch, axis=0))
    assert np.allclose(np.median(A, axis=1), cn.median(A_arch, axis=1))

    # tests covariance
    a = np.cov(A)
    # b = cn.cov(A_arch)
    # assert np.allclose(np.cov(A), cn.cov(A_arch))
    # assert np.allclose(np.median(A, axis=0), cn.median(A_arch, axis=0))
    # assert np.allclose(np.median(A, axis=1), cn.median(A_arch, axis=1))
