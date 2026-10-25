import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest

test_data = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 20]],
                      dtype=np.int32),
             np.array([[0.2, 1.0, 0.5], [0.4, 0.5, 0.6], [0.7, 0.2, 0.25]],
                      dtype=np.float32)]


@pytest.mark.parametrize("A", test_data)
def test_var(A):
    cocos.device.init()
    cocos.device.info()
    A_arch = cn.array(A)

    # tests variance
    for ddof in (0, 1):
        assert np.allclose(np.var(A, ddof=ddof), cn.var(A_arch, ddof=ddof))
        assert np.allclose(np.var(A, axis=0, ddof=ddof),
                           cn.var(A_arch, axis=0, ddof=ddof))
        assert np.allclose(np.var(A, axis=1, ddof=ddof),
                           cn.var(A_arch, axis=1, ddof=ddof))
        assert np.allclose(np.var(A, ddof=ddof), A_arch.var(ddof=ddof))

    # tests standard deviation
    for ddof in (0, 1):
        assert np.allclose(np.std(A, ddof=ddof), cn.std(A_arch, ddof=ddof))
        assert np.allclose(np.std(A, axis=0, ddof=ddof),
                           cn.std(A_arch, axis=0, ddof=ddof))
        assert np.allclose(np.std(A, axis=1, ddof=ddof),
                           cn.std(A_arch, axis=1, ddof=ddof))
        assert np.allclose(np.std(A, ddof=ddof), A_arch.std(ddof=ddof))

    with pytest.raises(ValueError):
        cn.var(A_arch, ddof=2)

    # tests median
    assert np.allclose(np.median(A), cn.median(A_arch))
    assert np.allclose(np.median(A, axis=0), cn.median(A_arch, axis=0))
    assert np.allclose(np.median(A, axis=1), cn.median(A_arch, axis=1))


x = np.array([[0.0, 1.0, 2.0, 3.0, 5.0],
              [1.0, 0.5, 0.25, 4.0, -1.0],
              [2.0, -1.0, 0.0, 1.5, 0.5]])
y = np.array([[1.0, 2.0, 0.0, -2.0, 1.0]])
cov_test_data = [(x, None, {}),
                 (x, None, {'bias': True}),
                 (x, None, {'ddof': 0}),
                 (x, None, {'ddof': 2}),
                 (x.T, None, {'rowvar': False}),
                 (x, y, {}),
                 (x[0], x[1], {}),
                 (x.astype(np.float32), None, {}),
                 (np.array([[1, 2, 4], [3, 3, 9]], dtype=np.int32), None, {}),
                 (x + 1j * x[::-1], None, {})]


@pytest.mark.parametrize("m, y, kwargs", cov_test_data)
def test_cov_corrcoef(m, y, kwargs):
    cocos.device.init()
    if np.iscomplexobj(m) and not cocos.device.is_dbl_supported():
        m = m.astype(np.complex64)
    m_cocos = cn.array(m)
    y_cocos = None if y is None else cn.array(y)

    assert np.allclose(np.cov(m, y, **kwargs),
                       np.array(cn.cov(m_cocos, y_cocos, **kwargs)),
                       rtol=1e-4)

    corrcoef_kwargs = {key: value for key, value in kwargs.items()
                       if key == 'rowvar'}
    assert np.allclose(np.corrcoef(m, y, **corrcoef_kwargs),
                       np.array(cn.corrcoef(m_cocos, y_cocos,
                                            **corrcoef_kwargs)),
                       rtol=1e-4)


def test_cov_of_a_single_variable():
    cocos.device.init()
    v = np.array([1.0, 2.0, 4.0, 7.0])
    assert np.isclose(np.cov(v), cn.cov(cn.array(v)))
    assert np.isclose(np.cov(v, ddof=0), cn.cov(cn.array(v), ddof=0))
    assert np.isclose(np.corrcoef(v), cn.corrcoef(cn.array(v)))


def test_var_std_complex():
    cocos.device.init()
    dtype = np.complex128 if cocos.device.is_dbl_supported() else np.complex64
    z = np.array([[1 + 2j, 3 - 1j, -2 + 0.5j],
                  [0.5, 2j, 1 - 1j]], dtype=dtype)
    z_cocos = cn.array(z)
    for ddof in (0, 1):
        assert np.allclose(np.var(z, ddof=ddof), cn.var(z_cocos, ddof=ddof))
        assert np.allclose(np.std(z, ddof=ddof), cn.std(z_cocos, ddof=ddof))
        for axis in (0, 1):
            assert np.allclose(np.var(z, axis=axis, ddof=ddof),
                               np.array(cn.var(z_cocos, axis=axis, ddof=ddof)))
            assert np.allclose(np.std(z, axis=axis, ddof=ddof),
                               np.array(cn.std(z_cocos, axis=axis, ddof=ddof)))


def test_corrcoef_keeps_nan_of_constant_variables():
    cocos.device.init()
    m = np.array([[1.0, 1.0, 1.0, 1.0],
                  [1.0, 2.0, 3.0, 5.0],
                  [4.0, 3.0, 2.0, 2.0]])
    with np.errstate(invalid='ignore', divide='ignore'):
        expected = np.corrcoef(m)
        assert np.allclose(expected,
                           np.array(cn.corrcoef(cn.array(m))),
                           equal_nan=True)
        assert np.isnan(cn.corrcoef(cn.array(np.ones(4))))


def test_cov_corrcoef_of_a_complex_variable():
    cocos.device.init()
    dtype = np.complex128 if cocos.device.is_dbl_supported() else np.complex64
    z = np.array([1 + 1j, 2 - 1j, 3j], dtype=dtype)
    assert np.isclose(np.cov(z), cn.cov(cn.array(z)))
    assert np.isclose(np.corrcoef(z), cn.corrcoef(cn.array(z)))


@pytest.mark.parametrize("m_dtype, y_dtype", [(np.float32, np.float64),
                                              (np.int32, np.float32),
                                              (np.float64, np.complex128)])
def test_cov_of_mixed_types(m_dtype, y_dtype):
    cocos.device.init()
    if not cocos.device.is_dbl_supported() \
            and np.dtype(np.result_type(m_dtype, y_dtype)).itemsize >= 8:
        pytest.skip("double precision is not supported by this device")
    m = x.astype(m_dtype)
    y_ = (y[0] + 1j * y[0][::-1]).astype(y_dtype) \
        if np.issubdtype(y_dtype, np.complexfloating) else y.astype(y_dtype)
    assert np.allclose(np.cov(m, y_),
                       np.array(cn.cov(cn.array(m), cn.array(y_))),
                       rtol=1e-4)
