import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [np.array([[1, -1],
                       [0, 1]],
                      dtype=np.int32),
             np.array([[0.2, 1.0, 0.5],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.2, 0.25]],
                      dtype=np.float32)]


@pytest.mark.parametrize("A_numpy", test_data)
def test_trigonometric(A_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    all_positive = np.all(A_numpy > 0)

    all_finite = cn.all(cn.isfinite(A_cocos))
    assert np.allclose(np.isfinite(A_numpy), cn.isfinite(A_cocos))

    if all_finite:
        assert np.allclose(np.sinh(A_numpy), cn.sinh(A_cocos))
        assert np.allclose(np.cosh(A_numpy), cn.cosh(A_cocos))
        assert np.allclose(np.tanh(A_numpy), cn.tanh(A_cocos))
        assert np.allclose(np.sin(A_numpy), cn.sin(A_cocos))
        assert np.allclose(np.cos(A_numpy), cn.cos(A_cocos))
        assert np.allclose(np.tan(A_numpy), cn.tan(A_cocos))

        assert np.allclose(np.arcsin(np.sin(A_numpy)),
                           cn.arcsin(cn.sin(A_cocos)))

        assert np.allclose(np.arccos(np.cos(A_numpy)),
                           cn.arccos(cn.cos(A_cocos)))

        assert np.allclose(np.arctan(np.tan(A_numpy)),
                           cn.arctan(cn.tan(A_cocos)))

        assert np.allclose(np.arcsinh(np.sinh(A_numpy)),
                           cn.arcsinh(cn.sinh(A_cocos)))

        assert np.allclose(np.arccosh(np.cosh(A_numpy)),
                           cn.arccosh(cn.cosh(A_cocos)))

        assert np.allclose(np.arctanh(np.tanh(A_numpy)),
                           cn.arctanh(cn.tanh(A_cocos)))

        if cn.isfloating(A_cocos):
            assert np.allclose(A_cocos, cn.arcsin(cn.sin(A_cocos)))
            assert np.allclose(A_cocos, cn.arccos(cn.cos(A_cocos)))
            assert np.allclose(A_cocos, cn.arctan(cn.tan(A_cocos)))
            assert np.allclose(A_cocos, cn.arcsinh(cn.sinh(A_cocos)))
            assert np.allclose(A_cocos, cn.arccosh(cn.cosh(A_cocos)))
            assert np.allclose(A_cocos, cn.arctanh(cn.tanh(A_cocos)))
