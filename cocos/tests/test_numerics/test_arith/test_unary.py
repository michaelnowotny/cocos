import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest
import scipy.special as sp


test_data = [np.array([[np.inf, -np.inf, np.NaN]],
                      dtype=np.float32),
             np.array([[0, 0]],
                      dtype=np.int32),
             np.array([[1, 2, -3],
                       [4, -5, 6],
                       [7, 8, 20]],
                      dtype=np.int32),
             np.array([[1, 2, 3],
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
def test_unary(A_numpy):
    cocos.device.init()
    A_cocos = cn.array(A_numpy)

    all_positive = np.all(A_numpy > 0)

    all_finite = cn.all(cn.isfinite(A_cocos))
    assert np.allclose(np.isfinite(A_numpy), cn.isfinite(A_cocos))

    if all_finite:
        if cn.isrealobj(A_cocos):
            if np.count_nonzero(A_numpy) == \
                    A_numpy.size and cn.isfloating(A_cocos):
                assert np.allclose(1 / A_numpy, 1 / A_cocos)
            if all_positive:
                assert np.allclose(np.sqrt(A_numpy), cn.sqrt(A_cocos))
                assert np.allclose(np.cbrt(A_numpy), cn.cbrt(A_cocos))

                # af.root does not seem to work
                # if cn.isfloating(A_cocos):
                #     print("cbrt numpy")
                #     print(np.cbrt(A_numpy))
                #     print("third root cocos")
                #     print(cn.root(A_cocos, 3))
                #     print("cbrt cocos")
                #     print(cn.cbrt(A_cocos))
                #
                #     assert np.allclose(np.cbrt(A_numpy), cn.root(A_cocos, 3))
                assert np.allclose(np.log(A_numpy), cn.log(A_cocos))
                assert np.allclose(np.log1p(A_numpy), cn.log1p(A_cocos))
                assert np.allclose(np.log2(A_numpy), cn.log2(A_cocos))
                assert np.allclose(np.log10(A_numpy), cn.log10(A_cocos))
                assert np.allclose(sp.gamma(A_numpy), cn.gamma(A_cocos))
                assert np.allclose(sp.gammaln(A_numpy), cn.gammaln(A_cocos))
                assert np.allclose(np.power(2, A_numpy), cn.power2(A_cocos))

                if cn.isinteger(A_cocos):
                    assert np.allclose(sp.factorial(A_numpy),
                                       cn.factorial(A_cocos))

            # assert np.allclose(np.round(A), cn.round(A_cocos))
            assert np.allclose(np.isneginf(A_numpy), cn.isneginf(A_cocos))
            assert np.allclose(np.isposinf(A_numpy), cn.isposinf(A_cocos))
            assert np.allclose(np.trunc(A_numpy), cn.trunc(A_cocos))
            assert np.allclose(np.floor(A_numpy), cn.floor(A_cocos))
            assert np.allclose(np.ceil(A_numpy), cn.ceil(A_cocos))
            assert np.allclose(np.sign(A_numpy), cn.sign(A_cocos))
            assert np.allclose(np.sinh(A_numpy), cn.sinh(A_cocos))
            assert np.allclose(np.cosh(A_numpy), cn.cosh(A_cocos))
            assert np.allclose(np.tanh(A_numpy), cn.tanh(A_cocos))
            assert np.allclose(np.sin(A_numpy), cn.sin(A_cocos))
            assert np.allclose(np.cos(A_numpy), cn.cos(A_cocos))
            assert np.allclose(np.tan(A_numpy), cn.tan(A_cocos))
            assert np.allclose(sp.expit(A_numpy), cn.expit(A_cocos))
            assert np.allclose(np.expm1(A_numpy), cn.expm1(A_cocos))
            assert np.allclose(sp.erf(A_numpy), cn.erf(A_cocos))
            assert np.allclose(sp.erfc(A_numpy), cn.erfc(A_cocos))

            if cn.isinteger(A_cocos):
                # print("invert numpy")
                # print(np.invert(A))
                # print("invert cocos")
                # print(cn.invert(A_cocos))
                assert np.allclose(np.invert(A_numpy), cn.invert(A_cocos))
        else:
            assert np.allclose(np.real(A_numpy), cn.real(A_cocos))
            assert np.allclose(np.imag(A_numpy), cn.imag(A_cocos))
            assert np.allclose(np.conj(A_numpy), cn.conj(A_cocos))
            assert np.allclose(np.angle(A_numpy), cn.angle(A_cocos))

        if not all_positive:
            np.allclose(np.absolute(A_numpy), cn.absolute(A_cocos))

        assert np.allclose(np.exp(A_numpy), cn.exp(A_cocos))
        assert np.allclose(A_numpy == 0, cn.iszero(A_cocos))

    assert np.allclose(np.isfinite(A_numpy), cn.isfinite(A_cocos))
    assert np.allclose(np.isinf(A_numpy), cn.isinf(A_cocos))
    assert np.allclose(np.isnan(A_numpy), cn.isnan(A_cocos))
