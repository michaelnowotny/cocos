"""
Reductions of large single precision arrays must be about as accurate as
NumPy's on every backend. ArrayFire's CPU backend accumulates in single
precision, which made the mean of a million float32 values lose three to four
significant digits.
"""
import cocos.numerics as cn
import numpy as np
import pytest


@pytest.fixture
def x() -> np.ndarray:
    rng = np.random.default_rng(1)
    return (10.0 + 3.0 * rng.standard_normal(1_000_000)).astype(np.float32)


@pytest.mark.parametrize("reduction, reference",
                         [(cn.sum, np.sum),
                          (cn.mean, np.mean),
                          (cn.var, np.var),
                          (cn.std, np.std)])
def test_single_precision_reductions_are_accurate(x, reduction, reference):
    expected = reference(x.astype(np.float64))
    assert reduction(cn.array(x)) == pytest.approx(expected, rel=2e-6)


def test_single_precision_reductions_along_an_axis(x):
    a = np.column_stack([x, 2.0 * x])
    expected = a.astype(np.float64).mean(axis=0)
    result = cn.mean(cn.array(a), axis=0)
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.asarray(result).ravel(), expected, rtol=2e-6)


def test_weighted_average_is_accurate(x):
    weights = np.random.default_rng(2).random(x.size).astype(np.float32)
    expected = np.average(x.astype(np.float64), weights=weights.astype(np.float64))
    assert cn.average(cn.array(x), weights=cn.array(weights)) \
        == pytest.approx(expected, rel=2e-6)


def test_complex_variance_is_real_and_accurate(x):
    z = (x + 1j * x[::-1]).astype(np.complex64)
    expected = np.var(z.astype(np.complex128))
    assert cn.var(cn.array(z)) == pytest.approx(expected, rel=2e-6)
    variance_by_column = cn.var(cn.array(np.column_stack([z, z])), axis=0)
    assert variance_by_column.dtype == np.float32
