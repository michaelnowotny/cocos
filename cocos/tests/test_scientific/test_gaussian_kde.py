import numpy as np
import pytest
import scipy.stats as ss

import cocos.numerics as cn
from cocos.scientific.kde import gaussian_kde

n = 1000


@pytest.fixture()
def points() -> np.ndarray:
    return np.random.randn(n)


@pytest.fixture()
def xi() -> np.ndarray:
    return np.linspace(-5, 5, 101)


def test_gaussian_kde_scipy_vs_cocos(points: np.ndarray,
                                     xi: np.ndarray):
    gkde_cocos = gaussian_kde(points)
    gkde_scipy = ss.kde.gaussian_kde(points)

    density_estimate_cocos = gkde_cocos.evaluate(xi)
    density_estimate_scipy = gkde_scipy.evaluate(xi)

    assert np.allclose(density_estimate_cocos,
                       density_estimate_scipy)


def test_gaussian_kde_scipy_vs_cocos_gpu(points: np.ndarray,
                                         xi: np.ndarray):
    gkde_cocos = gaussian_kde(cn.array(points.squeeze()),
                              gpu=True)
    gkde_scipy = ss.kde.gaussian_kde(points)

    density_estimate_cocos = gkde_cocos.evaluate(xi)
    density_estimate_scipy = gkde_scipy.evaluate(xi)

    assert np.allclose(density_estimate_cocos,
                       density_estimate_scipy)
