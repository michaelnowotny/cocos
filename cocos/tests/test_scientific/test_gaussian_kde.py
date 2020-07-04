import numpy as np
import pytest
import scipy.stats as ss

import cocos.numerics as cn
from cocos.scientific.kde import gaussian_kde

n = 1000
grid_size = 101


points_1d = np.random.randn(n)
xi_1d = np.linspace(-5, 5, grid_size)


def generate_points_2d():
    points_x = np.random.randn(1, n)
    points_y = np.random.randn(1, n)
    return np.vstack((points_x, points_y))


def generate_xi_2d():
    x_grid = np.linspace(-5.0, 5.0, grid_size)
    y_grid = np.linspace(-5.0, 5.0, grid_size)
    xy_grid_x, xy_grid_y = np.meshgrid(x_grid, y_grid)
    return np.hstack((xy_grid_x.reshape((-1, 1)), xy_grid_y.reshape((-1, 1)))).T


points_2d = generate_points_2d()
xi_2d = generate_xi_2d()


@pytest.mark.parametrize('points, xi', [(points_1d, xi_1d),
                                        (points_2d, xi_2d)])
def test_gaussian_kde_scipy_vs_cocos(points: np.ndarray,
                                     xi: np.ndarray):
    gkde_cocos = gaussian_kde(points)
    gkde_scipy = ss.kde.gaussian_kde(points)

    density_estimate_cocos = gkde_cocos.evaluate(xi)
    density_estimate_scipy = gkde_scipy.evaluate(xi)

    assert np.allclose(density_estimate_cocos,
                       density_estimate_scipy)


@pytest.mark.parametrize('points, xi', [(points_1d, xi_1d),
                                        (points_2d, xi_2d)])
def test_gaussian_kde_scipy_vs_cocos_gpu(points: np.ndarray,
                                         xi: np.ndarray):
    gkde_cocos = gaussian_kde(cn.array(points.squeeze()),
                              gpu=True)
    gkde_scipy = ss.kde.gaussian_kde(points)

    density_estimate_cocos = gkde_cocos.evaluate(xi)
    density_estimate_scipy = gkde_scipy.evaluate(xi)

    assert np.allclose(density_estimate_cocos,
                       density_estimate_scipy)
