import numpy as np
import pytest

from cocos.scientific.kde import (
    gaussian_kernel_estimate,
    gaussian_kernel_estimate_vectorized
)

n = 1000


@pytest.fixture()
def points() -> np.ndarray:
    return np.random.randn(n, 1)


@pytest.fixture()
def weights(points: np.ndarray) -> np.ndarray:
    return np.full_like(points, fill_value=1.0 / n)


@pytest.fixture()
def xi() -> np.ndarray:
    return np.linspace(-5, 5, 101).reshape((-1, 1))


@pytest.fixture()
def precision(points: np.ndarray) -> np.ndarray:
    return np.atleast_2d(1.0 / np.var(points))


def test_gaussian_kernel_estimate_loop_vs_vectorized_cpu(points: np.ndarray,
                                                         weights: np.ndarray,
                                                         xi: np.ndarray,
                                                         precision: np.ndarray):
    kde = gaussian_kernel_estimate(points, weights, xi, precision, np.float64)
    kde_vectorized = np.array(gaussian_kernel_estimate_vectorized(points=points,
                                                                  values=weights,
                                                                  xi=xi,
                                                                  precision=precision,
                                                                  dtype=np.float64,
                                                                  gpu=False))

    assert np.allclose(kde_vectorized, kde[:, 0])  # the loop returns shape (m, 1)


def test_gaussian_kernel_estimate_loop_vs_vectorized_gpu(points: np.ndarray,
                                                         weights: np.ndarray,
                                                         xi: np.ndarray,
                                                         precision: np.ndarray):
    kde = gaussian_kernel_estimate(points, weights, xi, precision, np.float64)
    kde_vectorized = np.array(gaussian_kernel_estimate_vectorized(points=points,
                                                                  values=weights,
                                                                  xi=xi,
                                                                  precision=precision,
                                                                  dtype=np.float64,
                                                                  gpu=True))

    assert np.allclose(kde_vectorized, kde[:, 0])  # the loop returns shape (m, 1)


@pytest.mark.parametrize("gpu", [False, True])
def test_gaussian_kernel_estimate_vectorized_with_one_dimensional_arrays(
        points: np.ndarray,
        weights: np.ndarray,
        xi: np.ndarray,
        precision: np.ndarray,
        gpu: bool):
    kde = gaussian_kernel_estimate(points, weights, xi, precision, np.float64)
    kde_vectorized = np.array(gaussian_kernel_estimate_vectorized(points=points[:, 0],
                                                                  values=weights,
                                                                  xi=xi[:, 0],
                                                                  precision=precision,
                                                                  dtype=np.float64,
                                                                  gpu=gpu))

    assert np.allclose(kde_vectorized, kde[:, 0])
