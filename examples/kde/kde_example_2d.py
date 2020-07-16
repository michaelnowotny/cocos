import cocos.numerics as cn
from cocos.scientific.kde import (
    gaussian_kde,
    evaluate_gaussian_kde_in_batches
)

from contexttimer import Timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.stats as ss


n = 2000  # number of data points
grid_size = 200+1  # number of points at which to evaluate the kde
R = 10  # number of repetitions for performance benchmark
maximum_number_of_elements_per_batch = 10 * n * n


if __name__ == '__main__':
    # generate random sample
    points_x = np.random.randn(1, n)
    points_y = np.random.randn(1, n)
    points_xy = np.vstack((points_x, points_y))
    # print(f'points_xy.shape = {points_xy.shape}')

    # generate grid at which to evaluate the sample
    x_grid = np.linspace(-5.0, 5.0, grid_size)
    y_grid = np.linspace(-5.0, 5.0, grid_size)
    xy_grid_x, xy_grid_y  = np.meshgrid(x_grid, y_grid)
    xy_grid = np.hstack((xy_grid_x.reshape((-1, 1)), xy_grid_y.reshape((-1, 1)))).T
    # print(f'xy_grid.shape = {xy_grid.shape}')

    # construct and evaluate scipy gaussian kde object
    gaussian_kde_scipy = ss.kde.gaussian_kde(points_xy)
    density_estimate_scipy = gaussian_kde_scipy.evaluate(xy_grid)
    # print(f'density_estimate_scipy.shape = {density_estimate_scipy.shape}')

    # construct and evaluate cocos gaussian kde object using gpu evaluation
    gaussian_kde_cocos = gaussian_kde(cn.array(points_xy), gpu=True)
    density_estimate_cocos = np.array(gaussian_kde_cocos.evaluate(xy_grid))
    # print(f'density_estimate_cocos.shape = {density_estimate_cocos.shape}')

    # evaluate cocos gaussian kde object using gpu evaluation in batches
    batched_density_estimate_cocos = \
        evaluate_gaussian_kde_in_batches(gaussian_kde_cocos, xy_grid, 10 * n * n)
    # print(f'batched_density_estimate_cocos.shape = {batched_density_estimate_cocos.shape}')

    # verify that results are numerically close
    print(f'maximum absolute difference between results gpu using Cocos and cpu using SciPy: '
          f'{np.max(abs(density_estimate_cocos - density_estimate_scipy))}')

    if np.allclose(density_estimate_cocos, density_estimate_scipy):
        print('estimates from cocos and scipy are numerically close')
    else:
        print('estimates from cocos and scipy deviate by more than the default tolerance')

    # verify that results are numerically close
    print(f'maximum absolute difference between batched and single-pass Cocos results: '
          f'{np.max(abs(density_estimate_cocos - batched_density_estimate_cocos))}')

    if np.allclose(density_estimate_cocos, batched_density_estimate_cocos):
        print('estimates from via single pass and via batched processing in Cocos are '
              'numerically close')
    else:
        print('estimates from via single pass and via batched processing in Cocos deviate by more '
              'than the default tolerance')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(X=xy_grid_x,
                      Y=xy_grid_y,
                      Z=density_estimate_scipy.reshape((grid_size, grid_size)),
                      color='b',
                      # alpha=0.8,
                      label='SciPy')

    ax.plot_wireframe(X=xy_grid_x,
                      Y=xy_grid_y,
                      Z=np.array(density_estimate_cocos.reshape((grid_size, grid_size))),
                      color='r',
                      # alpha=0.8,
                      label='Cocos')
    plt.legend()
    plt.show()

    # run benchmark comparing cpu performance using SciPy with gpu performance using Cocos
    with Timer() as scipy_timer:
        for _ in range(R):
            gaussian_kde_scipy.evaluate(xy_grid)

    print(f'Time to evaluate gaussian kde on cpu using scipy was {scipy_timer.elapsed / R} seconds')

    with Timer() as cocos_timer:
        for _ in range(R):
            gaussian_kde_cocos.evaluate(xy_grid)

    print(f'Time to evaluate gaussian kde on gpu using cocos was {cocos_timer.elapsed / R} seconds')

    print(f'speedup on gpu is {scipy_timer.elapsed/cocos_timer.elapsed}')

    with Timer() as batched_cocos_timer:
        for _ in range(R):
            evaluate_gaussian_kde_in_batches(
                kde=gaussian_kde_cocos,
                points=xy_grid,
                maximum_number_of_elements_per_batch=maximum_number_of_elements_per_batch)

    print(f'Time to evaluate gaussian kde on gpu using cocos in batched was '
          f'{batched_cocos_timer.elapsed / R} seconds')

    print(f'batched evaluation is {batched_cocos_timer.elapsed/cocos_timer.elapsed} times slower '
          f'than evaluation in a single pass on the gpu')
