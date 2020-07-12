from contexttimer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

import cocos.numerics as cn
import cocos.device as cd
from cocos.scientific.kde import gaussian_kde

n = 10000  # number of data points
grid_size = n+1  # number of points at which to evaluate the kde
R = 10  # number of repetitions for performance benchmark


if __name__ == '__main__':
    # generate random sample
    points = np.random.randn(n)

    # generate grid at which to evaluate the sample
    grid = np.linspace(-5.0, 5.0, grid_size)

    # construct and evaluate scipy gaussian kde object
    gaussian_kde_scipy = ss.kde.gaussian_kde(points)
    density_estimate_scipy = gaussian_kde_scipy.evaluate(grid)

    # construct and evaluate cocos gaussian kde object using gpu evaluation
    gaussian_kde_cocos = gaussian_kde(cn.array(points), gpu=True)
    density_estimate_cocos = np.array(gaussian_kde_cocos.evaluate(grid))

    # verify that results are numerically close
    print(f'maximum absolute difference between results gpu using Cocos and cpu using SciPy: '
          f'{np.max(abs(density_estimate_cocos - density_estimate_scipy))}')

    if np.allclose(density_estimate_cocos, density_estimate_scipy):
        print('estimates from cocos and scipy are numerically close')
    else:
        print('estimates from cocos and scipy deviate by more than the default tolerance')

    # plot kernel density estimates
    plt.plot(grid, density_estimate_cocos, label='gaussian kernel density estimated using Cocos')
    plt.plot(grid, density_estimate_scipy, label='gaussian kernel density estimated using SciPy')
    plt.legend(loc=1)
    plt.show()

    # run benchmark comparing cpu performance using SciPy with gpu performance using Cocos
    with Timer() as scipy_timer:
        for _ in range(R):
            gaussian_kde_scipy.evaluate(grid)

    print(f'Time to evaluate gaussian kde on cpu using scipy was {scipy_timer.elapsed / R} seconds')

    with Timer() as cocos_timer:
        for _ in range(R):
            gaussian_kde_cocos.evaluate(grid)
            cd.sync()

    print(f'Time to evaluate gaussian kde on gpu using cocos was {cocos_timer.elapsed / R} seconds')

    print(f'speedup on gpu is {scipy_timer.elapsed/cocos_timer.elapsed}')
