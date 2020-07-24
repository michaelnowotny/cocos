import math
import matplotlib.pyplot as plt
import numpy
import time
import typing as tp

from cocos.device import (
    ComputeDeviceManager,
    info,
    sync
)

from cocos.multi_processing.device_pool import ComputeDevicePool
from cocos.multi_processing.multi_core_batch_processing import map_reduce_multicore

from cocos.numerics.numerical_package_bundle import (
    NumericalPackageBundle,
    CocosBundle
)

from cocos.numerics.random import randn_antithetic

from stochastic_volatility.heston_utilities import (
    simulate_and_compute_option_price_gpu,
    simulate_and_compute_option_price_multicore,
    simulate_and_compute_option_price,
    simulate_heston_model,
    compute_option_price_from_simulated_paths
)


def create_result_table(number_of_devices_to_runtime_map: tp.Dict[int, float]) \
        -> str:
    res = "<table>\n"
    res += "<tbody>\n"
    res += "<tr>\n"
    res += "<th>Number of GPUs</th>\n"
    res += "<th>Total Time in Seconds</th>\n"
    res += "<th>Speedup Compared to Single GPU</th>\n"
    res += "</tr>\n"

    for number_of_devices, runtime \
            in number_of_devices_to_runtime_map.items():
        res += "<tr>\n"
        res += f"<td>{number_of_devices}</td>\n"
        res += f"<td>{runtime}</td>\n"
        res += f"<td>{number_of_devices_to_runtime_map[1] / runtime}</td>\n"
        res += "</tr>\n"

    res += "</table>"

    return res


def create_bar_plot(number_of_devices_to_runtime_map: tp.Dict[int, float]):
    objects = []
    performance = []

    for number_of_devices, runtime \
            in number_of_devices_to_runtime_map.items():
        objects.append(number_of_devices)
        performance.append(number_of_devices_to_runtime_map[1] / runtime)

    y_pos = numpy.arange(len(objects))

    plt.figure(1)
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Speedup Factor')
    plt.title('Performance Relative to a Single GPU \n'
              'in Monte Carlo Simulation of Heston Model \n')

    plt.savefig(f'heston_pricing_benchmark_results_multi_gpu')

    plt.show()


if __name__ == '__main__':
    info()

    compute_device_pool = ComputeDevicePool()

    # model parameters
    x0 = 0.0  # initial log stock price
    v0 = 0.101 ** 2  # initial volatility
    r = math.log(1.0319)  # risk-free rate
    rho = -0.7  # instantaneous correlation between Brownian motions
    sigma_v = 0.61  # variance of volatility
    kappa = 6.21  # mean reversion speed
    v_bar = 0.019  # mean variance

    # option parameters
    T = 1.0  # time to expiration
    K = 0.95  # strike price

    # simulation parameters
    nT = int(math.ceil(500 * T))  # number of time-steps to simulate

    # warm-up
    R = 20000  # actual number of paths to simulate for pricing

    kwargs = \
        dict(x0=x0,
             v0=v0,
             r=r,
             rho=rho,
             sigma_v=sigma_v,
             kappa=kappa,
             v_bar=v_bar,
             T=T,
             K=K,
             nT=nT,
             R=R)

    print('warm-up')
    # tic = time.time()
    option_price = simulate_and_compute_option_price_gpu(compute_device_pool=compute_device_pool,
                                                         **kwargs)
    # toc = time.time() - tic
    # print(f'option price = {option_price} computed in {toc} seconds')
    print('warmed up')

    # actual run
    R = 2000000  # actual number of paths to simulate for pricing

    kwargs = \
        dict(x0=x0,
             v0=v0,
             r=r,
             rho=rho,
             sigma_v=sigma_v,
             kappa=kappa,
             v_bar=v_bar,
             T=T,
             K=K,
             nT=nT,
             R=R)

    number_of_devices_to_runtime_map = {}

    for i in range(1, compute_device_pool.number_of_devices + 1):
        print(f'computing on {i} GPUs')
        tic = time.time()
        option_price = \
            simulate_and_compute_option_price_gpu(compute_device_pool=compute_device_pool,
                                                  number_of_batches=i,
                                                  **kwargs)
        sync()
        gpu_time = time.time() - tic
        print(f'option price = {option_price} computed on {i} GPUs in '
              f'{gpu_time} seconds')

        number_of_devices_to_runtime_map[i] = gpu_time

    if compute_device_pool.number_of_devices > 1:
        for i in range(2, compute_device_pool.number_of_devices + 1):
            print(f'Performance on {i} GPUs increased by a factor of'
                  f' {number_of_devices_to_runtime_map[1] / number_of_devices_to_runtime_map[i]} '
                  f'over a single GPU.')

    result_table = create_result_table(number_of_devices_to_runtime_map)
    print(result_table)

    create_bar_plot(number_of_devices_to_runtime_map)
