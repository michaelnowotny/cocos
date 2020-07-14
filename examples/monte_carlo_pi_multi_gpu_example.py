from contexttimer import Timer
import math
import matplotlib.pyplot as plt
import numpy
import time
import typing as tp

from cocos.numerics.numerical_package_bundle import (
    NumericalPackageBundle,
    CocosBundle
)

from cocos.numerics.random import randn_antithetic
from cocos.device import (
    ComputeDeviceManager,
    info,
    sync
)

from cocos.device_pool import ComputeDevicePool


def estimate_pi(n: int, gpu: bool = True) -> float:
    if gpu:
        import cocos.numerics as np
    else:
        import numpy as np

    x = np.random.rand(n)
    y = np.random.rand(n)

    in_quarter_circle = (x * x + y * y) <= 1.0
    return 4.0 * float(np.mean(in_quarter_circle))


if __name__ == '__main__':
    n = 100000000
    gpu_pool = ComputeDevicePool()
    number_of_devices_to_runtime_map = {}

    for number_of_devices_to_use in range(1, gpu_pool.number_of_devices + 1):
        print(f'computing on {number_of_devices_to_use} GPUs')
        number_of_batches = number_of_devices_to_use
        with Timer() as timer:
            pi = gpu_pool.map_reduce(lambda x: estimate_pi(n=n, gpu=True),
                                     reduction=lambda x, y: x + y/ number_of_devices_to_use,
                                     initial_value=0.0,
                                     number_of_batches=number_of_devices_to_use)
            sync()

        gpu_time = timer.elapsed
        print(f'Estimation of pi = {pi} on {number_of_devices_to_use} GPUs in '
              f'{gpu_time} seconds')

        number_of_devices_to_runtime_map[number_of_devices_to_use] = gpu_time

    if gpu_pool.number_of_devices > 1:
        for number_of_devices_to_use in range(2, gpu_pool.number_of_devices + 1):
            print(f'Performance on {number_of_devices_to_use} GPUs increased by a factor of'
                  f' {number_of_devices_to_runtime_map[1] / number_of_devices_to_runtime_map[number_of_devices_to_use]} '
                  f'over a single GPU.')

