from contexttimer import Timer
import math
import numpy
import typing as tp

import cocos.numerics as cn
from cocos.numerics.numerical_package_selector import select_num_pack
from cocos.device import sync
from cocos.device_pool import ComputeDevicePool


def estimate_pi(n: int, gpu: bool = True) -> float:
    np = select_num_pack(gpu)
    x = np.random.rand(n)
    y = np.random.rand(n)

    in_quarter_circle = (x * x + y * y) <= 1.0
    return 4.0 * float(np.mean(in_quarter_circle))


def multi_gpu_benchmark(n: int, gpu_pool: ComputeDevicePool) -> tp.Dict[int, float]:
    number_of_devices_to_runtime_map = {}

    for number_of_devices_to_use in range(1, gpu_pool.number_of_devices + 1):
        with Timer() as timer:
            pi = gpu_pool.map_reduce(lambda: estimate_pi(n=math.ceil(n/number_of_devices_to_use),
                                                         gpu=True),
                                     reduction=lambda x, y: x + y / number_of_devices_to_use,
                                     initial_value=0.0,
                                     number_of_batches=number_of_devices_to_use)

            sync()

        gpu_time = timer.elapsed
        number_of_devices_to_runtime_map[number_of_devices_to_use] = gpu_time

    return number_of_devices_to_runtime_map


def main():
    n = 200000000
    gpu_pool = ComputeDevicePool()

    multi_gpu_benchmark(n=100, gpu_pool=gpu_pool)
    number_of_devices_to_runtime_map = multi_gpu_benchmark(n=n, gpu_pool=gpu_pool)

    for number_of_devices_to_use, gpu_time in number_of_devices_to_runtime_map.items():
        print(f'Estimation of pi on {number_of_devices_to_use} GPUs in {gpu_time} seconds')

    if gpu_pool.number_of_devices > 1:
        for number_of_devices_to_use in range(2, gpu_pool.number_of_devices + 1):
            gpu_speedup = number_of_devices_to_runtime_map[1] / \
                          number_of_devices_to_runtime_map[number_of_devices_to_use]
            print(f'Performance on {number_of_devices_to_use} GPUs increased by a factor of'
                  f' {gpu_speedup} over a single GPU.')


if __name__ == '__main__':
    main()
