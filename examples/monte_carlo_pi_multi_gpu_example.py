from contexttimer import Timer
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy
import typing as tp

from cocos.map_reduce_multicore import map_reduce_multicore
import cocos.numerics as cn
from cocos.numerics.numerical_package_selector import select_num_pack
from cocos.device import sync
from cocos.device_pool import ComputeDevicePool

SINGLE_CORE_NUMPY = 'Single Core NumPy'


def estimate_pi(n: int, gpu: bool = True) -> float:
    np = select_num_pack(gpu)
    x = np.random.rand(n)
    y = np.random.rand(n)

    in_quarter_circle = (x * x + y * y) <= 1.0
    return 4.0 * float(np.mean(in_quarter_circle))


def single_core_benchmark(n: int, repetitions: int = 1) -> float:
    with Timer() as timer:
        for _ in range(repetitions):
            estimate_pi(n, gpu=False)

    return timer.elapsed / repetitions


def multi_core_benchmark(n: int, core_config: tp.Iterable[int], repetitions: int = 1) -> tp.Dict[int, float]:
    number_of_cores_to_runtime_map = {}

    for number_of_cores in core_config:
        with Timer() as timer:
            for _ in range(repetitions):
                pi = \
                    map_reduce_multicore(f=lambda: estimate_pi(n=math.ceil(n/number_of_cores), gpu=False),
                                         reduction=lambda x, y: x + y / number_of_cores,
                                         initial_value=0.0,
                                         number_of_batches=number_of_cores)

                sync()

        number_of_cores_to_runtime_map[number_of_cores] = timer.elapsed / repetitions

    return number_of_cores_to_runtime_map


def single_gpu_benchmark(n: int, repetitions: int = 1) -> float:
    with Timer() as timer:
        for _ in range(repetitions):
            estimate_pi(n, gpu=True)

    return timer.elapsed / repetitions


def multi_gpu_benchmark(n: int, batches: int, gpu_pool: ComputeDevicePool, repetitions: int = 1) -> tp.Dict[int, float]:
    number_of_devices_to_runtime_map = {}

    for number_of_devices_to_use in range(1, gpu_pool.number_of_devices + 1):
        with Timer() as timer:
            for _ in range(repetitions):
                pi = gpu_pool.map_reduce(lambda: estimate_pi(n=math.ceil(n / number_of_devices_to_use / batches), gpu=True),
                                         reduction=lambda x, y: x + y / number_of_devices_to_use / batches,
                                         initial_value=0.0,
                                         number_of_batches=number_of_devices_to_use * batches)

                sync()

        gpu_time = timer.elapsed / repetitions
        number_of_devices_to_runtime_map[number_of_devices_to_use] = gpu_time

    return number_of_devices_to_runtime_map


def create_result_table(means_of_computation_to_runtime_map: tp.Dict[str, float]) -> str:
    single_core_runtime = means_of_computation_to_runtime_map[SINGLE_CORE_NUMPY]

    res = "<table>\n"
    res += "<tbody>\n"
    res += "<tr>\n"
    res += "<th></th>\n"
    res += "<th>Total Time in Seconds</th>\n"
    res += "<th>Speedup Compared to NumPy</th>\n"
    res += "</tr>\n"

    for means_of_computation, runtime in means_of_computation_to_runtime_map.items():
        res += "<tr>\n"
        res += f"<td>{means_of_computation}</td>\n"
        res += f"<td>{runtime}</td>\n"
        res += f"<td>{single_core_runtime / runtime}</td>\n"
        res += "</tr>\n"

    res += "</table>"

    return res


def create_bar_plot(means_of_computation_to_runtime_map: tp.Dict[str, float]):
    single_core_runtime = means_of_computation_to_runtime_map[SINGLE_CORE_NUMPY]

    objects = list(means_of_computation_to_runtime_map.keys())

    y_pos = numpy.arange(len(objects))

    performance = [single_core_runtime / runtime
                   for runtime
                   in means_of_computation_to_runtime_map.values()]

    plt.figure(1)
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation=45, ha="right")
    plt.ylabel('Speedup Factor')
    plt.title('Performance Relative to NumPy \n'
              'in Monte Carlo Approximation of Pi \n')

    plt.savefig('monte_carlo_pi_benchmark_results')

    plt.show()


def main():
    n = 100000000
    repetitions = 1
    batches = 1

    # single core benchmark
    single_core_runtime = single_core_benchmark(n, repetitions=repetitions)
    means_of_computation_to_runtime_map = {SINGLE_CORE_NUMPY: single_core_runtime}
    print(f'Estimation of pi using single core NumPy performed in {single_core_runtime} seconds')

    # multi-core benchmark
    multi_core_benchmark(n=100, core_config=range(1, multiprocessing.cpu_count() + 1), repetitions=repetitions)
    number_of_cores_to_runtime_map = multi_core_benchmark(n=n, core_config=range(1, multiprocessing.cpu_count() + 1))

    for number_of_cores_to_use, cpu_time in number_of_cores_to_runtime_map.items():
        means_of_computation_to_runtime_map[f'NumPy with {number_of_cores_to_use} CPU core(s)'] = cpu_time
        print(f'Estimation of pi on {number_of_cores_to_use} core(s) using NumPy performed in {cpu_time} seconds')

    # single gpu
    single_gpu_benchmark(n=100)
    single_gpu_runtime = batches * single_gpu_benchmark(n=math.ceil(n/batches), repetitions=repetitions)
    means_of_computation_to_runtime_map['Cocos Single GPU'] = single_gpu_runtime
    print(f'Estimation of pi using single GPU Cocos performed in {single_core_runtime} seconds')

    # multi-gpu benchmark
    gpu_pool = ComputeDevicePool()

    multi_gpu_benchmark(n=100, batches=batches, gpu_pool=gpu_pool, repetitions=repetitions)
    number_of_devices_to_runtime_map = multi_gpu_benchmark(n=n, batches=batches, gpu_pool=gpu_pool)

    # display results
    for number_of_devices_to_use, gpu_time in number_of_devices_to_runtime_map.items():
        means_of_computation_to_runtime_map[f'Cocos with {number_of_devices_to_use} GPU(s)'] = gpu_time
        print(f'Estimation of pi on {number_of_devices_to_use} GPUs in {gpu_time} seconds')

    if gpu_pool.number_of_devices > 1:
        for number_of_devices_to_use in range(2, gpu_pool.number_of_devices + 1):
            print(f'Performance on {number_of_devices_to_use} GPUs increased by a factor of'
                  f' {number_of_devices_to_runtime_map[1] / number_of_devices_to_runtime_map[number_of_devices_to_use]} '
                  f'over a single GPU.')

    print(create_result_table(means_of_computation_to_runtime_map))
    create_bar_plot(means_of_computation_to_runtime_map)


if __name__ == '__main__':
    main()
