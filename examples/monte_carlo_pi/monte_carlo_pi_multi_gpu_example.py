from contexttimer import Timer
import math
import matplotlib.pyplot as plt
import multiprocessing
import numexpr as ne
import numpy
import typing as tp

from cocos.multi_processing.multi_core_batch_processing import map_reduce_multicore
from cocos.numerics.numerical_package_selector import select_num_pack
from cocos.device import sync
from cocos.numerics.random import rand_with_dtype
from cocos.multi_processing.device_pool import ComputeDevicePool

SINGLE_CORE_NUMPY = 'NumPy Single Core'


def estimate_pi(n: int, batches: int = 1, gpu: bool = True) -> float:
    np = select_num_pack(gpu)

    n_per_batch = math.ceil(n/batches)

    pi = 0.0
    for _ in range(batches):
        x = rand_with_dtype([n_per_batch], dtype=numpy.float32, num_pack=np)
        y = rand_with_dtype([n_per_batch], dtype=numpy.float32, num_pack=np)

        in_quarter_circle = (x * x + y * y) <= 1.0
        del x, y
        pi += 4.0 * float(np.mean(in_quarter_circle))
        del in_quarter_circle

    return pi / batches


def estimate_pi_numexpr(n: int) -> float:
    np = numpy

    x = rand_with_dtype([n], dtype=numpy.float32, num_pack=np)
    y = rand_with_dtype([n], dtype=numpy.float32, num_pack=np)

    in_quarter_circle = ne.evaluate('x * x + y * y <= 1.0')
    pi = 4.0 * float(np.mean(in_quarter_circle))
    # pi = 4.0 * float(ne.evaluate('sum(x * x + y * y <= 1.0)')) / n

    return pi


def estimate_pi_cupy(n: int, batches: int = 1) -> float:
    import cupy as np

    n_per_batch = math.ceil(n/batches)

    pi = 0.0
    for _ in range(batches):
        x = np.random.rand(n_per_batch, dtype=numpy.float32)
        y = np.random.rand(n_per_batch, dtype=numpy.float32)

        in_quarter_circle = (x * x + y * y) <= 1.0
        del x, y
        pi += 4.0 * float(np.mean(in_quarter_circle))
        del in_quarter_circle

    return pi / batches


def single_core_benchmark(n: int,
                          repetitions: int = 1,
                          verbose: bool = False) -> float:
    if verbose:
        print('single core benchmark - begin')

    with Timer() as timer:
        for _ in range(repetitions):
            pi = estimate_pi(n, gpu=False)
            if verbose:
                print(pi)

    if verbose:
        print('single core benchmark - end')

    return timer.elapsed / repetitions


def single_core_benchmark_numexpr(n: int,
                                  repetitions: int = 1,
                                  verbose: bool = False) -> float:
    if verbose:
        print('single core benchmark numexpr - begin')

    with Timer() as timer:
        for _ in range(repetitions):
            pi = estimate_pi_numexpr(n)
            if verbose:
                print(pi)

    if verbose:
        print('single core benchmark numexpr - end')

    return timer.elapsed / repetitions


def multi_core_benchmark(n: int,
                         core_config: tp.Iterable[int],
                         repetitions: int = 1,
                         verbose: bool = False) -> tp.Dict[int, float]:
    if verbose:
        print('multi core benchmark - begin')

    number_of_cores_to_runtime_map = {}

    for number_of_cores in core_config:
        with Timer() as timer:
            for _ in range(repetitions):
                pi = \
                    map_reduce_multicore(f=lambda: estimate_pi(n=math.ceil(n/number_of_cores), gpu=False),
                                         reduction=lambda x, y: x + y / number_of_cores,
                                         initial_value=0.0,
                                         number_of_batches=number_of_cores)
                if verbose:
                    print(pi)

        number_of_cores_to_runtime_map[number_of_cores] = timer.elapsed / repetitions

    if verbose:
        print('multi core benchmark - end')

    return number_of_cores_to_runtime_map


def single_gpu_cupy_benchmark(n: int,
                              batches: int,
                              repetitions: int = 1,
                              verbose: bool = False) -> float:
    if verbose:
        print('single gpu cupy benchmark - begin')

    # import cupy
    with Timer() as timer:
        for _ in range(repetitions):
            pi = estimate_pi_cupy(n, batches=batches)
            # cupy.cuda.Stream.null.synchronize()
            if verbose:
                print(pi)

    if verbose:
        print('single gpu cupy benchmark - end')

    return timer.elapsed / repetitions


def single_gpu_benchmark(n: int,
                         batches: int,
                         repetitions: int = 1,
                         verbose: bool = False) -> float:
    if verbose:
        print('single gpu benchmark - begin')

    with Timer() as timer:
        for _ in range(repetitions):
            pi = estimate_pi(n, batches=batches, gpu=True)
            sync()
            if verbose:
                print(pi)

    if verbose:
        print('single gpu benchmark - end')

    return timer.elapsed / repetitions


def multi_gpu_benchmark(n: int,
                        batches: int,
                        gpu_pool: ComputeDevicePool,
                        repetitions: int = 1,
                        verbose: bool = False) -> tp.Dict[int, float]:
    if verbose:
        print('multi gpu benchmark - begin')

    number_of_devices_to_runtime_map = {}

    for number_of_devices_to_use in range(1, gpu_pool.number_of_devices + 1):
        with Timer() as timer:
            for _ in range(repetitions):
                pi = gpu_pool.map_reduce(lambda: estimate_pi(n=math.ceil(n / number_of_devices_to_use), batches=batches, gpu=True),
                                         reduction=lambda x, y: x + y / number_of_devices_to_use,
                                         initial_value=0.0,
                                         number_of_batches=number_of_devices_to_use)

                sync()
                if verbose:
                    print(pi)

        gpu_time = timer.elapsed / repetitions
        number_of_devices_to_runtime_map[number_of_devices_to_use] = gpu_time

    if verbose:
        print('multi gpu benchmark - end')

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

    plt.savefig('monte_carlo_pi_benchmark_results', bbox_inches='tight')

    plt.show()


def main():
    n = 1000000000
    repetitions = 1
    batches = 20
    means_of_computation_to_runtime_map = {}
    verbose = True

    # single core benchmark
    single_core_benchmark(n=100, repetitions=repetitions)
    single_core_runtime = single_core_benchmark(n, repetitions=repetitions, verbose=verbose)
    means_of_computation_to_runtime_map[SINGLE_CORE_NUMPY] = single_core_runtime
    print(f'Estimation of pi using single core NumPy performed in {single_core_runtime} seconds')

    # single core benchmark numexpr
    single_core_benchmark_numexpr(n=100, repetitions=repetitions)
    single_core_runtime_numexpr = single_core_benchmark_numexpr(n, repetitions=repetitions, verbose=verbose)
    means_of_computation_to_runtime_map['NumExpr Single Core'] = single_core_runtime_numexpr
    print(f'Estimation of pi using single core Numexpr performed in {single_core_runtime_numexpr} seconds')

    # multi core benchmark
    multi_core_benchmark(n=100, core_config=range(1, multiprocessing.cpu_count() + 1), repetitions=repetitions)
    number_of_cores_to_runtime_map = \
        multi_core_benchmark(n=n, core_config=range(1, multiprocessing.cpu_count() + 1), verbose=verbose)

    for number_of_cores_to_use, cpu_time in number_of_cores_to_runtime_map.items():
        means_of_computation_to_runtime_map[f'NumPy with {number_of_cores_to_use} CPU core(s)'] = cpu_time
        print(f'Estimation of pi on {number_of_cores_to_use} core(s) using NumPy performed in {cpu_time} seconds')

    # single gpu
    single_gpu_benchmark(n=100, batches=1)
    single_gpu_runtime = single_gpu_benchmark(n=n, batches=batches, repetitions=repetitions, verbose=verbose)
    means_of_computation_to_runtime_map['Cocos Single GPU'] = single_gpu_runtime
    print(f'Estimation of pi using single GPU Cocos performed in {single_gpu_runtime} seconds')

    # multi gpu benchmark
    gpu_pool = ComputeDevicePool()

    multi_gpu_benchmark(n=100, batches=batches, gpu_pool=gpu_pool, repetitions=repetitions)
    number_of_devices_to_runtime_map = multi_gpu_benchmark(n=n, batches=batches, gpu_pool=gpu_pool, verbose=verbose)

    for number_of_devices_to_use, gpu_time in number_of_devices_to_runtime_map.items():
        means_of_computation_to_runtime_map[f'Cocos with {number_of_devices_to_use} GPU(s)'] = gpu_time
        print(f'Estimation of pi on {number_of_devices_to_use} GPUs in {gpu_time} seconds')

    if gpu_pool.number_of_devices > 1:
        for number_of_devices_to_use in range(2, gpu_pool.number_of_devices + 1):
            print(f'Performance on {number_of_devices_to_use} GPUs increased by a factor of'
                  f' {number_of_devices_to_runtime_map[1] / number_of_devices_to_runtime_map[number_of_devices_to_use]} '
                  f'over a single GPU.')

    # cupy single gpu
    try:
        single_gpu_cupy_benchmark(n=100, batches=1)
        single_gpu_cupy_runtime = \
            single_gpu_cupy_benchmark(n=n, batches=batches, repetitions=repetitions, verbose=verbose)
        means_of_computation_to_runtime_map['CuPy Single GPU'] = single_gpu_cupy_runtime
        print(f'Estimation of pi using single GPU CuPy performed in {single_gpu_cupy_runtime} seconds')
    except Exception as e:
        print(e)
        print('CuPy is not installed or not working correctly.')

    print(create_result_table(means_of_computation_to_runtime_map))
    create_bar_plot(means_of_computation_to_runtime_map)


if __name__ == '__main__':
    main()
