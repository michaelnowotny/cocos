from cocos.multi_processing.utilities import (
    generate_slices_with_batch_size,
    generate_slices_with_number_of_batches
)

from contexttimer import Timer
from line_profiler import LineProfiler
import multiprocessing
import numpy
import typing as tp

from cocos.device import sync
from cocos.multi_processing.single_gpu_batch_processing import (
    map_reduce_single_gpu,
    map_combine_single_gpu
)
from cocos.multi_processing.multi_core_batch_processing import (
    map_reduce_multicore,
    map_combine_multicore
)

import cocos.numerics as cn

from cocos.numerics.data_types import NumericArray
from cocos.numerics.numerical_package_selector import select_num_pack
from cocos.numerics.random import rand_with_dtype

SINGLE_CORE_NUMPY = 'Single Core NumPy'


def split_arrays(a: NumericArray,
                 b: NumericArray,
                 c: NumericArray,
                 number_of_batches: int) -> tp.Tuple[tp.Dict[str, NumericArray], ...]:
    assert a.shape == b.shape
    assert a.shape == b.shape
    assert a.ndim == 1

    n = len(a)

    slices = generate_slices_with_number_of_batches(n=n, number_of_batches=number_of_batches)

    kwargs_list = []
    for begin_index, end_index in slices:
        sub_a = a[begin_index:end_index]
        sub_b = b[begin_index:end_index]
        sub_c = c[begin_index:end_index]
        kwargs_list.append({'a': sub_a, 'b': sub_b, 'c': sub_c})

    return tuple(kwargs_list)


def generate_data(n: int, gpu: bool) -> tp.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    num_pack = select_num_pack(gpu)
    a = rand_with_dtype([n], dtype=numpy.float32, num_pack=num_pack)
    b = rand_with_dtype([n], dtype=numpy.float32, num_pack=num_pack)
    c = rand_with_dtype([n], dtype=numpy.float32, num_pack=num_pack)
    return a, b, c


def process_data(a: NumericArray,
                 b: NumericArray,
                 c: NumericArray,
                 gpu: bool):
    np = select_num_pack(gpu)

    return (np.sin(a) + np.cos(b)) * np.arctan(c)


def single_core_benchmark(n: int, repetitions: int = 1) -> float:
    a, b, c = generate_data(n, gpu=False)

    with Timer() as timer:
        for _ in range(repetitions):
            process_data(a, b, c, gpu=False)

    return timer.elapsed / repetitions


def multi_core_benchmark(n: int, core_config: tp.Iterable[int], repetitions: int = 1) \
        -> tp.Dict[int, float]:
    # This does not increase performance on Anaconda and Windows as elementwise trigonometric
    # operations seem to be multi-threaded to begin with.

    number_of_cores_to_runtime_map = {}
    a_complete, b_complete, c_complete = generate_data(n, gpu=False)

    for number_of_cores in core_config:
        with Timer() as timer:
            for _ in range(repetitions):
                kwargs_list = split_arrays(a=a_complete,
                                           b=b_complete,
                                           c=c_complete,
                                           number_of_batches=number_of_cores)

                result = \
                    map_combine_multicore(f=lambda a, b, c: process_data(a, b, c, gpu=False),
                                          combination=lambda x: numpy.hstack(x),
                                          kwargs_list=kwargs_list)

                # print(type(result))
                assert isinstance(result, numpy.ndarray)
                # print(result.shape)
                assert result.shape == (n,)

        number_of_cores_to_runtime_map[number_of_cores] = timer.elapsed / repetitions

    return number_of_cores_to_runtime_map


def single_gpu_benchmark(n: int, batches: int, repetitions: int = 1) -> float:
    a_complete, b_complete, c_complete = generate_data(n, gpu=True)

    with Timer() as timer:
        for _ in range(repetitions):
            process_data(a_complete, b_complete, c_complete, gpu=True)
            sync()

    return timer.elapsed / repetitions


def host_to_device_transfer_function(a, b, c):
    return [], {'a': cn.array(a), 'b': cn.array(b), 'c': cn.array(c)}


def device_to_host_transfer_function(x):
    if x is None:
        return None
    else:
        return numpy.array(x)


def batched_single_gpu_benchmark(n: int,
                                 batches: int,
                                 repetitions: int = 1,
                                 use_map_combine: bool = False) -> float:
    a_complete, b_complete, c_complete = generate_data(n, gpu=False)

    with Timer() as timer:
        for _ in range(repetitions):
            kwargs_list = split_arrays(a=a_complete,
                                       b=b_complete,
                                       c=c_complete,
                                       number_of_batches=batches)

            if use_map_combine:
                map_combine_single_gpu(f=lambda a, b, c: process_data(a, b, c, gpu=True),
                                       combination=lambda x: numpy.hstack(x),
                                       host_to_device_transfer_function=host_to_device_transfer_function,
                                       device_to_host_transfer_function=device_to_host_transfer_function,
                                       kwargs_list=kwargs_list)
            else:
                map_reduce_single_gpu(f=lambda a, b, c: process_data(a, b, c, gpu=True),
                                      reduction=lambda x, y: numpy.hstack((x, y)),
                                      host_to_device_transfer_function=host_to_device_transfer_function,
                                      device_to_host_transfer_function=device_to_host_transfer_function,
                                      initial_value=numpy.zeros((0,)),
                                      kwargs_list=kwargs_list)

            sync()

    return timer.elapsed / repetitions


def main():
    n = 200000000
    repetitions = 1
    batches = 20
    means_of_computation_to_runtime_map = {}

    # single gpu
    single_gpu_benchmark(n=100, batches=1)
    single_gpu_runtime = single_gpu_benchmark(n=n, batches=batches, repetitions=repetitions)
    means_of_computation_to_runtime_map['Cocos Single GPU'] = single_gpu_runtime
    print(f'Data processing using single GPU Cocos performed in {single_gpu_runtime} seconds')

    # batched single gpu using map-reduce
    batched_single_gpu_benchmark(n=100, batches=1)
    batched_single_gpu_runtime = batched_single_gpu_benchmark(n=n,
                                                              batches=batches,
                                                              repetitions=repetitions)
    means_of_computation_to_runtime_map['Batched Cocos Single GPU Map Reduce'] = batched_single_gpu_runtime
    print(f'Data processing using batched single GPU using Cocos map-reduce performed in '
          f'{batched_single_gpu_runtime} seconds')

    # batched single gpu using map-combine
    batched_single_gpu_benchmark(n=100, batches=1, use_map_combine=True)
    batched_single_gpu_runtime = batched_single_gpu_benchmark(n=n,
                                                              batches=batches,
                                                              repetitions=repetitions,
                                                              use_map_combine=True)
    means_of_computation_to_runtime_map['Batched Cocos Single GPU Map Combine'] = batched_single_gpu_runtime
    print(f'Data processing using batched single GPU using Cocos map-combine performed in '
          f'{batched_single_gpu_runtime} seconds')

    # single core benchmark
    single_core_runtime = single_core_benchmark(n, repetitions=repetitions)
    means_of_computation_to_runtime_map[SINGLE_CORE_NUMPY] = single_core_runtime
    print(f'Data processing using single core NumPy performed in {single_core_runtime} seconds')

    # multi core benchmark (does not make sense on Windows and Anaconda)
    multi_core_benchmark(n=100, core_config=range(1, multiprocessing.cpu_count() + 1), repetitions=repetitions)
    number_of_cores_to_runtime_map = multi_core_benchmark(n=n, core_config=range(1, multiprocessing.cpu_count() + 1))

    for number_of_cores_to_use, cpu_time in number_of_cores_to_runtime_map.items():
        means_of_computation_to_runtime_map[f'NumPy with {number_of_cores_to_use} CPU core(s)'] = cpu_time
        print(f'Data processing on {number_of_cores_to_use} core(s) using NumPy performed in {cpu_time} seconds')


# def process_data_in_infinite_loop():
#     a, b, c = generate_data(100000000, gpu=False)
#
#     while True:
#         process_data(a, b, c, gpu=False)


if __name__ == '__main__':
    use_profiler = True
    if use_profiler:
        profile = LineProfiler(batched_single_gpu_benchmark,
                               host_to_device_transfer_function,
                               device_to_host_transfer_function,
                               split_arrays,
                               map_reduce_single_gpu,
                               map_combine_single_gpu,
                               multi_core_benchmark,
                               map_combine_multicore)

        profile.enable_by_count()

    main()

    if use_profiler:
        profile.print_stats()
