from cocos.multi_processing.utilities import (
    generate_slices_with_batch_size,
    generate_slices_with_number_of_batches
)

from contexttimer import Timer
import multiprocessing
import numpy
import typing as tp

from cocos.device import sync
from cocos.multi_processing.map_reduce import map_reduce_multicore
from cocos.numerics.data_types import NumericArray
from cocos.numerics.numerical_package_selector import select_num_pack

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


def generate_data(n: int) -> tp.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    return numpy.random.rand(n), numpy.random.rand(n), numpy.random.rand(n)


def process_data(a: NumericArray,
                 b: NumericArray,
                 c: NumericArray,
                 gpu: bool):
    np = select_num_pack(gpu)
    # print(len(a))

    return (np.sin(a) + np.cos(b)) * np.arctan(c)


def single_core_benchmark(n: int, repetitions: int = 1) -> float:
    a, b, c = generate_data(n)

    with Timer() as timer:
        for _ in range(repetitions):
            process_data(a, b, c, gpu=False)

    return timer.elapsed / repetitions


def multi_core_benchmark(n: int, core_config: tp.Iterable[int], repetitions: int = 1) \
        -> tp.Dict[int, float]:
    # This does not increase performance on Anaconda and Windows as elementwise trigonometric
    # operations seem to be multi-threaded to begin with.

    number_of_cores_to_runtime_map = {}
    a_complete, b_complete, c_complete = generate_data(n)

    for number_of_cores in core_config:
        with Timer() as timer:
            for _ in range(repetitions):
                kwargs_list = split_arrays(a=a_complete,
                                           b=b_complete,
                                           c=c_complete,
                                           number_of_batches=number_of_cores)

                result = \
                    map_reduce_multicore(f=lambda a, b, c: process_data(a, b, c, gpu=False),
                                         reduction=lambda x, y: numpy.hstack((x, y)),
                                         # initial_value=numpy.zeros((0, )),
                                         kwargs_list=kwargs_list)
                assert isinstance(result, numpy.ndarray)
                assert result.shape == (n,)

        number_of_cores_to_runtime_map[number_of_cores] = timer.elapsed / repetitions

    return number_of_cores_to_runtime_map


def main():
    n = 100000000
    repetitions = 1
    batches = 20
    means_of_computation_to_runtime_map = {}

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


def process_data_in_infinite_loop():
    a, b, c = generate_data(100000000)

    while True:
        process_data(a, b, c, gpu=False)


if __name__ == '__main__':
    main()
