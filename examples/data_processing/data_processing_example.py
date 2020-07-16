from cocos.multi_processing.utilities import (
    generate_slices_with_batch_size,
    generate_slices_with_number_of_batches
)

from contexttimer import Timer
import numpy

from cocos.numerics.data_types import NumericArray
from cocos.numerics.numerical_package_selector import select_num_pack

SINGLE_CORE_NUMPY = 'Single Core NumPy'


def split_arrays(a: NumericArray,
                 b: NumericArray,
                 c: NumericArray,
                 number_of_batches: int):
    pass


def process_data(a: NumericArray,
                 b: NumericArray,
                 c: NumericArray,
                 batches: int,
                 gpu: bool):
    np = select_num_pack(gpu)

    return (np.sin(a) + np.cos(b)) * np.arctan(c)


def single_core_benchmark(n: int, batches: int, repetitions: int = 1) -> float:
    a = numpy.random.rand(n)
    b = numpy.random.rand(n)
    c = numpy.random.rand(n)

    with Timer() as timer:
        for _ in range(repetitions):
            process_data(a, b, c, batches=batches, gpu=False)

    return timer.elapsed / repetitions


def main():
    n = 10000000
    repetitions = 10
    batches = 20
    means_of_computation_to_runtime_map = {}

    # single core benchmark
    single_core_runtime = single_core_benchmark(n, batches=batches, repetitions=repetitions)
    means_of_computation_to_runtime_map[SINGLE_CORE_NUMPY] = single_core_runtime
    print(f'Data processing using single core NumPy performed in {single_core_runtime} seconds')


if __name__ == '__main__':
    main()
