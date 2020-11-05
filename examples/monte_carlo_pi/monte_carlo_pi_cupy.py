from contexttimer import Timer
import cupy
import math
import numpy


def estimate_pi_cupy(n: int, batches: int = 1) -> float:
    """
    Estimate the sum of the sum of the 2d slices.

    Args:
        n: (todo): write your description
        batches: (todo): write your description
    """
    n_per_batch = math.ceil(n/batches)

    pi = 0.0
    for _ in range(batches):
        x = cupy.random.rand(n_per_batch, dtype=numpy.float32)
        y = cupy.random.rand(n_per_batch, dtype=numpy.float32)

        in_quarter_circle = (x * x + y * y) <= 1.0
        del x, y
        pi += 4.0 * float(cupy.mean(in_quarter_circle))
        del in_quarter_circle

    return pi / batches


def single_gpu_cupy_benchmark(n: int, batches: int, repetitions: int = 1) -> float:
    """
    Perform a single benchmark benchmark.

    Args:
        n: (todo): write your description
        batches: (list): write your description
        repetitions: (int): write your description
    """
    with Timer() as timer:
        for _ in range(repetitions):
            estimate_pi_cupy(n, batches=batches)
            # cupy.cuda.Stream.null.synchronize()

    return timer.elapsed / repetitions


def main():
    """
    Main function.

    Args:
    """
    n = 1000000000
    repetitions = 1
    batches = 20

    try:
        single_gpu_cupy_benchmark(n=100, batches=1)
        single_gpu_cupy_runtime = single_gpu_cupy_benchmark(n=n, batches=batches, repetitions=repetitions)
        print(f'Estimation of pi using single GPU CuPy performed in {single_gpu_cupy_runtime} seconds')
    except Exception as e:
        print(e)
        print('CuPy is not installed or not working correctly.')


if __name__ == '__main__':
    main()
