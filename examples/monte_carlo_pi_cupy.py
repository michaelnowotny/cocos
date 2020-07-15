from contexttimer import Timer
import math


def estimate_pi_cupy(n: int, batches: int = 1) -> float:
    import cupy as np

    n_per_batch = math.ceil(n/batches)

    pi = 0.0
    for _ in range(batches):
        x = np.random.rand(n_per_batch)
        y = np.random.rand(n_per_batch)

        in_quarter_circle = (x * x + y * y) <= 1.0
        pi += 4.0 * float(np.mean(in_quarter_circle))

    return pi / batches


def single_gpu_cupy_benchmark(n: int, batches: int, repetitions: int = 1) -> float:
    with Timer() as timer:
        for _ in range(repetitions):
            estimate_pi_cupy(n, batches=batches)

    return timer.elapsed / repetitions


def main():
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
