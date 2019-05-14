import time
import cocos.device as cd


def estimate_pi(n: int, gpu: bool = True) -> float:
    if gpu:
        import cocos.numerics as np
        import cocos.numerics.random as random
    else:
        import numpy as np
        import numpy.random as random

    x = np.random.rand(n)
    y = np.random.rand(n)
    in_quarter_circle = (x * x + y * y) <= 1.0
    estimate = int(np.sum(in_quarter_circle))

    return estimate / n * 4


# initialize cocos device - the architecture is selected automatically
# the architecture can be specified explitly by providing one of
#   'cpu', 'cuda', and 'opencl'
cd.init()

# print information regarding the available devices in the machine
cd.info()

# number of draws
n = 100000000
print(f'simulating {n} draws')

# run estimation of Pi on the cpu via numpy
pi_cpu = estimate_pi(n, gpu=False)
print(f'Estimate of Pi on cpu: {pi_cpu}')

# run estimation of Pi on the cpu via numpy
pi_gpu = estimate_pi(n, gpu=True)
print(f'Estimate of Pi on gpu: {pi_gpu}')

# run a benchmark - repeating the simulation R times on both cpu and gpu
R = 10

# on gpu
tic = time.time()
for r in range(R):
    pi_gpu = float(estimate_pi(n, gpu=True))
    cd.sync()

time_on_gpu = time.time() - tic

print(f'time elapsed on gpu: {time_on_gpu}')


# on cpu
tic = time.time()
for r in range(R):
    pi_cpu = float(estimate_pi(n, gpu=False))

time_on_cpu = time.time() - tic

print(f'time elapsed on cpu: {time_on_cpu}')

# compute and print the speedup factor
print(f'speedup factor on gpu: {time_on_cpu/time_on_gpu}')
