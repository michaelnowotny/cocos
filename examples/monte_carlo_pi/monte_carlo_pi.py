import time
import cocos.device as cd
from cocos.numerics.numerical_package_selector import select_num_pack
import cocos.numerics as cn
import numpy


def estimate_pi(n: int, gpu: bool = True) -> float:
    """
    Estimate a random pi.

    Args:
        n: (int): write your description
        gpu: (int): write your description
    """
    np = select_num_pack(gpu)

    x = np.random.rand(n)
    y = np.random.rand(n)

    in_quarter_circle = (x * x + y * y) <= 1.0
    return 4.0 * float(np.mean(in_quarter_circle))


# initialize cocos device - the architecture is selected automatically
# the architecture can be specified explitly by providing one of
#   'cpu', 'cuda', and 'opencl'
cd.init()

# print information regarding the available devices on the machine
cd.info()

# number of draws
n = 100000000
print(f'simulating {n} draws')

# run estimation of Pi on the cpu via numpy
pi_cpu = estimate_pi(n, gpu=False)
print(f'Estimate of Pi on cpu: {pi_cpu}')

# run estimation of Pi on the gpu via Cocos
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
