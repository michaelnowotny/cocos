# ![coconut](https://raw.githubusercontent.com/michaelnowotny/cocos/master/images/coconut-isolated-tiny.jpg) Cocos (Core Computational System) - Scientific GPU Computing in Python


## Overview

Cocos is a package for numeric and scientific computing on GPUs for Python with a NumPy-like API. 
It supports both CUDA and OpenCL on Windows, Mac OS, and Linux. 
Internally, it relies on the ArrayFire C/C++ library.
Cocos offers a multi-GPU map-reduce framework. 
In addition to its numeric functionality, it allows parallel computation of SymPy expressions on the GPU.

## Highlights

*   Fast vectorized computation on GPUs with a NumPy-like API.
*   Multi GPU support via map-reduce.
*   High-performance random number generators for beta, chi-square, exponential, gamma, logistic, lognormal, normal, uniform, and Wald distributions. Antithetic random numbers for uniform and normal distributions.
*   Provides a GPU equivalent to SymPy's lambdify, which enables numeric evaluation of symbolic SymPy (multi-dimensional array) expressions on the GPU for vectors of input parameters in parallel.
*   Adaptation of SciPy's gaussian_kde to the GPU

## Table of Contents

1.  [Installation](#installation)  
2.  [Getting Started](#getting-started)  
3.  [Multi-GPU Computing](#multi-gpu-computing)
4.  [Memory Limitations on the GPU Device](#memory-limitations-on-the-gpu-device)
5.  [Examples](#packaged-examples)  
5.1. [Estimating Pi via Monte Carlo](#estimating-pi-via-monte-carlo)  
5.2. [Option Pricing in a Stochastic Volatility Model via Monte Carlo](#option-pricing-in-a-stochastic-volatility-model-via-monte-carlo)  
5.3. [Numeric evaluation of SymPy array expressions on the GPU](#numeric-evaluation-of-sympy-array-expressions-on-the-gpu)  
5.4. [Kernel Density Estimation](#kernel-density-estimation)  
6.  [Benchmark](#benchmark)  
7.  [Functionality](#functionality)  
8.  [Limitations and Differences with NumPy](#limitations-and-differences-with-numpy)  
9.  [A Note on Hardware Configurations for Multi-GPU Computing](#a-note-on-hardware-configurations-for-multi-gpu-computing)  
10.  [License](#license)  

## Installation

### 1.  Download and install 
- Windows or Linux: [ArrayFire 3.6.4](http://arrayfire.s3.amazonaws.com/index.html#!/3.6.4%2F)
- MacOS: [ArrayFire 3.5.1](http://arrayfire.s3.amazonaws.com/index.html#!/3.5.1%2F)  

### 2. Make sure that your System is able to locate ArrayFire's libraries
ArrayFire's functionality is contained in dynamic libries, 
dynamic link libraries (.dll) on Windows and shared objects (.so) on Unix.  

This step is to ensure that these library files can be located on your system. 
On Windows, this can be done by adding `%AF_PATH%\lib` to the path environment variable. 
On Linux and Mac, one can either install (or copy) the ArrayFire libraries and their dependencies 
to `/usr/local/lib` or modify the environment variable `LD_LIBRARY_PATH` (Linux) or 
`DYLD_LIBRARY_PATH` (MacOS) to include the ArrayFire library directory.  
    
### 3.  Install Cocos via PIP: 
<pre>
pip install cocos 
</pre>
or 
<pre>
pip3 install cocos 
</pre>
if not using Anaconda.

To get the latest version, clone the repository from github, 
open a terminal/command prompt, navigate to the root folder and install via
<pre>
pip install .
</pre>
or 
<pre>
pip3 install . 
</pre>
if not using Anaconda.

## Getting Started
### Platform Information:
Print available devices
<pre>
import cocos.device as cd
cd.info()
</pre>

Select a device
<pre>
cd.ComputeDeviceManager.set_compute_device(0)
</pre>

### First Steps:
<pre>    
# begin by importing the numerics package
import cocos.numerics as cn

# create two arrays from lists
a = cn.array([[1.0, 2.0], [3.0, 4.0]])
b = cn.array([[5.0], [6.0]])

# print their contents
print(a)
print(b)

# matrix product of b and a
c = a @ b
print(c)

# create an array of normally distributed random numbers
d = cn.random.randn(2, 2)
print(d)
</pre>

## Multi-GPU Computing:
Cocos provides `map-reduce` as well as the related `map-combine` as multi-GPU 
programming models. The computations are separated into 'batches' and then distributed 
across GPU devices in a pool. Cocos implements multi-GPU support via process-based parallelism. 

To run the function `my_gpu_function` over separate batches of input data on multiple 
GPUs in parallel, first create a `ComputeDevicePool`:
<pre>
compute_device_pool = cocos.multi_processing.device_pool.ComputeDevicePool()
</pre>

To construct the batches, separate the arguments of the function into 
*   a list of args lists and (one list per batch)
*   a list of kwargs dictionaries (one dictionary per batch)

<pre>
args_list = [args_list_1, arg_list_2, ..., arg_list_n]
kwargs_list = [kwargs_dict_1, kwargs_dict_2, ..., kwargs_dict_n]
</pre>

Run the function in separate batches via `map_reduce`
<pre>
result = \
    compute_device_pool.map_reduce(f=my_gpu_function,
                                   reduction=my_reduction_function,
                                   initial_value=...,
                                   host_to_device_transfer_function=...,
                                   device_to_host_transfer_function=...,
                                   args_list=args_list
                                   kwargs_list=kwargs_list)

</pre>
The reduction function iteratively aggregates two results from the list of results generated by 
`my_gpu_function` from left to right, beginning at `initial_value` (i.e. reducing 
`initial_value` and the result of `my_gpu_function` corresponding to the first batch). 
The list of results is in the same order to the list of args and kwargs.

If the function requires input arrays on the GPU, it must be provided to `map_reduce` as 
a NumPy array. The data is then sent to the process managing the GPU assigned to this 
batch, where it is moved to the GPU device by a `host_to_device_transfer_function`. 
This function needs to be implemented by the user.

Likewise, results that involve GPU arrays are transferred to the host via a user-supplied 
`device_to_host_transfer_function` and are then sent back to the main process before 
reduction takes place.  

`map_combine` is a variation of `map_reduce`, in which a `combination` function aggregates the 
list of results in a single step. 

Please refer to the documentation of `cocos.multi_processing.device_pool.ComputeDevicePool.map_reduce` as well as 
`cocos.multi_processing.device_pool.ComputeDevicePool.map_combine` for further details. 
See 'examples/heston_pricing_multi_gpu_example.py' for a fully worked example. 

## Memory Limitations on the GPU Device
It is common for modern standard desktop computers to support up to support up to 128GB of RAM. 
Video cards by contrast only feature a small fraction of VRAM. The consequence is that algorithms that 
work well on a CPU can experience into memory limitations when run on a GPU device.

In some cases this problem can be resolved by running the computation in batches or chunks and transferring results 
from the GPU to the host after each batch has been processed.

Using `map_reduce_single_device` and `map_combine_single_device` found in 
`cocos.multi_processing.single_device_batch_processing`, computations on a single GPU can be split into chunks 
and run sequentially. The interface is modeled after the multi GPU functionality described in the previous section. 

Calls to `map_reduce_single_device` and `map_combine_single_device` can  be nested in a multi GPU computation, 
which is how multi GPU evaluation of kernel density estimates is realized in Cocos 
(see `cocos.scientific.kde.gaussian_kde.evaluate_in_batches_on_multiple_gpus`).

## Packaged examples:
1.  [Estimating Pi via Monte Carlo](#estimating-pi-via-monte-carlo)  
2.  [Option Pricing in a Stochastic Volatility Model via Monte Carlo](#option-pricing-in-a-stochastic-volatility-model-via-monte-carlo)  
3.  [Numeric evaluation of SymPy array expressions on the GPU](#numeric-evaluation-of-sympy-array-expressions-on-the-gpu)  
4.  [Kernel Density Estimation](#kernel-density-estimation)  

### Estimating Pi via Monte-Carlo

The following code estimates Pi via Monte-Carlo simulation. 
Since Cocos offers a NumPy-like API, the same code works on the both the GPU and the CPU via NumPy.

<pre>    
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

# run estimation of Pi on the cpu via NumPy
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
    pi_gpu = estimate_pi(n, gpu=True)
    cd.sync()

time_on_gpu = time.time() - tic

print(f'time elapsed on gpu: {time_on_gpu}')

# on cpu
tic = time.time()
for r in range(R):
    pi_cpu = estimate_pi(n, gpu=False)

time_on_cpu = time.time() - tic

print(f'time elapsed on cpu: {time_on_cpu}')

# compute and print the speedup factor
print(f'speedup factor on gpu: {time_on_cpu/time_on_gpu}')
</pre>


### Option Pricing in a Stochastic Volatility Model via Monte Carlo
In this example, we are simulating sample paths of the logarithmic price of an 
underlying security under the risk-neutral probability measure via the 
Euler–Maruyama discretization method.

The stochastic process is Heston's classical 1992 setup of a security price 
subject to stochastic volatility. The log price and its instantanous variance 
are governed by the following system of stochastic differential equations (SDE):

![log price sde](https://raw.githubusercontent.com/michaelnowotny/cocos/master/images/security_price_sde.png)

![volatility sde](https://raw.githubusercontent.com/michaelnowotny/cocos/master/images/volatility_sde.png)

The simulation code below demonstrates how to write code that supports both CPU 
and GPU computing. The complete code is available under examples.

<pre>
def simulate_heston_model(T: float,
                          N: int,
                          R: int,
                          mu: float,
                          kappa: float,
                          v_bar: float,
                          sigma_v: float,
                          rho: float,
                          x0: float,
                          v0: float,
                          gpu: bool = False) -> tp.Tuple:
    """
    This function simulates R paths from the Heston stochastic volatility model
    over a time horizon of length T divided into N steps.

    :param T: time horizon of the simulation
    :param N: number of steps
    :param R: number of paths to simulate
    :param mu: expected return
    :param kappa: mean-reversion speed of volatility
    :param v_bar: long-run mean of volatility
    :param sigma_v: volatility of volatility
    :param rho: instantaneous correlation of shocks to price and to volatility
    :param x0: initial log price
    :param v0: initial volatility
    :param gpu: whether to compute on the GPU
    :return: a tuple of two R-dimensional numeric arrays for log price and
             volatility
    """
    if gpu:
        import cocos.numerics as np
        import cocos.numerics.random as random
    else:
        import numpy as np
        import numpy.random as random

    Delta_t = T / float(N - 1)

    x = [np.full((R,), x0, dtype=numpy.float32),
         np.zeros((R,), dtype=numpy.float32)]

    v = [np.full((R,), v0, dtype=numpy.float32),
         np.zeros((R,), dtype=numpy.float32)]

    sqrt_delta_t = math.sqrt(Delta_t)
    sqrt_one_minus_rho_square = math.sqrt(1 - rho ** 2)

    # m = np.array([[rho, sqrt_one_minus_rho_square]])
    m = np.zeros((2,), dtype=numpy.float32)
    m[0] = rho
    m[1] = sqrt_one_minus_rho_square
    zero_array = np.zeros((R,), dtype=numpy.float32)

    t_current = 0
    for t in range(1, N):
        t_previous = (t + 1) % 2
        t_current = t % 2

        # generate antithetic standard normal random variables
        dBt = random.randn(R, 2) * sqrt_delta_t

        sqrt_v_lag = np.sqrt(v[t_previous])
        x[t_current] = x[t_previous] \
                     + (mu - 0.5 * v[t_previous]) * Delta_t \
                     + np.multiply(sqrt_v_lag, dBt[:, 0])
        v[t_current] = v[t_previous] \
                     + kappa * (v_bar - v[t_previous]) * Delta_t \
                     + sigma_v * np.multiply(sqrt_v_lag, np.dot(dBt, m))
        v[t_current] = np.maximum(v[t_current], 0.0)

    x = x[t_current]
    v = np.maximum(v[t_current], 0.0)

    return x, v
</pre>

The following code computes the option price from simulated price paths of the 
underlying. It demonstrates how to dynamically choose between Cocos and NumPy 
based on the type input array. Note that in this particular setup, one would 
typically use Fourier techniques to price the option rather than Monte Carlo simulation. 
Simulation techniques can be useful when considering stochastic processes outside the 
affine framework or more generally whenever the conditional characteristic function 
of the transition density is costly to evaluate or when considering path-dependent options.  

<pre>
def compute_option_price(r: float,
                         T: float,
                         K: float,
                         x_simulated,
                         num_pack: tp.Optional[types.ModuleType] = None):
    """
    Compute the function of a plain-vanilla call option from simulated
    log-returns.

    :param r: the risk-free rate
    :param T: the time to expiration
    :param K: the strike price
    :param x_simulated: a numeric array of simulated log prices of the underlying
    :param num_pack: a module - either numpy or cocos.numerics
    :return: option price
    """

    if not num_pack:
        use_gpu, num_pack = get_gpu_and_num_pack_by_dtype(x_simulated)

    return math.exp(-r * T) \
           * num_pack.mean(num_pack.maximum(num_pack.exp(x_simulated) - K, 0))
</pre>


### Numeric evaluation of SymPy array expressions on the GPU
Cocos can compile symbolic arrays defined using SymPy to GPU or CPU code. 

As an example, consider the following vector valued function.  
![vector_valued_function](https://raw.githubusercontent.com/michaelnowotny/cocos/master/images/f_x1_x2_x3.png).  
Here g is a scalar valued function that can be specified at a later point when 
numerical evaluation is of interest. 
One reason for this setup would be to retain generality. 
Another reason might be that a symbolic representation of this function is not 
available but one has access to an algorithm (a Python function) that can compute g. 


The goal is to evaluate this function at many x = (x_1, x_2, x_3) in parallel on a GPU. 

In order to specify this function using SumPy, begin by defining symbolic arguments
<pre>
x1, x2, x3, t = sym.symbols('x1, x2, x3, t')
</pre>

Separate the state symbols 'x1', 'x2', and 'x3' from the time symbol 't' and collect them in a tuple
<pre>
argument_symbols = (x1, x2, x3)
</pre>

Declare an as of yet unspecified function g
<pre>
g = sym.Function('g')
</pre>

Define the function f
<pre>
f = sym.Matrix([[x1 + x2], [(g(t) * x1 + x3) ** 2], [sym.exp(x1 + x2 + g(t))]])
</pre>

Compute the Jacobian of g w.r.t x1, x2, and x3 symbolically
<pre>
jacobian_f = f.jacobian([x1, x2, x3])
</pre>
The Jacobian is given by  
![jacobian_of_vector_valued_function](https://raw.githubusercontent.com/michaelnowotny/cocos/master/images/jacobian_f_x1_x2_x3.png)

Specify the concrete form of g as g(t) = ln(t)
<pre>
def numeric_time_function(t: float):
    return np.log(t)
</pre>

Convert the symbolic array expression to an object that can evaluated numerically on the cpu or gpu
<pre>
jacobian_f_lambdified \
    = LambdifiedMatrixExpression(
        argument_symbols=argument_symbols,
        time_symbol=t,
        symbolic_matrix_expression=jacobian_f,
        symbolic_time_function_name_to_numeric_time_function_map={'g': numeric_time_function})
</pre>

Generate n = 10000000 random vectors for x1, x2, and x3 at which to evaluate the function in parallel
<pre>
n = 10000000
X_gpu = cn.random.rand(n, 3)
X_cpu = np.array(X_gpu)
</pre>

Numerically evaluate the Jacobian on the GPU for t=1
<pre>
jacobian_f_numeric_gpu = \
    (jacobian_f_lambdified
     .evaluate_with_kwargs(x1=X_gpu[:, 0],
                           x2=X_gpu[:, 1],
                           x3=X_gpu[:, 2],
                           t=1.0))
</pre>

Numerically evaluate the Jacobian on the CPU for t=1
<pre>
jacobian_f_numeric_gpu = \
    (jacobian_f_lambdified
     .evaluate_with_kwargs(x1=X_cpu[:, 0],
                           x2=X_cpu[:, 1],
                           x3=X_cpu[:, 2],
                           t=1.0))
</pre>

Verify that the results match
<pre>
print(f'numerical results from cpu and gpu match: '
      f'{np.allclose(jacobian_f_numeric_gpu, jacobian_f_numeric_cpu)}')
</pre>

### Kernel Density Estimation
`cocos.scientific.kde import gaussian_kde` is a replacement for SciPy's 
`scipy.stats.kde.gaussian_kde` class that works on the GPU. 
GPU support is turned on by setting `gpu=True` in its constructor. 
Evaluating a kernel density estimate from a dataset in the NumPy array `points` 
on a grid in the NumPy array `grid` works as follows:
<pre>
from cocos.scientific.kde import gaussian_kde
gaussian_kde = gaussian_kde(points, gpu=True)
density_estimate = gaussian_kde_cocos.evaluate(grid)
</pre>

To preserve GPU memory, the evaluation of the KDE can proceed sequentially in batches as follows
<pre>
density_estimate = gaussian_kde_cocos.evaluate_in_batches(grid, maximum_number_of_elements_per_batch)
</pre> 
where the `maximum_number_of_elements_per_batch` is the maximum number of data points times 
evaluation points to process in a single batch.

The Kernel density estimate can be computed on different points on multiple GPUs in parallel 
using the method `evaluate_in_batches_on_multiple_gpus` as follows
<pre>
comnpute_device_pool = ComputeDevicePool()
gaussian_kde = gaussian_kde(points, gpu=False)
density_estimate = \
    gaussian_kde_cocos.evaluate_in_batches_on_multiple_gpus(grid, 
                                                            maximum_number_of_elements_per_batch, 
                                                            comnpute_device_pool)
</pre> 
Note that the `gpu` parameter in the `gaussian_kde` constructor must be set to `false` for multi gpu 
support and `points` must be a a NumPy array.

## Benchmark
### Monte Carlo Pi Benchmark
This benchmark compares the runtime performance of the Monte Carlo pi example 
using NumPy on 1 through 8 cpu cores as well as 1-2 GPUs using Cocos. 

The results were produced on a machine with an Intel Core i7 9700K with 128GB of 
RAM and a NVidia GeForce GTX 1060 running Windows 10. A total of 1000000000 points 
are drawn in 20 batches.

<table>
<tbody>
<tr>
<th></th>
<th>Total Time in Seconds</th>
<th>Speedup Compared to NumPy</th>
</tr>
<tr>
<td>Single Core NumPy</td>
<td>17.2867612</td>
<td>1.0</td>
</tr>
<tr>
<td>NumPy with 1 CPU core(s)</td>
<td>17.1750117</td>
<td>1.0065065166738723</td>
</tr>
<tr>
<td>NumPy with 2 CPU core(s)</td>
<td>10.494477000000003</td>
<td>1.6472246496895457</td>
</tr>
<tr>
<td>NumPy with 3 CPU core(s)</td>
<td>8.422800300000006</td>
<td>2.0523769511667025</td>
</tr>
<tr>
<td>NumPy with 4 CPU core(s)</td>
<td>7.082252900000007</td>
<td>2.440856242227665</td>
</tr>
<tr>
<td>NumPy with 5 CPU core(s)</td>
<td>6.365301000000002</td>
<td>2.715780636296696</td>
</tr>
<tr>
<td>NumPy with 6 CPU core(s)</td>
<td>5.8881023</td>
<td>2.935879901407284</td>
</tr>
<tr>
<td>NumPy with 7 CPU core(s)</td>
<td>5.609009299999997</td>
<td>3.081963369181793</td>
</tr>
<tr>
<td>NumPy with 8 CPU core(s)</td>
<td>5.667201699999993</td>
<td>3.0503169139012685</td>
</tr>
<tr>
<td>Cocos Single GPU</td>
<td>0.17866180000000043</td>
<td>96.75689599007711</td>
</tr>
<tr>
<td>Cocos with 1 GPU(s)</td>
<td>0.1841428000000036</td>
<td>93.87693246762655</td>
</tr>
<tr>
<td>Cocos with 2 GPU(s)</td>
<td>0.09644749999999647</td>
<td>179.23493299464096</td>
</tr>
</table>

![benchmark_results](https://raw.githubusercontent.com/michaelnowotny/cocos/master/images/monte_carlo_pi_benchmark_results.png)

Package versions used:
- arrayfire: 3.6.4
- arrayfire-python: 3.6.20181017
- cocos: 0.1.14
- CUDA: 9.2
- cupy-cuda92: 6.2.0 
- NumPy: 1.16.4
- Python: 3.7.3


### Stochastic Volatility Model Benchmark
This benchmark compares the runtime performance of the option pricing example 
under a Heston stochastic volatility model on the CPU using NumPy on a single 
core as well as on all cores simultaneously and the GPU using Cocos and CuPy. 
CuPy is another package that provides a NumPy-like API for GPU computing.

The results were produced on a machine with an Intel Core i7 9700K with 128GB of 
RAM and two NVidia GeForce GTX 1060 running Windows 10. Two Million paths are being simulated with 
500 time steps per year.

<table>
<tbody>
<tr>
<th></th>
<th>Total Time in Seconds</th>
<th>Speedup Compared to NumPy</th>
</tr>
<tr>
<td>NumPy</td>
<td>32.782310247421265</td>
<td>1.0</td>
</tr>
<tr>
<td>Cocos</td>
<td>1.856126070022583</td>
<td>17.661682994960795</td>
</tr>
<tr>
<td>CuPy</td>
<td>2.815166473388672</td>
<td>11.64489224964396</td>
</tr>
<tr>
<td>NumPy Multicore</td>
<td>7.143897294998169</td>
<td>4.588855199580479</td>
</tr>
<tr>
<td>Cocos on 1 GPU</td>
<td>1.8460988998413086</td>
<td>17.757613229843344</td>
</tr>
<tr>
<td>Cocos on 2 GPUs</td>
<td>0.9753890037536621</td>
<td>33.60947285776512</td>
</tr>
</table>

![benchmark_results](https://raw.githubusercontent.com/michaelnowotny/cocos/master/images/heston_benchmark_results.png)

Package versions used:
- arrayfire: 3.6.4
- arrayfire-python: 3.6.20181017
- cocos: 0.1.14
- CUDA: 9.2
- cupy-cuda92: 6.2.0 
- NumPy: 1.16.4
- Python: 3.7.3

## Functionality

Most differences between NumPy and Cocos stem from two sources:

1.  NumPy is row-major (C style) whereas ArrayFire is column-major (Fortran style).
2.  Only part of NumPy's functionality is present in ArrayFire, which Cocos is based on.

### Attributes of the ndarray class

<table>

<tbody>

<tr>

<th>Attribute</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>T</td>

<td>Same as self.transpose(), except that self is returned if self.ndim < 2.</td>

<td></td>

</tr>

<tr>

<td>H</td>

<td>Conjugate transpose.</td>

<td>attribute is not present in NumPy</td>

</tr>

<tr>

<td>data</td>

<td>Python buffer object pointing to the start of the array’s data.</td>

<td>attribute not implemented</td>

</tr>

<tr>

<td>dtype</td>

<td>Data-type of the array’s elements.</td>

<td></td>

</tr>

<tr>

<td>flags</td>

<td>Information about the memory layout of the array.</td>

<td>attribute not implemented</td>

</tr>

<tr>

<td>flat</td>

<td>A 1-D iterator over the array.</td>

<td>attribute not implemented</td>

</tr>

<tr>

<td>imag</td>

<td>The imaginary part of the array.</td>

<td></td>

</tr>

<tr>

<td>real</td>

<td>The real part of the array.</td>

<td></td>

</tr>

<tr>

<td>size</td>

<td>Number of elements in the array.</td>

<td></td>

</tr>

<tr>

<td>itemsize</td>

<td>Length of one array element in bytes.</td>

<td></td>

</tr>

<tr>

<td>nbytes</td>

<td>Total bytes consumed by the elements of the array.</td>

<td></td>

</tr>

<tr>

<td>ndim</td>

<td>Number of array dimensions.</td>

<td></td>

</tr>

<tr>

<td>shape</td>

<td>Tuple of array dimensions.</td>

<td></td>

</tr>

<tr>

<td>strides</td>

<td>Tuple of bytes to step in each dimension when traversing an array.</td>

<td></td>

</tr>

<tr>

<td>ctypes</td>

<td>An object to simplify the interaction of the array with the ctypes module.</td>

<td>attribute not implemented</td>

</tr>

<tr>

<td>base</td>

<td>Base object if memory is from some other object.</td>

<td>attribute not implemented</td>

</tr>

</tbody>

</table>

### Methods of the ndarray class

<table>

<tbody>

<tr>

<th>Method</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>all(axis=None)</td>

<td>Returns True if all elements evaluate to True.</td>

<td>parameters "out" and "keepdims" are not supported</td>

</tr>

<tr>

<td>any(axis=None)</td>

<td>Returns True if any of the elements of a evaluate to True.</td>

<td>parameters "out" and "keepdims" are not supported</td>

</tr>

<tr>

<td>argmax(axis=None)</td>

<td>Return indices of the maximum values along the given axis.</td>

<td>parameter "out" is not supported</td>

</tr>

<tr>

<td>argmin(axis=None)</td>

<td>Return indices of the minimum values along the given axis of a.</td>

<td>parameter "out" is not supported</td>

</tr>

<tr>

<td>argpartition</td>

<td>Returns the indices that would partition this array.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>argsort(axis=-1, ascending=True)</td>

<td>Returns the indices that would sort this array.</td>

<td>has additional parameter "ascending"; does not support parameters "kind" and "order"</td>

</tr>

<tr>

<td>astype(dtype)</td>

<td>Copy of the array, cast to a specified type.</td>

<td>does not support parameters "order", "casting", "subok", and "copy"</td>

</tr>

<tr>

<td>byteswap</td>

<td>Swap the bytes of the array elements</td>

<td>method not implemented</td>

</tr>

<tr>

<td>choose</td>

<td>Use an index array to construct a new array from a set of choices.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>clip(min, max)</td>

<td>Return an array whose values are limited to [min, max].</td>

<td>parameter "out" is not supported</td>

</tr>

<tr>

<td>compress</td>

<td>Return selected slices of this array along given axis.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>conj()</td>

<td>Complex-conjugate all elements.</td>

<td></td>

</tr>

<tr>

<td>conjugate()</td>

<td>Return the complex conjugate, element-wise.</td>

<td></td>

</tr>

<tr>

<td>copy()</td>

<td>Return a copy of the array.</td>

<td></td>

</tr>

<tr>

<td>cumprod</td>

<td>Return the cumulative product of the elements along the given axis.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>cumsum(axis=-1)</td>

<td>Return the cumulative sum of the elements along the given axis.</td>

<td>parameters "dtype" and "out" are not supported</td>

</tr>

<tr>

<td>diagonal(offset = 0)</td>

<td>Return specified diagonals.</td>

<td>parameters "axis1" and "axis2" are not supported</td>

</tr>

<tr>

<td>dot(a, b)</td>

<td>Dot product of two arrays.</td>

<td>parameter "out" is not supported</td>

</tr>

<tr>

<td>dump</td>

<td>Dump a pickle of the array to the specified file.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>dumps</td>

<td>Returns the pickle of the array as a string.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>fill(value)</td>

<td>Fill the array with a scalar value.</td>

<td></td>

</tr>

<tr>

<td>flatten()</td>

<td>Return a copy of the array collapsed into one dimension.</td>

<td>parameter "order" is not supported; flattens array column by column in contrast to numpy, which operates row by row (may be changed in the future)</td>

</tr>

<tr>

<td>getfield</td>

<td>Returns a field of the given array as a certain type.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>item(*args)</td>

<td>Copy an element of an array to a standard Python scalar and return it.</td>

<td>implemented indirectly and therefore slow</td>

</tr>

<tr>

<td>itemset(*args)</td>

<td>Insert scalar into an array (scalar is cast to array’s dtype, if possible)</td>

<td>implemented indirectly and therefore slow</td>

</tr>

<tr>

<td>max(axis=None)</td>

<td>Return the maximum along a given axis.</td>

<td>parameter "out" is not supported; different criterion from numpy for complex arrays</td>

</tr>

<tr>

<td>mean(axis=None)</td>

<td>Returns the average of the array elements along given axis.</td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>min(axis=None)</td>

<td>Returns the average of the array elements along given axis.</td>

<td>parameter "out" is not supported; different criterion from numpy for complex arrays</td>

</tr>

<tr>

<td>newbyteorder</td>

<td>Return the array with the same data viewed with a different byte order.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>nonzero</td>

<td>Return the indices of the elements that are non-zero.</td>

<td>does not support arrays with more than three axis</td>

</tr>

<tr>

<td>partition</td>

<td>Rearranges the elements in the array in such a way that value of the element in kth position is in the position it would be in a sorted array.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>prod(axis=None)</td>

<td>Return the product of the array elements over the given axis</td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>ptp</td>

<td>Peak to peak (maximum - minimum) value along a given axis.</td>

<td>parameter "out" is not supported; different output from numpy for complex arrays due to differences of min and max criteria compared to numpy</td>

</tr>

<tr>

<td>put</td>

<td>Set a.flat[n] = values[n] for all n in indices.</td>

<td>method not implemented; straightforward once the attribute "flat" has been implemented</td>

</tr>

<tr>

<td>ravel</td>

<td>Return a flattened array.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>repeat(repeats, axis=None)</td>

<td>Repeat elements of an array.</td>

<td>order is reversed compared to numpy (see remarks for the method flatten)</td>

</tr>

<tr>

<td>reshape(shape)</td>

<td>Returns an array containing the same data with a new shape.</td>

<td>parameter "order" is not supported</td>

</tr>

<tr>

<td>resize(newshape)</td>

<td>Change shape and size of array in-place.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>round()</td>

<td>Return a with each element rounded to the given number of decimals.</td>

<td>parameters "decimals" and "out" are not supported</td>

</tr>

<tr>

<td>searchsorted</td>

<td>Find indices where elements of v should be inserted in a to maintain order.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>setfield</td>

<td>Put a value into a specified place in a field defined by a data-type.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>setflags</td>

<td>Set array flags WRITEABLE, ALIGNED, and UPDATEIFCOPY, respectively.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>sort(axis=-1, ascending=True)</td>

<td>Sort an array, in-place.</td>

<td>has additional parameter "ascending"; parameters "kind" and "order" are not supported</td>

</tr>

<tr>

<td>squeeze(axis=None)</td>

<td>Remove single-dimensional entries from the shape of a.</td>

<td></td>

</tr>

<tr>

<td>std(axis=None)</td>

<td>Returns the standard deviation of the array elements along given axis.</td>

<td>parameters "dtype", "out", "ddof", and "keepdims" are not supported</td>

</tr>

<tr>

<td>sum(axis=None)</td>

<td>Return the sum of the array elements over the given axis.</td>

<td>parameters "dtype", "out, and "keepdims" are not supported</td>

</tr>

<tr>

<td>swapaxes</td>

<td>Return a view of the array with axis1 and axis2 interchanged.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>take</td>

<td>Return an array formed from the elements of a at the given indices.</td>

<td>method not implemented</td>

</tr>

<tr>

<td>tobytes</td>

<td>Construct Python bytes containing the raw data bytes in the array.</td>

<td>method not implemented; could be implemented by first converting to numpy array for instance</td>

</tr>

<tr>

<td>tofile</td>

<td>Write array to a file as text or binary (default).</td>

<td>method not implemented; could be implemented by first converting to numpy array for instance</td>

</tr>

<tr>

<td>tolist</td>

<td>Return the array as a (possibly nested) list.</td>

<td>method not implemented; could be implemented by first converting to numpy array for instance</td>

</tr>

<tr>

<td>tostring</td>

<td>Construct Python bytes containing the raw data bytes in the array.</td>

<td>method not implemented; could be implemented by first converting to numpy array for instance</td>

</tr>

<tr>

<td>trace(offset=0)</td>

<td>Return the sum along diagonals of the array.</td>

<td>parameters "axis1", "axis2", "dtype", and "out" are not supported</td>

</tr>

<tr>

<td>transpose()</td>

<td>Returns a new array with axes transposed.</td>

<td>parameter "*axis" is not supported; look into arrayfire function "moddims" to support *axis; new array is not a view on the old data</td>

</tr>

<tr>

<td>var(axis=None, ddof=0)</td>

<td>Returns the variance of the array elements, along given axis.</td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>view</td>

<td>New view of array with the same data.</td>

<td>method not implemented</td>

</tr>

</tbody>

</table>

### Statistics Functions

#### Order Statistics

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>amin</td>

<td>Return the minimum of an array or minimum along an axis.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>amax</td>

<td>Return the maximum of an array or maximum along an axis.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nanmin</td>

<td>Return minimum of an array or minimum along an axis, ignoring any NaNs.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nanmax</td>

<td>Return the maximum of an array or maximum along an axis, ignoring any NaNs.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ptp(a, axis=None)</td>

<td>Range of values (maximum - minimum) along an axis.</td>

<td>parameter "out" not supported</td>

</tr>

<tr>

<td>percentile</td>

<td>Compute the qth percentile of the data along the specified axis.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nanpercentile</td>

<td>Compute the qth percentile of the data along the specified axis, while ignoring nan values.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Averages and Variances

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>median</td>

<td>Compute the median along the specified axis.</td>

<td>parameters "out", "overwrite_input"=False, and "keepdims" are not supported</td>

</tr>

<tr>

<td>average(a, axis=None, weights=None)</td>

<td>Compute the weighted average along the specified axis.</td>

<td>parameter "returned" is not supported; does not work on Mac OS X with AMD chip when weights are given</td>

</tr>

<tr>

<td>mean(a, axis=None)</td>

<td></td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>std(a, axis=None, ddof=0)</td>

<td>Compute the arithmetic mean along the specified axis.</td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>var(a, axis=None, ddof=0)</td>

<td>Compute the variance along the specified axis.</td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>nanmedian</td>

<td>Compute the median along the specified axis, while ignoring NaNs.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nanmean</td>

<td>Compute the arithmetic mean along the specified axis, ignoring NaNs.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nanstd</td>

<td>Compute the standard deviation along the specified axis, while ignoring NaNs.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nanvar</td>

<td>Compute the variance along the specified axis, while ignoring NaNs.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Correlating

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>corrcoeff(x, bias=False, ddof=None</td>

<td>Return Pearson product-moment correlation coefficients.</td>

<td>parameters "y" and "rowvar" are not supported</td>

</tr>

<tr>

<td>correlate</td>

<td>Cross-correlation of two 1-dimensional sequences.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>cov(m, bias=False, ddof=None)</td>

<td>Estimate a covariance matrix, given data and weights.</td>

<td>parameters "y", "rowvar", "fweights", and "aweights" are not supported</td>

</tr>

</tbody>

</table>

#### Histograms

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>histogram</td>

<td>Compute the histogram of a set of data.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>histogram2d</td>

<td>Compute the bi-dimensional histogram of two data samples.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>histogramdd</td>

<td>Compute the multidimensional histogram of some data.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>bincount</td>

<td>Count number of occurrences of each value in array of non-negative ints.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>digitize</td>

<td>Return the indices of the bins to which each value in input array belongs.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

### Array Creation Routines

#### Ones and Zeros

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>empty</td>

<td>Return a new array of given shape and type, initializing entries to zero.</td>

<td>in contrast to numpy it initializes entries to zero just like zeros</td>

</tr>

<tr>

<td>empty_like</td>

<td>Return a new array with the same shape and type as a given array.</td>

<td>in contrast to numpy it initializes entries to zero just like zeros</td>

</tr>

<tr>

<td>eye(N, M=None, k=0, dtype=np.float32)</td>

<td>Return a 2-D array with ones on the diagonal and zeros elsewhere.</td>

<td></td>

</tr>

<tr>

<td>identity(n, dtype=np.float32)</td>

<td>Return the identity array.</td>

<td></td>

</tr>

<tr>

<td>[ones(shape, dtype=np.float32)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.ones)</td>

<td>Return a new array of given shape and type, filled with ones.</td>

<td>parameter "order" is not supported</td>

</tr>

<tr>

<td>[ones_like(a, dtype=None)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.ones_like)</td>

<td>Return an array of ones with the same shape and type as a given array.</td>

<td>parameters "order" and "subok" are not supported</td>

</tr>

<tr>

<td>[zeros(shape, dtype=n.float32)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.zeros)</td>

<td>Return a new array of given shape and type, filled with zeros.</td>

<td>parameter "order" is not supported</td>

</tr>

<tr>

<td>[zeros_like(a, dtype=None)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.zeros_like)</td>

<td>Return an array of zeros with the same shape and type as a given array.</td>

<td>parameters "order" and "subok" are not supported</td>

</tr>

<tr>

<td>[full(shape, fill_value, dtype=n.float32)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.full)</td>

<td>Return a new array of given shape and type, filled with fill_value.</td>

<td>parameter "order" is not supported</td>

</tr>

<tr>

<td>[full_like(a, fill_value, dtype=None)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html#numpy.full_like)</td>

<td>Return a full array with the same shape and type as a given array.</td>

<td>parameters "order" and "subok" are not supported</td>

</tr>

</tbody>

</table>

#### From Existing Data

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>array(object, dtype=None)</td>

<td>Create an array.</td>

<td>parameters "copy", "order", "subok", and "ndmin" are not supported</td>

</tr>

<tr>

<td>asarray</td>

<td>Convert the input to an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asanyarray</td>

<td>Convert the input to an ndarray, but pass ndarray subclasses through.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ascontiguousarray</td>

<td>Return a contiguous array in memory (C order).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asmatrix</td>

<td>Interpret the input as a matrix.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>copy(a)</td>

<td>Return an array copy of the given object.</td>

<td>parameter "order" is not supported</td>

</tr>

<tr>

<td>frombuffer</td>

<td>Interpret a buffer as a 1-dimensional array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fromfile</td>

<td>Construct an array from data in a text or binary file.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fromfunction</td>

<td>Construct an array by executing a function over each coordinate.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fromiter</td>

<td>Create a new 1-dimensional array from an iterable object.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fromstring</td>

<td>A new 1-D array initialized from raw binary or text data in a string.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>loadtxt</td>

<td>Load data from a text file.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Building Matrices

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>diag(v, k=0)</td>

<td>Extract a diagonal or construct a diagonal array.</td>

<td></td>

</tr>

<tr>

<td>diagflat(v, k=0)</td>

<td>Create a two-dimensional array with the flattened input as a diagonal.</td>

<td></td>

</tr>

<tr>

<td>tri</td>

<td>An array with ones at and below the given diagonal and zeros elsewhere.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>tril(m)</td>

<td>Lower triangle of an array.</td>

<td>parameter "k" is not supported</td>

</tr>

<tr>

<td>triu(m)</td>

<td>Upper triangle of an array.</td>

<td>parameter "k" is not supported</td>

</tr>

<tr>

<td>vander</td>

<td>Generate a Vandermonde matrix.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

### Array Manipulation Routines

#### Basic Operations

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>copyto</td>

<td>Copies values from one array to another, broadcasting as necessary.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Changing Array Shape

<table>

<tbody>

<tr>

<th>Function</th>

<th>Description</th>

<th>Notes</th>

</tr>

<tr>

<td>reshape(a, newshape)</td>

<td>Gives a new shape to an array without changing its data.</td>

<td>parameter "order" is not supported</td>

</tr>

<tr>

<td>ravel</td>

<td>Return a contiguous flattened array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ndarray.flat</td>

<td>A 1-D iterator over the array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ndarray.flatten</td>

<td>Return a copy of the array collapsed into one dimension.</td>

<td>parameter "order" is not implemented</td>

</tr>

</tbody>

</table>

#### Transpose-like Operations

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>moveaxis</td>

<td>Move axes of an array to new positions.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>rollaxis(a, axis[, start])</td>

<td>Roll the specified axis backwards, until it lies in a given position.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>swapaxes(a, axis1, axis2)</td>

<td>Interchange two axes of an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ndarray.T</td>

<td>Same as self.transpose(), except that self is returned if self.ndim < 2.</td>

<td></td>

</tr>

<tr>

<td>transpose(a)</td>

<td>Permute the dimensions of an array.</td>

<td>parameter "axes" is not supported</td>

</tr>

</tbody>

</table>

#### Changing Number of Dimensions

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>atleast_1d(*arys)</td>

<td>Convert inputs to arrays with at least one dimension.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>atleast_2d(*arys)</td>

<td>View inputs as arrays with at least two dimensions.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>atleast_3d(*arys)</td>

<td>View inputs as arrays with at least three dimensions.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>broadcast</td>

<td>Produce an object that mimics broadcasting.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>broadcast_to(array, shape[, subok])</td>

<td>Broadcast an array to a new shape.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>broadcast_arrays(*args, **kwargs)</td>

<td>Broadcast any number of arrays against each other.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>expand_dims(a, axis)</td>

<td>Expand the shape of an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>[squeeze(a, axis=None)](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.squeeze.html)</td>

<td>Remove single-dimensional entries from the shape of an array.</td>

<td></td>

</tr>

</tbody>

</table>

#### Changing Kind of Array

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>asarray(a[, dtype, order])</td>

<td>Convert the input to an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asanyarray(a[, dtype, order])</td>

<td>Convert the input to an ndarray, but pass ndarray subclasses through.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asmatrix(data[, dtype])</td>

<td>Interpret the input as a matrix.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asfarray(a[, dtype])</td>

<td>Return an array converted to a float type.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asfortranarray(a[, dtype])</td>

<td>Return an array laid out in Fortran order in memory.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ascontiguousarray(a[, dtype])</td>

<td>Return a contiguous array in memory (C order).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asarray_chkfinite(a[, dtype, order])</td>

<td>Convert the input to an array, checking for NaNs or Infs.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>asscalar(a)</td>

<td>Convert an array of size 1 to its scalar equivalent.</td>

<td></td>

</tr>

<tr>

<td>require(a[, dtype, requirements])</td>

<td>Return an ndarray of the provided type that satisfies requirements.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Joining Arrays

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>concatenate((a1, a2, ...), axis=0)</td>

<td>Join a sequence of arrays along an existing axis.</td>

<td></td>

</tr>

<tr>

<td>stack(arrays[, axis])</td>

<td>Join a sequence of arrays along a new axis.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>column_stack(tup)</td>

<td>Stack 1-D arrays as columns into a 2-D array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>dstack(tup)</td>

<td>Stack arrays in sequence depth wise (along third axis).</td>

<td></td>

</tr>

<tr>

<td>hstack(tup)</td>

<td>Stack arrays in sequence horizontally (along second axis).</td>

<td></td>

</tr>

<tr>

<td>vstack(tup)</td>

<td>Stack arrays in sequence vertically (along first axis).</td>

<td></td>

</tr>

<tr>

<td>block(arrays)</td>

<td>Assemble an nd-array from nested lists of blocks.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Splitting Arrays

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>split(ary, indices_or_sections[, axis])</td>

<td>Split an array into multiple sub-arrays.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>array_split(ary, indices_or_sections[, axis])</td>

<td>Split an array into multiple sub-arrays.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>dsplit(ary, indices_or_sections)</td>

<td>Split array into multiple sub-arrays along the 3rd axis (depth).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>hsplit(ary, indices_or_sections)</td>

<td>Split an array into multiple sub-arrays horizontally (column-wise).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>vsplit(ary, indices_or_sections)</td>

<td>Split an array into multiple sub-arrays vertically (row-wise).</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Tiling Arrays

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>tile(A, reps)</td>

<td>Construct an array by repeating A the number of times given by reps.</td>

<td></td>

</tr>

<tr>

<td>repeat(a, repeats[, axis])</td>

<td>Repeat elements of an array.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Adding and Removing Elements (not implemented)

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>delete(arr, obj[, axis])</td>

<td>Return a new array with sub-arrays along an axis deleted.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>insert(arr, obj, values[, axis])</td>

<td>Insert values along the given axis before the given indices.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>append(arr, values[, axis])</td>

<td>Append values to the end of an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>resize(a, new_shape)</td>

<td>Return a new array with the specified shape.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>trim_zeros(filt[, trim])</td>

<td>Trim the leading and/or trailing zeros from a 1-D array or sequence.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>unique(ar[, return_index, return_inverse, ...])</td>

<td>Find the unique elements of an array.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Rearranging Elements

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>flip(m, axis)</td>

<td>Reverse the order of elements in an array along the given axis.</td>

<td></td>

</tr>

<tr>

<td>fliplr(m)</td>

<td>Flip array in the left/right direction.</td>

<td></td>

</tr>

<tr>

<td>flipud(m)</td>

<td>Flip array in the up/down direction.</td>

<td></td>

</tr>

<tr>

<td>reshape(a, newshape)</td>

<td>Gives a new shape to an array without changing its data.</td>

<td>parameter "order" is not supported</td>

</tr>

<tr>

<td>roll(a, shift, axis=None)</td>

<td>Roll array elements along a given axis.</td>

<td></td>

</tr>

<tr>

<td>rot90(m[, k, axes])</td>

<td>Rotate an array by 90 degrees in the plane specified by axes.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

### Binary Operations

#### Elementwise Bit Operations

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>bitwise_and(x1, x2)</td>

<td>Compute the bit-wise AND of two arrays element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>bitwise_or(x1, x2)</td>

<td>Compute the bit-wise OR of two arrays element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>bitwise_xor(x1, x2)</td>

<td>Compute the bit-wise XOR of two arrays element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>invert(x)</td>

<td>Compute bit-wise inversion, or bit-wise NOT, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>left_shift(x1, x2)</td>

<td>Shift the bits of an integer to the left.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>right_shift(x1, x2)</td>

<td>Shift the bits of an integer to the right.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

</tbody>

</table>

#### Bit Packing (not implemented)

### Indexing Routines

#### Generating Index Arrays (not implemented)

#### Indexing-like Operations

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>take(a, indices[, axis, out, mode])</td>

<td>Take elements from an array along an axis.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>choose(a, choices[, out, mode])</td>

<td>Construct an array from an index array and a set of arrays to choose from.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>compress(condition, a[, axis, out])</td>

<td>Return selected slices of an array along given axis.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>diag(v, k=0)</td>

<td>Extract a diagonal or construct a diagonal array.</td>

<td></td>

</tr>

<tr>

<td>diagonal(a[, offset, axis1, axis2])</td>

<td>Return specified diagonals.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>select(condlist, choicelist[, default])</td>

<td>Return an array drawn from elements in choicelist, depending on conditions.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>lib.stride_tricks.as_strided(x[, shape, ...])</td>

<td>Create a view into the array with the given shape and strides.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Inserting Data into Arrays (not implemented)

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>place(arr, mask, vals)</td>

<td>Change elements of an array based on conditional and input values.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>put(a, ind, v[, mode])</td>

<td>Replaces specified elements of an array with given values.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>putmask(a, mask, values)</td>

<td>Changes elements of an array based on conditional and input values.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fill_diagonal(a, val[, wrap])</td>

<td>Fill the main diagonal of the given array of any dimensionality.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Iterating over Arrays (not implemented)

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>nditer</td>

<td>Efficient multi-dimensional iterator object to iterate over arrays.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ndenumerate(arr)</td>

<td>Multidimensional index iterator.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ndindex(*shape)</td>

<td>An N-dimensional iterator object to index arrays.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>flatiter</td>

<td>Flat iterator object to iterate over arrays.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>lib.Arrayterator(var[, buf_size])</td>

<td>Buffered iterator for big arrays.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

### Linear Algebra

#### Matrix and Vector Products

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>dot(a, b)</td>

<td>Dot product of two arrays.</td>

<td>parameter "out" is not supported</td>

</tr>

<tr>

<td>linalg.multi_dot(arrays)</td>

<td>Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>vdot(a, b)</td>

<td>Return the dot product of two vectors.</td>

<td></td>

</tr>

<tr>

<td>inner(a, b)</td>

<td>Inner product of two arrays.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>outer(a, b)</td>

<td>Outer product of two arrays.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>matmul(a, b[, out])</td>

<td>Matrix product of two arrays.</td>

<td>only handles 2d arrays</td>

</tr>

<tr>

<td>tensordot(a, b[, axes])</td>

<td>Compute tensor dot product along specified axes for arrays >= 1-D.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>einsum(subscripts, *operands[, out, dtype, ...])</td>

<td>Evaluates the Einstein summation convention on the operands.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.matrix_power(M, n)</td>

<td>Raise a square matrix to the (integer) power n.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>kron(a, b)</td>

<td>Kronecker product of two arrays.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Decompositions

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>linalg.cholesky(a)</td>

<td>Cholesky decomposition.</td>

<td></td>

</tr>

<tr>

<td>linalg.qr(a)</td>

<td>Compute the qr factorization of a matrix.</td>

<td>parameter "mode" is not supported</td>

</tr>

<tr>

<td>linalg.svd(a[, full_matrices, compute_uv])</td>

<td>Singular Value Decomposition.</td>

<td></td>

</tr>

</tbody>

</table>

#### Matrix Eigenvalues (not implemented)

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>linalg.eig(a)</td>

<td>Compute the eigenvalues and right eigenvectors of a square array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.eigh(a[, UPLO])</td>

<td>Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.eigvals(a)</td>

<td>Compute the eigenvalues of a general matrix.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.eigvalsh(a[, UPLO])</td>

<td>Compute the eigenvalues of a Hermitian or real symmetric matrix.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Norms and Other Numbers

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>linalg.norm(x[, ord, axis, keepdims])</td>

<td>Matrix or vector norm.</td>

<td>parameter "keepdims" is not supported</td>

</tr>

<tr>

<td>linalg.cond(x[, p])</td>

<td>Compute the condition number of a matrix.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.det(a)</td>

<td>Compute the determinant of an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.matrix_rank(M, tol=None)</td>

<td>Return matrix rank of array using SVD method</td>

<td></td>

</tr>

<tr>

<td>linalg.slogdet(a)</td>

<td>Compute the sign and (natural) logarithm of the determinant of an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>trace(a, offset=0)</td>

<td>Return the sum along diagonals of the array.</td>

<td>parameters "axis1", "axis2", "dtype", and "out" are not supported</td>

</tr>

</tbody>

</table>

#### Solving Equations and Inverting Matrices

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>linalg.solve(a, b)</td>

<td>Solve a linear matrix equation, or system of linear scalar equations.</td>

<td>additional parameter "trans" indicating whether the argument should be transposed</td>

</tr>

<tr>

<td>linalg.tensorsolve(a, b[, axes])</td>

<td>Solve the tensor equation a x = b for x.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.lstsq(a, b[, rcond])</td>

<td>Return the least-squares solution to a linear matrix equation.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.inv(a)</td>

<td>Compute the (multiplicative) inverse of a matrix.</td>

<td></td>

</tr>

<tr>

<td>linalg.pinv(a[, rcond])</td>

<td>Compute the (Moore-Penrose) pseudo-inverse of a matrix.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>linalg.tensorinv(a[, ind])</td>

<td>Compute the ‘inverse’ of an N-dimensional array.</td>

<td></td>

</tr>

</tbody>

</table>

### Logic Functions

#### Truth Value Testing

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>all(a, axis=None)</td>

<td>Test whether all array elements along a given axis evaluate to True.</td>

<td>parameters "out" and "keepdims" are not supported</td>

</tr>

<tr>

<td>any(a, axis=None)</td>

<td>Test whether any array element along a given axis evaluates to True.</td>

<td>parameters "out" and "keepdims" are not supported</td>

</tr>

</tbody>

</table>

#### Array Contents

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>isfinite(x)</td>

<td>Test element-wise for finiteness (not infinity or not Not a Number).</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>isinf(x)</td>

<td>Test element-wise for positive or negative infinity.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>isnan(x)</td>

<td>Test element-wise for NaN and return result as a boolean array.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", and "extobj" are not supported</td>

</tr>

<tr>

<td>isneginf(x)</td>

<td>Test element-wise for negative infinity, return result as bool array.</td>

<td>parameter "out" is not supported</td>

</tr>

<tr>

<td>isposinf(x)</td>

<td>Test element-wise for positive infinity, return result as bool array.</td>

<td>parameter "out" is not supported</td>

</tr>

</tbody>

</table>

#### Array Type Testing

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>iscomplex(x)</td>

<td>Returns a bool array, where True if input element is complex.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>iscomplexobj(x)</td>

<td>Check for a complex type or an array of complex numbers.</td>

<td></td>

</tr>

<tr>

<td>isfortran(a)</td>

<td>Returns True if the array is Fortran contiguous but not C contiguous.</td>

<td></td>

</tr>

<tr>

<td>isreal(x)</td>

<td>Returns a bool array, where True if input element is real.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>isrealobj(x)</td>

<td>Return True if x is a not complex type or an array of complex numbers.</td>

<td></td>

</tr>

<tr>

<td>isscalar(num)</td>

<td>Returns True if the type of num is a scalar type.</td>

<td></td>

</tr>

</tbody>

</table>

#### Logical Operations

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>logical_and(x1, x2)</td>

<td>Compute the truth value of x1 AND x2 element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>logical_or(x1, x2)</td>

<td>Compute the truth value of x1 OR x2 element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>logical_not(x)</td>

<td>Compute the truth value of NOT x element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>logical_xor(x1, x2)</td>

<td>Compute the truth value of x1 XOR x2 element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

</tbody>

</table>

#### Comparison

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>allclose(a, b[, rtol, atol, equal_nan])</td>

<td>Returns True if two arrays are element-wise equal within a tolerance.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>isclose(a, b[, rtol, atol, equal_nan])</td>

<td>Returns a boolean array where two arrays are element-wise equal within a tolerance.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>array_equal(a1, a2)</td>

<td>True if two arrays have the same shape and elements, False otherwise.</td>

<td></td>

</tr>

<tr>

<td>array_equiv(a1, a2)</td>

<td>Returns True if input arrays are shape consistent and all elements equal.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>greater(x1, x2)</td>

<td>Return the truth value of (x1 > x2) element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>greater_equal(x1, x2)</td>

<td>Return the truth value of (x1 >= x2) element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>less(x1, x2)</td>

<td>Return the truth value of (x1 < x2) element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>less_equal(x1, x2)</td>

<td>Return the truth value of (x1 =< x2) element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>equal(x1, x2)</td>

<td>Return (x1 == x2) element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>not_equal(x1, x2)</td>

<td>Return (x1 != x2) element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

</tbody>

</table>

### Mathematical Functions

#### Trigonometric Functions

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>sin(x)</td>

<td>Trigonometric sine, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>cos(x)</td>

<td>Cosine element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>tan(x)</td>

<td>Compute tangent element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>arcsin(x)</td>

<td>Inverse sine, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>arccos(x)</td>

<td>Trigonometric inverse cosine, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>arctan(x)</td>

<td>Trigonometric inverse tangent, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>hypot(x1, x2)</td>

<td>Given the “legs” of a right triangle, return its hypotenuse.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>arctan2(x1, x2)</td>

<td>Element-wise arc tangent of x1/x2 choosing the quadrant correctly.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>degrees(x, /[, out, where, casting, order, ...])</td>

<td>Convert angles from radians to degrees.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>radians(x, /[, out, where, casting, order, ...])</td>

<td>Convert angles from degrees to radians.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>unwrap(p[, discont, axis])</td>

<td>Unwrap by changing deltas between values to 2*pi complement.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>deg2rad(x, /[, out, where, casting, order, ...])</td>

<td>Convert angles from degrees to radians.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>rad2deg(x, /[, out, where, casting, order, ...])</td>

<td>Convert angles from radians to degrees.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Hyperbolic Functions

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>sinh(x)</td>

<td>Hyperbolic sine, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>cosh(x)</td>

<td>Hyperbolic cosine, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>tanh(x)</td>

<td>Compute hyperbolic tangent element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>arcsinh(x)</td>

<td>Inverse hyperbolic sine element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>arccosh(x)</td>

<td>Inverse hyperbolic cosine, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>arctanh(x)</td>

<td>Inverse hyperbolic tangent element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

</tbody>

</table>

#### Rounding

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>around(a[, decimals, out])</td>

<td>Evenly round to the given number of decimals.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>round_(a[, decimals, out])</td>

<td>Evenly round to the given number of decimals.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>rint(x, /[, out, where, casting, order, ...])</td>

<td>Round elements of the array to the nearest integer.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fix(x[, out])</td>

<td>Round to nearest integer towards zero.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>floor(x)</td>

<td>Return the floor of the input, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>ceil(x)</td>

<td>Return the ceiling of the input, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>trunc(x)</td>

<td>Return the truncated value of the input, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

</tbody>

</table>

#### Sums, Products, Differences

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>prod(a, axis=None)</td>

<td>Return the product of array elements over a given axis.</td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>sum(a, axis=None)</td>

<td>Sum of array elements over a given axis.</td>

<td>parameters "dtype", "out", and "keepdims" are not supported</td>

</tr>

<tr>

<td>nanprod(a[, axis, dtype, out, keepdims])</td>

<td>Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nansum(a[, axis, dtype, out, keepdims])</td>

<td>Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>cumprod(a[, axis, dtype, out])</td>

<td>Return the cumulative product of elements along a given axis.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>cumsum(a, axis=None)</td>

<td>Return the cumulative sum of the elements along a given axis.</td>

<td>parameters "dtype" and "out" are not supported</td>

</tr>

<tr>

<td>nancumprod(a[, axis, dtype, out])</td>

<td>Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nancumsum(a[, axis, dtype, out])</td>

<td>Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>diff(a, n=1, axis=-1)</td>

<td>Calculate the n-th discrete difference along given axis.</td>

<td></td>

</tr>

<tr>

<td>ediff1d(ary[, to_end, to_begin])</td>

<td>The differences between consecutive elements of an array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>gradient(f, *varargs, **kwargs)</td>

<td>Return the gradient of an N-dimensional array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>cross(a, b[, axisa, axisb, axisc, axis])</td>

<td>Return the cross product of two (arrays of) vectors.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>trapz(y[, x, dx, axis])</td>

<td>Integrate along the given axis using the composite trapezoidal rule.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Exponents and Logarithms

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>exp(x)</td>

<td>Calculate the exponential of all elements in the input array.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>expm1(x)</td>

<td>Calculate exp(x) - 1 for all elements in the array.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>exp2(x)</td>

<td>Calculate 2**p for all p in the input array.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>log(x)</td>

<td>Natural logarithm, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>log10(x)</td>

<td>Return the base 10 logarithm of the input array, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>log2(x)</td>

<td>Base-2 logarithm of x.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>log1p(x)</td>

<td>Return the natural logarithm of one plus the input array, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>logaddexp(x1, x2, /[, out, where, casting, ...])</td>

<td>Logarithm of the sum of exponentiations of the inputs.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>logaddexp2(x1, x2, /[, out, where, casting, ...])</td>

<td>Logarithm of the sum of exponentiations of the inputs in base-2.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Other Special Functions (not implemented)

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>i0(x)</td>

<td>Modified Bessel function of the first kind, order 0.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>sinc(x)</td>

<td>Return the sinc function.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Floating Point Routines (not implemented)

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>signbit(x, /[, out, where, casting, order, ...])</td>

<td>Returns element-wise True where signbit is set (less than zero).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>copysign(x1, x2, /[, out, where, casting, ...])</td>

<td>Change the sign of x1 to that of x2, element-wise.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>frexp(x[, out1, out2], / [[, out, where, ...])</td>

<td>Decompose the elements of x into mantissa and twos exponent.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ldexp(x1, x2, /[, out, where, casting, ...])</td>

<td>Returns x1 * 2**x2, element-wise.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nextafter(x1, x2, /[, out, where, casting, ...])</td>

<td>Return the next floating-point value after x1 towards x2, element-wise.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>spacing(x, /[, out, where, casting, order, ...])</td>

<td>Return the distance between x and the nearest adjacent number.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Arithmetic Operations

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>add(x1, x2)</td>

<td>Add arguments element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>reciprocal(x)</td>

<td>Return the reciprocal of the argument, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>negative(x)</td>

<td>Numerical negative, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>multiply(x1, x2)</td>

<td>Multiply arguments element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>divide(x1, x2)</td>

<td>Divide arguments element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>power(x1, x2)</td>

<td>First array elements raised to powers from second array, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>subtract(x1, x2)</td>

<td>Subtract arguments, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>true_divide(x1, x2)</td>

<td>Returns a true division of the inputs, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>floor_divide(x1, x2)</td>

<td>Return the largest integer smaller or equal to the division of the inputs.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>float_power(x1, x2, /[, out, where, ...])</td>

<td>First array elements raised to powers from second array, element-wise.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fmod(x1, x2, /[, out, where, casting, ...])</td>

<td>Return the element-wise remainder of division.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>mod(x1, x2)</td>

<td>Return the element-wise remainder of division.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>modf(x[, out1, out2], / [[, out, where, ...])</td>

<td>Return the fractional and integral parts of an array, element-wise.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>remainder(x1, x2)</td>

<td>Return element-wise remainder of division.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>divmod(x1, x2[, out1, out2], / [[, out, ...])</td>

<td>Return element-wise quotient and remainder simultaneously.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Handling Complex Numbers

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>angle(z)</td>

<td>Return the angle of the complex argument.</td>

<td>parameter "deg" is not supported</td>

</tr>

<tr>

<td>real(val)</td>

<td>Return the real part of the complex argument.</td>

<td></td>

</tr>

<tr>

<td>imag(val)</td>

<td>Return the imaginary part of the complex argument.</td>

<td></td>

</tr>

<tr>

<td>conj(x)</td>

<td>Return the complex conjugate, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

</tbody>

</table>

#### Miscellaneous

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>convolve(a, v[, mode])</td>

<td>Returns the discrete, linear convolution of two one-dimensional sequences.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>convolve(a, v[, mode])</td>

<td>Returns the discrete, linear convolution of two one-dimensional sequences.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>clip(a, a_min, a_max)</td>

<td>Clip (limit) the values in an array.</td>

<td>parameter "out" is not supported</td>

</tr>

<tr>

<td>sqrt(x)</td>

<td>Return the positive square-root of an array, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>cbrt(x)</td>

<td>Return the cube-root of an array, element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>square(x)</td>

<td>Return the element-wise square of the input.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>absolute(x)</td>

<td>Calculate the absolute value element-wise.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>fabs(x, /[, out, where, casting, order, ...])</td>

<td>Compute the absolute values element-wise.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>sign(x)</td>

<td>Returns an element-wise indication of the sign of a number.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>heaviside(x1, x2, /[, out, where, casting, ...])</td>

<td>Compute the Heaviside step function.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>maximum(x1, x2)</td>

<td>Element-wise maximum of array elements.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>minimum(x1, x2)</td>

<td>Element-wise minimum of array elements.</td>

<td>parameters "out", "where", "casting", "order", "dtype", "subok", "signature", "extobj" are not supported</td>

</tr>

<tr>

<td>fmax(x1, x2, /[, out, where, casting, ...])</td>

<td>Element-wise maximum of array elements.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>fmin(x1, x2, /[, out, where, casting, ...])</td>

<td>Element-wise minimum of array elements.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>nan_to_num(x[, copy])</td>

<td>Replace nan with zero and inf with finite numbers.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>real_if_close(a[, tol])</td>

<td>If complex input returns a real array if complex parts are close to zero.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>interp(x, xp, fp[, left, right, period])</td>

<td>One-dimensional linear interpolation.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

### Random Sampling

#### Simple Random Data

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>rand(d0, d1, ..., dn)</td>

<td>Random values in a given shape.</td>

<td></td>

</tr>

<tr>

<td>randn(d0, d1, ..., dn)</td>

<td>Return a sample (or samples) from the “standard normal” distribution.</td>

<td></td>

</tr>

<tr>

<td>randint(low[, high, size, dtype])</td>

<td>Return random integers from low (inclusive) to high (exclusive).</td>

<td></td>

</tr>

<tr>

<td>random_integers(low[, high, size])</td>

<td>Random integers of type np.int between low and high, inclusive.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>random_sample([size])</td>

<td>Return random floats in the half-open interval [0.0, 1.0).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>random([size])</td>

<td>Return random floats in the half-open interval [0.0, 1.0).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>ranf([size])</td>

<td>Return random floats in the half-open interval [0.0, 1.0).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>sample([size])</td>

<td>Return random floats in the half-open interval [0.0, 1.0).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>choice(a[, size, replace, p])</td>

<td>Generates a random sample from a given 1-D array</td>

<td>replace=False and p != None are not supported</td>

</tr>

<tr>

<td>bytes(length)</td>

<td>Return random bytes.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Permutations

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>shuffle(x)</td>

<td>Modify a sequence in-place by shuffling its contents.</td>

<td></td>

</tr>

<tr>

<td>permutation(x)</td>

<td>Randomly permute a sequence, or return a permuted range.</td>

<td></td>

</tr>

</tbody>

</table>

#### Distributions

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>beta(a, b, size=None)</td>

<td>Draw samples from a Beta distribution.</td>

<td></td>

</tr>

<tr>

<td>binomial(n, p[, size])</td>

<td>Draw samples from a binomial distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>chisquare(df, size=None)</td>

<td>Draw samples from a chi-square distribution.</td>

<td></td>

</tr>

<tr>

<td>dirichlet(alpha[, size])</td>

<td>Draw samples from the Dirichlet distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>exponential(scale=1.0, size=None)</td>

<td>Draw samples from an exponential distribution.</td>

<td></td>

</tr>

<tr>

<td>f(dfnum, dfden[, size])</td>

<td>Draw samples from an F distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>gamma(shape, scale=1.0, size=None)</td>

<td>Draw samples from a Gamma distribution.</td>

<td></td>

</tr>

<tr>

<td>geometric(p[, size])</td>

<td>Draw samples from the geometric distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>gumbel([loc, scale, size])</td>

<td>Draw samples from a Gumbel distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>hypergeometric(ngood, nbad, nsample[, size])</td>

<td>Draw samples from a Hypergeometric distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>laplace([loc, scale, size])</td>

<td>Draw samples from the Laplace or double exponential distribution with specified location (or mean) and scale (decay).</td>

<td>function not implemented</td>

</tr>

<tr>

<td>logistic(loc=0.0, scale=1.0, size=None)</td>

<td>Draw samples from a logistic distribution.</td>

<td></td>

</tr>

<tr>

<td>lognormal(mean=0.0, sigma=1.0, size=None)</td>

<td>Draw samples from a log-normal distribution.</td>

<td></td>

</tr>

<tr>

<td>logseries(p[, size])</td>

<td>Draw samples from a logarithmic series distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>multinomial(n, pvals[, size])</td>

<td>Draw samples from a multinomial distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>multivariate_normal(mean, cov[, size, ...)</td>

<td>Draw random samples from a multivariate normal distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>negative_binomial(n, p[, size])</td>

<td>Draw samples from a negative binomial distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>noncentral_chisquare(df, nonc[, size])</td>

<td>Draw samples from a noncentral chi-square distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>noncentral_f(dfnum, dfden, nonc[, size])</td>

<td>Draw samples from the noncentral F distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>normal(loc=0.0, scale=1.0, size=None)</td>

<td>Draw random samples from a normal (Gaussian) distribution.</td>

<td></td>

</tr>

<tr>

<td>pareto(a[, size])</td>

<td>Draw samples from a Pareto II or Lomax distribution with specified shape.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>poisson([lam, size])</td>

<td>Draw samples from a Poisson distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>power(a[, size])</td>

<td>Draws samples in [0, 1] from a power distribution with positive exponent a - 1.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>rayleigh([scale, size])</td>

<td>Draw samples from a Rayleigh distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>standard_cauchy([size])</td>

<td>Draw samples from a standard Cauchy distribution with mode = 0.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>standard_exponential(size=None)</td>

<td>Draw samples from the standard exponential distribution.</td>

<td></td>

</tr>

<tr>

<td>standard_gamma(shape, size=None)</td>

<td>Draw samples from a standard Gamma distribution.</td>

<td></td>

</tr>

<tr>

<td>standard_normal(size=None)</td>

<td>Draw samples from a standard Normal distribution (mean=0, stdev=1).</td>

<td></td>

</tr>

<tr>

<td>standard_t(df[, size])</td>

<td>Draw samples from a standard Student’s t distribution with df degrees of freedom.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>triangular(left, mode, right[, size])</td>

<td>Draw samples from the triangular distribution over the interval [left, right].</td>

<td>function not implemented</td>

</tr>

<tr>

<td>uniform(low=0.0, high=1.0, size=None)</td>

<td>Draw samples from a uniform distribution.</td>

<td></td>

</tr>

<tr>

<td>vonmises(mu, kappa[, size])</td>

<td>Draw samples from a von Mises distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>wald(mean, scale, size=None)</td>

<td>Draw samples from a Wald, or inverse Gaussian, distribution.</td>

<td></td>

</tr>

<tr>

<td>weibull(a[, size])</td>

<td>Draw samples from a Weibull distribution.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>zipf(a[, size])</td>

<td>Draw samples from a Zipf distribution.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

#### Random Generator

<table>

<tbody>

<tr>

<th>function</th>

<th>description</th>

<th>notes</th>

</tr>

<tr>

<td>seed(seed=None)</td>

<td>Seed the generator.</td>

<td></td>

</tr>

<tr>

<td>get_seed()</td>

<td>Returns the current seed of the generator.</td>

<td>function has no counterpart in numpy</td>

</tr>

<tr>

<td>get_state()</td>

<td>Return a tuple representing the internal state of the generator.</td>

<td>function not implemented</td>

</tr>

<tr>

<td>set_state(state)</td>

<td>Set the internal state of the generator from a tuple.</td>

<td>function not implemented</td>

</tr>

</tbody>

</table>

## Limitations and Differences with NumPy

*   Requires Python 3.7 or higher.
*   Arrays may have no more than 4 axes.
*   Cocos provides only a subset of NumPy's functions and methods. In many cases, Cocos does not support all of the parameters of its corresponding NumPy function or method.
*   Trailing singleton dimensions are cut off, e.g. there is no difference between an array with shape (2, 2, 1) and an array with shape (2, 2) in Cocos.
*   Matrix multiplication is not supported for integer types.

## A Note on Hardware Configurations for Multi-GPU Computing
Cocos implements multi-GPU functionality via process-based parallelism (one process per GPU device). 
It is recommended to have at least one physical CPU core per GPU in the system in order to prevent 'starving' the GPUs. 

## License
MIT License

Copyright (c) 2019 Michael Nowotny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
