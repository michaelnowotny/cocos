import cocos.numerics as cn
from cocos.numerics.numerical_package_selector import \
    select_num_pack_by_dtype_from_iterable

import cocos.device as cd

from cocos.symbolic import (
    LambdifiedMatrixExpression, \
    find_length_of_state_vectors
)
import numpy as np
import sympy as sym
import time

sym.init_printing()

# define symbolic arguments to the function
x1, x2, x3, t = sym.symbols('x1, x2, x3, t')
argument_symbols = (x1, x2, x3)

# define a function f: R^3 -> R^3
g = sym.Function('g')
f = sym.Matrix([[x1 + x2], [(g(t) * x1 + x3) ** 2], [sym.exp(x1 + x2 + g(t))]])
print("defining the vector valued function f(x1, x2, x3) as")
print(f)
print()

# Compute the Jacobian symbolically
jacobian_f = f.jacobian([x1, x2, x3])
print("symbolically derived Jacobian:")
print(jacobian_f)
print()


def numeric_time_function(t: float):
    """
    Numeric time function

    Args:
        t: (todo): write your description
    """
    return np.log(t)


def jacobian_direct(t, x1, x2, x3):
    """
    R compute jacobian function.

    Args:
        t: (array): write your description
        x1: (array): write your description
        x2: (array): write your description
        x3: (array): write your description
    """
    g = numeric_time_function
    state_vectors = (x1, x2, x3)
    R = find_length_of_state_vectors(state_vectors)
    num_pack = select_num_pack_by_dtype_from_iterable(state_vectors)
    result = num_pack.zeros((R, 3, 3))

    result[:, 0, 0] = 1.0
    result[:, 0, 1] = 1.0
    result[:, 0, 2] = 0.0

    result[:, 1, 0] = 2.0 * g(t) * (g(t) * x1 + x3)
    result[:, 1, 1] = 0.0
    result[:, 1, 2] = 2.0 * (g(t) * x1 + x3)

    result[:, 2, 0] = num_pack.exp(x1 + x2 + g(t))
    result[:, 2, 1] = num_pack.exp(x1 + x2 + g(t))
    result[:, 2, 2] = 0.0

    return result


# Convert the symbolic array expression to an object that can evaluated
# numerically on the cpu or gpu.
jacobian_f_lambdified \
    = LambdifiedMatrixExpression(
        argument_symbols=argument_symbols,
        time_symbol=t,
        symbolic_matrix_expression=jacobian_f,
        symbolic_time_function_name_to_numeric_time_function_map={'g': numeric_time_function})

# Define a 3 dimensional vector X = (x1, x2, x3) = (1, 2, 3)
X_gpu = cn.array([[1], [2], [3]])

# Numerically evaluate the Jacobian at X = (1, 2, 3)
print("numerical Jacobian at X = (1, 2, 3)")
print(jacobian_f_lambdified.evaluate(X_gpu.transpose(), t=1.0))
print(jacobian_direct(t=1.0, x1=1.0, x2=2.0, x3=3.0))

# Compare the performance on cpu and gpu by numerically evaluating the Jacobian
# at n different vectors (x1, x2, x3)
n = 10000000
print(f'evaluating Jacobian at {n} vectors\n')

X_gpu = cn.random.rand(n, 3)
X_cpu = np.array(X_gpu)

# evaluate on gpu
tic = time.time()
# jacobian_f_numeric_gpu = jacobian_f_lambdified.evaluate(X_gpu, t=1.0)
# jacobian_f_numeric_gpu = \
#     (jacobian_f_lambdified
#      .evaluate_with_dictionary(
#         symbolic_to_numeric_parameter_map={x1: X_gpu[:, 0],
#                                            x2: X_gpu[:, 1],
#                                            x3: X_gpu[:, 2]},
#         t=1.0))
jacobian_f_numeric_gpu = \
    (jacobian_f_lambdified
     .evaluate_with_kwargs(x1=X_gpu[:, 0],
                           x2=X_gpu[:, 1],
                           x3=X_gpu[:, 2],
                           t=1.0))
cd.sync()
time_gpu = time.time() - tic
print(f'time on gpu: {time_gpu}')

tic = time.time()
jacobian_f_numeric_gpu_direct = \
    jacobian_direct(t=1.0,
                    x1=X_gpu[:, 0],
                    x2=X_gpu[:, 1],
                    x3=X_gpu[:, 2])
cd.sync()
time_gpu_direct = time.time() - tic
print(f'time for direct computation on gpu: {time_gpu_direct}')

print(f'numerical results from gpu match results from direct computation: '
      f'{np.allclose(jacobian_f_numeric_gpu_direct, jacobian_f_numeric_gpu)}')

# evaluate on cpu
tic = time.time()
# jacobian_f_numeric_cpu = jacobian_f_lambdified.evaluate(X_cpu, t=1.0)
# jacobian_f_numeric_cpu = \
#     (jacobian_f_lambdified
#      .evaluate_with_dictionary(
#         symbolic_to_numeric_parameter_map={x1: X_cpu[:, 0],
#                                            x2: X_cpu[:, 1],
#                                            x3: X_cpu[:, 2]},
#         t=1.0))
jacobian_f_numeric_cpu = \
    (jacobian_f_lambdified
     .evaluate_with_kwargs(x1=X_cpu[:, 0],
                           x2=X_cpu[:, 1],
                           x3=X_cpu[:, 2],
                           t=1.0))
time_cpu = time.time() - tic
print(f'time on cpu: {time_cpu}')

tic = time.time()
jacobian_f_numeric_cpu_direct = \
    jacobian_direct(t=1.0,
                    x1=X_cpu[:, 0],
                    x2=X_cpu[:, 1],
                    x3=X_cpu[:, 2])
time_cpu_direct = time.time() - tic
print(f'time for direct computation on cpu: {time_cpu_direct}')

print(f'numerical results from cpu match results from direct computation: '
      f'{np.allclose(jacobian_f_numeric_cpu_direct, jacobian_f_numeric_cpu)}')

# output performance gain on gpu
print(f'speedup on gpu vs cpu: {time_cpu / time_gpu}\n')

# Verify that the results match
print(f'numerical results from cpu and gpu match: '
      f'{np.allclose(jacobian_f_numeric_gpu, jacobian_f_numeric_cpu)}')
