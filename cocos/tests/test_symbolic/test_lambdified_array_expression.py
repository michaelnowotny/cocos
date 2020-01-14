import cocos.numerics as cn
from cocos.numerics.numerical_package_selector import \
    select_num_pack_by_dtype_from_iterable

from cocos.numerics.data_types import NumericArray

from cocos.symbolic import \
    LambdifiedVectorExpression, \
    find_length_of_state_vectors

import numpy as np
import sympy as sym


def jacobian_direct(x1, x2, x3) -> NumericArray:
    state_vectors = (x1, x2, x3)
    R = find_length_of_state_vectors(state_vectors)
    num_pack = select_num_pack_by_dtype_from_iterable(state_vectors)
    result = num_pack.zeros((R, 3, 3))

    result[:, 0, 0] = 1.0
    result[:, 0, 1] = 1.0
    result[:, 0, 2] = 0.0

    result[:, 1, 0] = 2.0 * (x1 + x3)
    result[:, 1, 1] = 0.0
    result[:, 1, 2] = 2.0 * (x1 + x3)

    result[:, 2, 0] = num_pack.exp(x1 + x2)
    result[:, 2, 1] = num_pack.exp(x1 + x2)
    result[:, 2, 2] = 0.0

    return result


def test_lambdified_array_expression():
    # define symbolic arguments to the function
    x1, x2, x3, t = sym.symbols('x1, x2, x3, t')
    argument_symbols = (x1, x2, x3)

    # define a function f: R^3 -> R^3
    f = sym.Matrix([[x1 + x2], [(x1 + x3) ** 2], [sym.exp(x1 + x2)]])

    # Compute the Jacobian symbolically
    jacobian_f = f.jacobian([x1, x2, x3])

    # Convert the symbolic array expression to an object that can evaluated
    # numerically on the cpu or gpu.
    jacobian_f_lambdified \
        = LambdifiedVectorExpression(argument_symbols=argument_symbols,
                                     # time_symbol=t,
                                     symbolic_vector_expression=jacobian_f)

    # Compare the results on cpu and gpu with direct evaluation of the jaboian
    # at n different vectors (x1, x2, x3)
    n = 10000
    print(f'evaluating Jacobian at {n} vectors\n')

    X_gpu = cn.random.rand(n, 3)
    X_cpu = np.array(X_gpu)

    # evaluate on GPU
    jacobian_f_numeric_gpu_direct = \
        jacobian_direct(x1=X_gpu[:, 0],
                        x2=X_gpu[:, 1],
                        x3=X_gpu[:, 2])

    jacobian_f_numeric_gpu_using_matrix = \
        jacobian_f_lambdified.evaluate(X_gpu, t=0)

    assert np.allclose(jacobian_f_numeric_gpu_direct,
                       jacobian_f_numeric_gpu_using_matrix)

    jacobian_f_numeric_gpu_using_kwargs = \
        (jacobian_f_lambdified
         .evaluate_with_kwargs(x1=X_gpu[:, 0],
                               x2=X_gpu[:, 1],
                               x3=X_gpu[:, 2],
                               # t=0
                               )
         )
    assert np.allclose(jacobian_f_numeric_gpu_direct,
                       jacobian_f_numeric_gpu_using_kwargs)

    jacobian_f_numeric_gpu_using_dictionary = \
        (jacobian_f_lambdified
         .evaluate_with_dictionary(
            symbolic_to_numeric_parameter_map={x1: X_gpu[:, 0],
                                               x2: X_gpu[:, 1],
                                               x3: X_gpu[:, 2]},
            # t=0
        )
        )
    assert np.allclose(jacobian_f_numeric_gpu_direct,
                       jacobian_f_numeric_gpu_using_dictionary)

    # evaluate on CPU
    jacobian_f_numeric_cpu_direct = \
        jacobian_direct(x1=X_cpu[:, 0],
                        x2=X_cpu[:, 1],
                        x3=X_cpu[:, 2])

    jacobian_f_numeric_cpu_using_matrix = \
        jacobian_f_lambdified.evaluate(X_cpu, t=0)

    assert np.allclose(jacobian_f_numeric_cpu_direct,
                       jacobian_f_numeric_cpu_using_matrix)

    jacobian_f_numeric_cpu_using_kwargs = \
        (jacobian_f_lambdified
         .evaluate_with_kwargs(x1=X_cpu[:, 0],
                               x2=X_cpu[:, 1],
                               x3=X_cpu[:, 2],
                               # t=0
                               )
         )
    assert np.allclose(jacobian_f_numeric_cpu_direct,
                       jacobian_f_numeric_cpu_using_kwargs)

    jacobian_f_numeric_cpu_using_dictionary = \
        (jacobian_f_lambdified
         .evaluate_with_dictionary(
            symbolic_to_numeric_parameter_map={x1: X_cpu[:, 0],
                                               x2: X_cpu[:, 1],
                                               x3: X_cpu[:, 2]},
            # t=0
        )
        )
    assert np.allclose(jacobian_f_numeric_cpu_direct,
                       jacobian_f_numeric_cpu_using_dictionary)

    # Verify that the results on CPU and GPU match
    assert np.allclose(jacobian_f_numeric_cpu_using_matrix,
                       jacobian_f_numeric_cpu_using_matrix)
