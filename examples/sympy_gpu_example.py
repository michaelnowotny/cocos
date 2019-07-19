import cocos.numerics as cn
import cocos.device as cd
import cocos.symbolic as cs
import numpy as np
import sympy as sym
import time

sym.init_printing()

# define symbolic arguments to the function
x1, x2, x3, t = sym.symbols('x1, x2, x3, t')
argument_symbols = (x1, x2, x3)

# define a function f: R^3 -> R^3
f = sym.Matrix([[x1 + x2], [(x1+x3)**2], [sym.exp(x1 + x2)]])
print("defining the vector valued function f(x1, x2, x3) as")
print(f)
print()

# Compute the Jacobian symbolically
jacobian_f = f.jacobian([x1, x2, x3])
print("symbolically derived Jacobian:")
print(jacobian_f)
print()

# Convert the symbolic array expression to an object that can evaluated
# numerically on the cpu or gpu.
jacobian_f_lambdified \
    = cs.LambdifiedVectorExpression(argument_symbols=argument_symbols,
                                    time_symbol=t,
                                    symbolic_vector_expression=jacobian_f)

# Define a 3 dimensional vector X = (x1, x2, x3) = (1, 2, 3)
X_gpu = cn.array([[1], [2], [3]])

# Numerically evaluate the Jacobian at X = (1, 2, 3)
print("numerical Jacobian at X = (1, 2, 3)")
jacobian_f_lambdified.evaluate(X_gpu.transpose(), t=0)

# Compare the performance on cpu and gpu by numerically evaluating the Jacobian
# at n different vectors (x1, x2, x3)
n = 10000000
print(f'evaluating Jacobian at {n} vectors\n')

X_gpu = cn.random.rand(n, 3)
X_cpu = np.array(X_gpu)

tic = time.time()
jacobian_f_numeric_gpu = jacobian_f_lambdified.evaluate(X_gpu, t=0)
cd.sync()
time_gpu = time.time() - tic
print(f'time on gpu: {time_gpu}')

tic = time.time()
jacobian_f_numeric_cpu = jacobian_f_lambdified.evaluate(X_cpu, t=0)
time_cpu = time.time() - tic
print(f'time on cpu: {time_cpu}')

print(f'speedup on gpu vs cpu: {time_cpu / time_gpu}\n')

# Verify that the results match
print(f'numerical results from cpu and gpu match: '
      f'{np.allclose(jacobian_f_numeric_gpu, jacobian_f_numeric_cpu)}')
