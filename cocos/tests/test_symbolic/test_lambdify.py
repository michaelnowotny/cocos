import numpy as np
import sympy as sym

import cocos.numerics as cn
import cocos.symbolic as cs
import cocos.symbolic._lambdification


def test_lambdify():
    x = sym.Symbol('x')

    expr = sym.sin(x) + sym.cos(x)

    lambdified_cn = cocos.symbolic._lambdification.lambdify(x, expr)
    lambdified_sym = sym.lambdify(x, expr)

    x_vals = np.linspace(-np.pi, np.pi, 100, dtype=np.float32)
    x_vals_gpu = cn.array(x_vals)
    assert np.allclose(lambdified_sym(x_vals), lambdified_cn(x_vals_gpu))
