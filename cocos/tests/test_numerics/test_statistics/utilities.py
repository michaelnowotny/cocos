import numpy as np
from scipy.stats import kstest
import typing as tp

import cocos.numerics as cn


def perform_ks_test(x: tp.Union[np.ndarray, cn.ndarray],
                    alpha: float,
                    distribution,
                    args: tp.Tuple = None,
                    verbose: bool = False) -> bool:
    """
    R compute the test.

    Args:
        x: (todo): write your description
        tp: (todo): write your description
        Union: (str): write your description
        np: (todo): write your description
        ndarray: (array): write your description
        cn: (todo): write your description
        ndarray: (array): write your description
        alpha: (float): write your description
        distribution: (str): write your description
        tp: (todo): write your description
        Tuple: (todo): write your description
        verbose: (bool): write your description
    """
    D, p_value = kstest(x, distribution, args=args)
    if verbose:
        print(f"D={D}, p-value={p_value}")

    alpha = 0.01
    reject = p_value < alpha
    if verbose:
        print(f"D={D}, p-value={p_value}")
        print(f"reject = {reject}")

    return reject
