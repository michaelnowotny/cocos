import numpy as np
import scipy.stats
from scipy.stats import kstest
import typing as tp

import cocos.numerics as cn


def perform_ks_test(x: tp.Union[np.ndarray, cn.ndarray],
                    alpha: float,
                    distribution,
                    args: tp.Tuple = None,
                    verbose: bool = False) -> bool:
    # SciPy 1.18 rejects args for some distributions given by name, so pass
    # the cumulative distribution function of the frozen distribution instead.
    cdf = getattr(scipy.stats, distribution)(*(args or ())).cdf
    D, p_value = kstest(np.asarray(x), cdf)
    if not np.isfinite(p_value):
        raise ValueError(f"The KS test of the {distribution} distribution with "
                         f"parameters {args} did not produce a p-value.")
    if verbose:
        print(f"D={D}, p-value={p_value}")

    reject = p_value < alpha
    if verbose:
        print(f"D={D}, p-value={p_value}")
        print(f"reject = {reject}")

    return reject
