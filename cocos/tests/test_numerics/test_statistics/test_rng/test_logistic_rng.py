import numpy as np
import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 1000000
test_data = [(5.0, np.sqrt(3.0) / np.pi, n_kolmogorov_smirnov),
            (5.0, 2.0, n_kolmogorov_smirnov),
            (9.0, 3.0, n_kolmogorov_smirnov),
            (9.0, 4.0, n_kolmogorov_smirnov),
            (6.0, 2.0, n_kolmogorov_smirnov),
            (2.0, 1.0, n_kolmogorov_smirnov)]


@pytest.mark.parametrize("loc, scale, n_kolmogorov_smirnov", test_data)
def test_logistic_distribution(loc, scale, n_kolmogorov_smirnov):
    u = cn.random.logistic(loc, scale, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='logistic',
                             args=(loc, scale),
                             verbose=True)

    assert not reject
