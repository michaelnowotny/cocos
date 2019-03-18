import numpy as np
import pytest

import cocos.numerics as cn
from tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 5000000
test_data = [( 0.0, 0.2, n_kolmogorov_smirnov),
             ( 0.5, 1.0, n_kolmogorov_smirnov),
             ( 0.0, 1.5, n_kolmogorov_smirnov),
             (-1.0, 0.5, n_kolmogorov_smirnov)]


@pytest.mark.parametrize("mu, sigma, n_kolmogorov_smirnov", test_data)
def test_lognormal_distribution(mu, sigma, n_kolmogorov_smirnov):
    u = cn.random.lognormal(mu, sigma, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='lognorm',
                             args=(sigma, 0.0, np.exp(mu)),
                             verbose=True)

    assert not reject
