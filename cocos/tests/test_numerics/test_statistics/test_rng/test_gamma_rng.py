import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 1500000
test_data = [(1, 2, n_kolmogorov_smirnov),
             (2, 2, n_kolmogorov_smirnov),
             (3, 2, n_kolmogorov_smirnov),
             (5, 1, n_kolmogorov_smirnov),
             (9, 0.5, n_kolmogorov_smirnov),
             (7.5, 1, n_kolmogorov_smirnov),
             (0.5, 1, n_kolmogorov_smirnov)]


@pytest.mark.parametrize("a, b, n_kolmogorov_smirnov", test_data)
def test_gamma_distribution(a, b, n_kolmogorov_smirnov):
    u = cn.random.gamma(a, b, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='gamma',
                             args=(a, 0.0, b),
                             verbose=True)

    assert not reject
