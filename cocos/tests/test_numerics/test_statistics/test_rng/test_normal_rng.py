import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 5000000
test_data = [( 0.0, 0.2, n_kolmogorov_smirnov),
             ( 0.0, 1.0, n_kolmogorov_smirnov),
             ( 0.0, 5.0, n_kolmogorov_smirnov),
             (-2.0, 0.5, n_kolmogorov_smirnov)]


@pytest.mark.parametrize("mu, sigma, n_kolmogorov_smirnov", test_data)
def test_normal_distribution(mu, sigma, n_kolmogorov_smirnov):
    u = cn.random.normal(mu, sigma, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='norm',
                             args=(mu, sigma),
                             verbose=True)

    assert not reject
