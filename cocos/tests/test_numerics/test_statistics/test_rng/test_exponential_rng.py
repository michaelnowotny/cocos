import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 2000000
test_data = [
             (0.5, n_kolmogorov_smirnov),
             (1.0, n_kolmogorov_smirnov),
             (1.5, n_kolmogorov_smirnov),
            ]


@pytest.mark.parametrize("scale, n_kolmogorov_smirnov", test_data)
def test_exponential_distribution(scale, n_kolmogorov_smirnov):
    u = cn.random.exponential(scale, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='expon',
                             args=(0.0, scale),
                             verbose=True)

    assert not reject
