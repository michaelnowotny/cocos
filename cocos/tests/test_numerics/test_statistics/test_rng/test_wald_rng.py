import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 1000000
test_data = [
             (1.0, 1.0, n_kolmogorov_smirnov),
             (1.0, 0.2, n_kolmogorov_smirnov),
             (1.0, 3.0, n_kolmogorov_smirnov),
             (3.0, 1.0, n_kolmogorov_smirnov),
             (3.0, 0.2, n_kolmogorov_smirnov)
            ]


@pytest.mark.parametrize("mean, scale, n_kolmogorov_smirnov", test_data)
def test_wald_distribution(mean, scale, n_kolmogorov_smirnov):
    u = cn.random.wald(mean, scale, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='invgauss',
                             args=(mean, scale, 0),
                             verbose=True)

    assert not reject
