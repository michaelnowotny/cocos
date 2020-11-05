import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 1000000
test_data = [(0, 1, n_kolmogorov_smirnov),
             (2, 5, n_kolmogorov_smirnov),
             (3, 12, n_kolmogorov_smirnov),
             (5, 10, n_kolmogorov_smirnov),
             (1, 2, n_kolmogorov_smirnov),
             (10, 50, n_kolmogorov_smirnov),
             (20, 25, n_kolmogorov_smirnov)]


@pytest.mark.parametrize("a, b, n_kolmogorov_smirnov", test_data)
def test_uniform_distribution(a, b, n_kolmogorov_smirnov):
    """
    Rejects a random variates.

    Args:
        a: (todo): write your description
        b: (todo): write your description
        n_kolmogorov_smirnov: (todo): write your description
    """
    u = cn.random.uniform(a, b, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='uniform',
                             args=(a, b - a))

    assert not reject
