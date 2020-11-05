import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 10000
test_data = [(0.5, 0.5, n_kolmogorov_smirnov),
             (5.0, 1.0, n_kolmogorov_smirnov),
             (1.0, 3.0, n_kolmogorov_smirnov),
             (2.0, 2.0, n_kolmogorov_smirnov),
             (2.0, 5.0, n_kolmogorov_smirnov)]


@pytest.mark.parametrize("a, b, n_kolmogorov_smirnov", test_data)
def test_beta_distribution(a, b, n_kolmogorov_smirnov):
    """
    Test the beta of the beta.

    Args:
        a: (array): write your description
        b: (array): write your description
        n_kolmogorov_smirnov: (todo): write your description
    """
    u = cn.random.beta(a, b, n_kolmogorov_smirnov)
    print(u.shape)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='beta',
                             args=(a, b),
                             verbose=True)

    assert not reject
