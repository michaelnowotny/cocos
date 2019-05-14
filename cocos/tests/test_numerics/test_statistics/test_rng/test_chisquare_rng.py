import pytest

import cocos.numerics as cn
from cocos.tests.test_numerics.test_statistics.utilities import perform_ks_test


n_kolmogorov_smirnov = 2000000
test_data = [(1, n_kolmogorov_smirnov),
             (2, n_kolmogorov_smirnov),
             (3, n_kolmogorov_smirnov),
             (4, n_kolmogorov_smirnov),
             (5, n_kolmogorov_smirnov),
             (6, n_kolmogorov_smirnov),
             (9, n_kolmogorov_smirnov)]


@pytest.mark.parametrize("df, n_kolmogorov_smirnov", test_data)
def test_chisquare_distribution(df, n_kolmogorov_smirnov):
    u = cn.random.chisquare(df, n_kolmogorov_smirnov)
    reject = perform_ks_test(u,
                             alpha=0.01,
                             distribution='chi2',
                             args=(df, 0.0, 0.0),
                             verbose=True)

    assert not reject
