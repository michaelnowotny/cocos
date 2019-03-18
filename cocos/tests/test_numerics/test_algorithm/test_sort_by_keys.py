import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [(np.array([1, 5, 3, 2],
                       dtype=np.int32),
              np.array([50, 200, -5, 10],
                       dtype=np.int32))]


@pytest.mark.parametrize("keys_numpy, values_numpy", test_data)
def test_sort_by_keys(keys_numpy, values_numpy):
    cocos.device.init()

    keys_cocos = cn.array(keys_numpy)
    values_cocos = cn.array(values_numpy)

    # out_keys_numpy, out_values_numpy = np.sort_by_keys(keys_numpy, values_numpy, axis=0)
    # out_keys_cocos, out_values_cocos = cn.sort_by_keys(keys_cocos, values_cocos, axis=0)
    #
    # assert np.allclose(out_keys_cocos, out_keys_numpy)
    # assert np.allclose(out_values_cocos, out_values_numpy)
