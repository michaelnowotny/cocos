import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [(np.array([1, 5, 3, 2],
                       dtype=np.int32),
              np.array([50, 200, -5, 10],
                       dtype=np.int32)),
             (np.array([0.5, -1.0, 2.0, 0.0, 3.5],
                       dtype=np.float32),
              np.array([1.0, 2.0, 3.0, 4.0, 5.0],
                       dtype=np.float32))]


@pytest.mark.parametrize("keys_numpy, values_numpy", test_data)
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_by_keys(keys_numpy, values_numpy, ascending):
    cocos.device.init()

    keys_cocos = cn.array(keys_numpy)
    values_cocos = cn.array(values_numpy)

    order = np.argsort(keys_numpy)
    if not ascending:
        order = order[::-1]

    out_keys_cocos, out_values_cocos \
        = cn.sort_by_keys(keys_cocos, values_cocos, ascending=ascending)

    assert np.allclose(np.array(out_keys_cocos), keys_numpy[order])
    assert np.allclose(np.array(out_values_cocos), values_numpy[order])


def test_sort_by_keys_of_flattened_arrays():
    cocos.device.init()
    keys = np.array([[4, 1, 7], [3, 9, 2]], dtype=np.int32)
    values = 10 * keys

    out_keys, out_values = cn.sort_by_keys(cn.array(keys),
                                           cn.array(values),
                                           axis=None)

    assert np.array_equal(np.array(out_keys), np.sort(keys, axis=None))
    assert np.array_equal(np.array(out_values), 10 * np.sort(keys, axis=None))
