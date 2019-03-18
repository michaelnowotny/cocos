import array as pyarray
import numpy as np
import pytest

import cocos.device
import cocos.numerics as cn


# backend = 'cpu'
# backend = 'cuda'
# backend = 'opencl'
backend = None


def test_concatenate_hstack_v_stack_dstack():
    cocos.device.init(backend)
    cocos.device.info()

    # define data type
    dtype = np.int32

    # using numpy
    first_array_numpy = np.array([[1, 2], [3, 4]], dtype=dtype)
    second_array_numpy = np.array([[5, 6], [7, 8]], dtype=dtype)
    third_array_numpy = np.array([[9, 10], [11, 12]], dtype=dtype)
    h_stacked_array_numpy = np.hstack((first_array_numpy,
                                       second_array_numpy,
                                       third_array_numpy))

    v_stacked_array_numpy = np.vstack((first_array_numpy,
                                       second_array_numpy,
                                       third_array_numpy))

    d_stacked_array_numpy = np.dstack((first_array_numpy,
                                       second_array_numpy,
                                       third_array_numpy))
    # print(d_stacked_array_numpy)

    # using Cocos
    first_array_cocos = cn.array(first_array_numpy)
    second_array_cocos = cn.array(second_array_numpy)
    third_array_cocos = cn.array(third_array_numpy)
    h_stacked_array_cocos = cn.hstack((first_array_cocos,
                                       second_array_cocos,
                                       third_array_cocos))

    v_stacked_array_cocos = cn.vstack((first_array_cocos,
                                       second_array_cocos,
                                       third_array_cocos))

    d_stacked_array_cocos = cn.dstack((first_array_cocos,
                                       second_array_cocos,
                                       third_array_cocos))
    # print(d_stacked_array_cocos)

    assert np.allclose(h_stacked_array_cocos, h_stacked_array_numpy)
    assert np.allclose(v_stacked_array_cocos, v_stacked_array_numpy)
    assert np.allclose(d_stacked_array_cocos, d_stacked_array_numpy)

    # tests concatenate
    for axis in range(2):
        concatenated_array = np.concatenate((first_array_numpy,
                                             second_array_numpy,
                                             third_array_numpy),
                                            axis=axis)

        concatenated_array_cocos = cn.concatenate((first_array_cocos,
                                                   second_array_cocos,
                                                   third_array_cocos),
                                                  axis=axis)

        assert np.allclose(concatenated_array_cocos, concatenated_array)
