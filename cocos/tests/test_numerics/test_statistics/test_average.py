import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [(np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 20]],
                       dtype=np.int32), None),
             (np.array([[0.2, 1.0, 0.5],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.2, 0.25]],
                       dtype=np.float32),
              np.array([0.2, 0.3, 0.5],
                       dtype=np.float32)),
             (np.array([[0.5, 2.3, 3.1],
                        [4, 5.5, 6],
                        [7 - 9j, 8 + 1j, 2 + 10j]],
                       dtype=np.complex64),
              np.array([0.2, 0.3, 0.5],
                       dtype=np.float32))]


@pytest.mark.parametrize("A_numpy, weights", test_data)
def test_average(A_numpy, weights):
    print(A_numpy)
    cocos.device.init()
    print('init')
    A_cocos = cn.array(A_numpy)

    if isinstance(weights, np.ndarray):
        weights_cocos = cn.array(weights)
    else:
        weights_cocos = None

    # conduct tests
    average_numpy = np.average(A_numpy)
    average_cocos = cn.average(A_cocos)
    assert np.allclose(average_numpy, average_cocos)

    average_numpy_axis_0 = np.average(A_numpy, axis=0, weights=weights)
    average_cocos_axis_0 = cn.average(A_cocos, axis=0, weights=weights_cocos)
    truth_value = np.allclose(average_numpy_axis_0, average_cocos_axis_0)
    if not truth_value:
        print("input array")
        print(A_numpy)
        print("input weights")
        print(weights)
        print("output numpy axis 0")
        print(average_numpy_axis_0)
        print("output cocos axis 0")
        print(average_cocos_axis_0)
    assert truth_value

    # ToDo: Taking the average over axis 1 with weights produces results that are identical to the operation without weights
    average_numpy_axis_1 = np.average(A_numpy, axis=1, weights=weights)
    average_cocos_axis_1 = cn.average(A_cocos, axis=1, weights=weights_cocos)
    truth_value = np.allclose(average_numpy_axis_1, average_cocos_axis_1)

    if not truth_value:
        print("input array")
        print(A_numpy)
        print("input weights")
        print(weights)
        print("output numpy axis 0")
        print(average_numpy_axis_0)
        print("output cocos axis 0")
        print(average_cocos_axis_0)

        print("output numpy axis 1")
        print(average_numpy_axis_1)
        print("output cocos axis 1")
        print(average_cocos_axis_1)

        print("output numpy axis 0 no weights")
        print(np.average(A_numpy, axis=0))

        print("output numpy axis 1 no weights")
        print(np.average(A_numpy, axis=1))
    assert truth_value


def test_average_3_axes():
    A_numpy = np.array([[[0.2, 1.0, 0.5],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.2, 0.25]],
                        [[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 20]],
                        [[0.5, 2.3, 3.1],
                         [4, 5.5, 6],
                         [7, 8, 2]]
                        ])

    A_cocos = cn.array(A_numpy)

    weights_numpy = np.array([0.2, 0.3, 0.5],
                             dtype=np.float32)
    weights_cocos = cn.array(weights_numpy)

    for axis in range(3):
        print(f'axis = {axis}')
        average_numpy = np.average(A_numpy, axis=axis, weights=weights_numpy)
        average_cocos \
            = np.array(cn.average(A_cocos, axis=axis, weights=weights_cocos))

        truth_value = np.allclose(average_numpy, average_cocos)
        if not truth_value:
            print('numpy')
            print(average_numpy)

            print('cocos')
            print(average_cocos)
        assert truth_value
