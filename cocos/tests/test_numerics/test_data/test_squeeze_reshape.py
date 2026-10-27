import cocos.device
import cocos.numerics as cn
import numpy as np
import pytest


test_data = [np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 20]],
                      dtype=np.int32),
             np.array([[0.2, 1.0, 0.5],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.2, 0.25]],
                      dtype=np.float32),
             np.array([[0.5, 2.3, 3.1],
                       [4, 5.5, 6],
                       [7 - 9j, 8 + 1j, 2 + 10j]],
                      dtype=np.complex64),
             np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 20]], dtype=np.int32),
             np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 20]],
                      dtype=np.int32)
            ]


@pytest.mark.parametrize("A_numpy", test_data)
def test_squeeze_reshape(A_numpy):
    cocos.device.init()
    newshapes = [(3, 1, 3), (1, 3, 3)]
    axess = [(None, 1), (None, 0)]

    for newshape, axes in zip(newshapes, axess):
        A_cocos = cn.array(A_numpy)

        # 3, 1, 3
        B_numpy = A_numpy.reshape(newshape)
        B_cocos = A_cocos.reshape(newshape)

        assert np.allclose(B_cocos, B_numpy)

        for axis in axes:
            C_numpy = B_numpy.squeeze(axis=axis)
            C_cocos = B_cocos.squeeze(axis=axis)

            assert np.allclose(C_cocos, C_numpy)


def main():
    test_squeeze_reshape(test_data[0])


if __name__ == '__main__':
    main()


def test_reshape_accepts_shape_and_newshape():
    a = np.arange(6, dtype=np.float32).reshape((2, 3))
    a_cocos = cn.array(a)
    expected = np.reshape(a, (3, 2))  # NumPy 2.0 has no shape keyword
    assert np.array_equal(np.asarray(cn.reshape(a_cocos, (3, 2))), expected)
    assert np.array_equal(np.asarray(cn.reshape(a_cocos, shape=(3, 2))), expected)
    assert np.array_equal(np.asarray(cn.reshape(a_cocos, newshape=(3, 2))), expected)
    with pytest.raises(TypeError):
        cn.reshape(a_cocos, (3, 2), newshape=(3, 2))


@pytest.mark.parametrize("shape", [(5, 1), (1, 5), (5, 1, 1), (2, 3, 1)])
def test_trailing_dimensions_of_length_one_are_kept(shape):
    """
    ArrayFire drops trailing dimensions of length one; converting to NumPy
    must restore the shape, so that (n,) * (n, 1) does not broadcast to n x n.
    """
    a = cn.array(np.ones(shape, dtype=np.float32))
    assert np.asarray(a).shape == shape
    assert np.asarray(cn.reshape(cn.array(np.arange(np.prod(shape), dtype=np.float32)), shape)).shape == shape
    if len(shape) <= 2:
        assert np.asarray(cn.random.randn(*shape)).shape == shape
        assert np.asarray(cn.random.rand(*shape)).shape == shape
