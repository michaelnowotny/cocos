import cocos.device
import cocos.numerics as cn
import numpy as np

from cocos.options import GPUOptions


def test_dot():
    cocos.device.init()

    A_numpy = np.array([[1.0, 2], [3, 4.0]], dtype=np.float32)
    B_numpy = np.array([[5, 6], [7, 8]], dtype=np.float32)
    c_numpy = np.array([[5], [7]], dtype=np.float32)
    d_numpy = c_numpy.transpose()
    e_numpy = np.array([[5, 7]], dtype=np.float32)
    f_numpy = np.array([8, 1], dtype=np.float32)

    # using numpy
    result_1 = np.dot(A_numpy, B_numpy)
    result_2 = np.dot(A_numpy, c_numpy)
    result_3 = np.dot(d_numpy, A_numpy)
    result_4 = np.dot(e_numpy, c_numpy)
    result_5 = np.dot(f_numpy, f_numpy)

    print("numpy result 1")
    print(result_1)

    print("numpy result 2")
    print(result_2)
    print(f"numpy result 2 shape = {result_2.shape}")

    print("numpy result 3")
    print(result_3)

    print("numpy result 4")
    print(result_4)

    print("numpy result 5")
    print(result_5)

    # using Cocos
    A_cocos = cn.array(A_numpy)
    B_cocos = cn.array(B_numpy)
    c_cocos = cn.array(c_numpy)
    d_cocos = cn.array(d_numpy)
    e_cocos = cn.array(e_numpy)
    f_cocos = cn.array(f_numpy)

    print(f"c_cocos.ndim = {c_cocos.ndim}")
    print(f"d_cocos.ndim = {d_cocos.ndim}")
    print(f"e_cocos.ndim = {e_cocos.ndim}")
    print(f"f_cocos.ndim = {f_cocos.ndim}")

    result_1_cocos = cn.dot(A_cocos, B_cocos)
    result_2_cocos = cn.dot(A_cocos, c_cocos)
    result_3_cocos = cn.dot(d_cocos, A_cocos)
    result_4_cocos = cn.dot(e_cocos, c_cocos)
    result_5_cocos = cn.dot(f_cocos, f_cocos)

    print("cocos result 1")
    print(result_1_cocos)

    print("cocos result 2")
    print(result_2_cocos)
    print(f"numpy result 2 shape = {result_2_cocos.shape}")

    print("cocos result 2 reshaped")
    print(np.reshape(result_2_cocos, (2, 1)))

    print("cocos result 3")
    print(result_3_cocos)

    print("cocos result 4")
    print(result_4_cocos)

    print("cocos result 5")
    print(result_5_cocos)

    # conduct tests
    assert np.allclose(result_1, result_1_cocos)
    assert np.allclose(result_3, result_3_cocos)
    assert np.allclose(result_4, result_4_cocos)
    assert np.allclose(result_5, result_5_cocos)
    if GPUOptions.use_gpu:
        assert np.allclose(result_2.flatten(), result_2_cocos)
    else:
        assert np.allclose(result_2.flatten(), result_2_cocos.flatten())
