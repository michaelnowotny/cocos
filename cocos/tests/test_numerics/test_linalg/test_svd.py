import cocos.device
import cocos.numerics as cn
import cocos.numerics.linalg as linalg
import numpy as np

# backend = 'cpu'
# backend = 'cuda'
# backend = 'opencl'
backend = None


def compare_cocos_numpy(cocos_array, numpy_array):
    return np.allclose(np.array(cocos_array), numpy_array)


def test_svd():
    cocos.device.init(backend)
    cocos.device.info()

    # selected_backend = device_info['backend'].lower()
    selected_backend = cocos.device.selected_backend()
    # print("backend lowercase = <{}>".format(selected_backend))
    # print("device = {}".format(device_info['device']))
    # print("backend = {}".format(device_info['backend']))
    # print("toolkit = {}".format(device_info['toolkit']))
    # print("compute = {}".format(device_info['compute']))

    dtype = np.float32
    if cocos.device.is_dbl_supported():
        dtype = np.float64

    A = np.array([[3, 2, 2], [2, 3, -2]], dtype=dtype)

    # numpy
    U, s, V = np.linalg.svd(A)
    # print("U numpy")
    # print(U)
    # print("")
    #
    # print("s")
    # print(s)
    # print("")
    #
    # print("V")
    # print(V)
    # print("")

    S = np.zeros((2, 3), dtype=dtype)
    S[:2, :2] = np.diag(s)
    reconstructed_A = np.dot(U, np.dot(S, V))
    assert np.allclose(reconstructed_A, A)

    # Cocos
    A_cocos = cn.array(A)
    U_cocos, s_cocos, V_cocos = cn.linalg.svd(A_cocos)
    print("U cocos")
    print(np.array(U_cocos))

    print("s cocos")
    print(np.array(s_cocos))

    print("V cocos")
    print(np.array(V_cocos))

    # S_cocos = cocos.diag(s_cocos)
    # reconstructed_A_cocos = cocos.dot(cocos.dot(U_cocos, S_cocos), V_cocos)

    S_cocos = cn.zeros((2, 3), dtype=dtype)
    # print("diagonalized cocos s array")
    # print(cocos.diag(s_cocos))
    # print("diag(s_cocos).dtype = {}".format(cocos.diag(s_cocos).dtype))
    S_cocos[0:2, 0:2] = cn.diag(s_cocos)
    first_step = cn.dot(S_cocos, V_cocos)
    second_step = cn.dot(U_cocos, first_step)
    reconstructed_A_cocos = second_step
    assert np.allclose(reconstructed_A_cocos, A_cocos)
    assert compare_cocos_numpy(s_cocos, s)

    if cocos.device.is_dbl_supported() and selected_backend != 'cuda' and selected_backend != 'opencl':
        assert compare_cocos_numpy(U_cocos, U)
        assert compare_cocos_numpy(V_cocos, V)
