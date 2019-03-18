import cocos.device
import cocos.numerics as cn
import numpy as np


def test_diff():
    cocos.device.init()

    if cocos.device.is_dbl_supported():
        A_numpy = np.random.randn(1000, 1000)
        A_cocos = cn.array(A_numpy)

        # # using numpy
        # mean_numpy = np.mean(A_numpy)
        #
        # # using Cocos

        # mean_cocos = cn.mean(A_cocos)

        # conduct tests

        # tests sum
        for n in range(1, 10):
            assert np.allclose(np.diff(A_numpy, n=n),
                               cn.diff(A_cocos, n=n))

            assert np.allclose(np.diff(A_numpy, n=n, axis=0),
                               cn.diff(A_cocos, n=n, axis=0))

            assert np.allclose(np.diff(A_numpy, n=n, axis=1),
                               cn.diff(A_cocos, n=n, axis=1))
