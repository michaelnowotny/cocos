import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy
import time
import typing as tp

from cocos.device import (
    ComputeDeviceManager,
    info,
    sync
)

from cocos.multi_processing.device_pool import ComputeDevicePool
from cocos.multi_processing.multi_core_batch_processing import map_reduce_multicore
from cocos.numerics.numerical_package_bundle import (
    NumericalPackageBundle,
    CocosBundle
)

from cocos.numerics.random import randn_antithetic


def simulate_heston_model(
        T: float,
        N: int,
        R: int,
        mu: float,
        kappa: float,
        v_bar: float,
        sigma_v: float,
        rho: float,
        x0: float,
        v0: float,
        numerical_package_bundle: tp.Type[NumericalPackageBundle]) \
        -> tp.Tuple:
    """
    This function simulates R paths from the Heston stochastic volatility model
    over a time horizon of length T divided into N steps.

    :param T: time horizon of the simulation
    :param N: number of steps
    :param R: number of paths to simulate
    :param mu: expected return
    :param kappa: mean-reversion speed of volatility
    :param v_bar: long-run mean of volatility
    :param sigma_v: volatility of volatility
    :param rho: instantaneous correlation of shocks to price and to volatility
    :param x0: initial log price
    :param v0: initial volatility
    :param gpu: whether to compute on the GPU
    :return: a tuple of two R-dimensional numeric arrays for log price and
             volatility
    """
    np = numerical_package_bundle.module()
    random = numerical_package_bundle.random_module()

    Delta_t = T / float(N - 1)

    x = [np.full((R,), x0, dtype=numpy.float32),
         np.zeros((R,), dtype=numpy.float32)]

    v = [np.full((R,), v0, dtype=numpy.float32),
         np.zeros((R,), dtype=numpy.float32)]

    sqrt_delta_t = math.sqrt(Delta_t)
    sqrt_one_minus_rho_square = math.sqrt(1 - rho ** 2)

    m = np.zeros((2,), dtype=numpy.float32)
    m[0] = rho
    m[1] = sqrt_one_minus_rho_square

    t_current = 0
    for t in range(1, N):
        t_previous = (t + 1) % 2
        t_current = t % 2

        # generate antithetic standard normal random variables
        dBt = randn_antithetic(shape=(R, 2),
                               antithetic_dimension=0,
                               num_pack=np,
                               dtype=numpy.float32) * sqrt_delta_t

        sqrt_v_lag = np.sqrt(v[t_previous])
        x[t_current] = x[t_previous] \
                       + (mu - 0.5 * v[t_previous]) * Delta_t \
                       + np.multiply(sqrt_v_lag, dBt[:, 0])
        v[t_current] = v[t_previous] \
                       + kappa * (v_bar - v[t_previous]) * Delta_t \
                       + sigma_v * np.multiply(sqrt_v_lag, np.dot(dBt, m))
        v[t_current] = np.maximum(v[t_current], numpy.finfo(numpy.float32).eps)

    x = x[t_current]
    v = np.maximum(v[t_current], numpy.finfo(numpy.float32).eps)

    return x, v


def compute_option_price_from_simulated_paths(
        r: float,
        T: float,
        K: float,
        x_simulated,
        numerical_package_bundle: tp.Type[NumericalPackageBundle]) -> float:
    """
    Compute the function of a plain-vanilla call option from simulated
    log-returns.

    :param r: the risk-free rate
    :param T: the time to expiration
    :param K: the strike price
    :param x_simulated: a numeric array of simulated log prices of the underlying
    :param numerical_package_bundle: a class implementing NumericalPackageBundle
    :return: option price
    """

    num_pack = numerical_package_bundle.module()

    return float(math.exp(-r * T) * num_pack.mean(num_pack.maximum(num_pack.exp(x_simulated) - K, 0)))


def simulate_and_compute_option_price(
        x0: float,
        v0: float,
        r: float,
        rho: float,
        sigma_v: float,
        kappa: float,
        v_bar: float,
        T: float,
        K: float,
        nT: int,
        R: int,
        numerical_package_bundle: tp.Type[NumericalPackageBundle],
        verbose: bool = False) -> float:

    if verbose:
        print(f'computing on device={ComputeDeviceManager.get_current_compute_device_id()}')

    # simulate random paths
    x_simulated, v_simulated \
        = simulate_heston_model(
            T=T,
            N=nT,
            R=R,
            mu=r,
            kappa=kappa,
            v_bar=v_bar,
            sigma_v=sigma_v,
            rho=rho,
            x0=x0,
            v0=v0,
            numerical_package_bundle=numerical_package_bundle)

    # compute option price
    option_price \
        = compute_option_price_from_simulated_paths(
            r=r,
            T=T,
            K=K,
            x_simulated=x_simulated,
            numerical_package_bundle=numerical_package_bundle)

    return option_price


def simulate_and_compute_option_price_multicore(
        x0: float,
        v0: float,
        r: float,
        rho: float,
        sigma_v: float,
        kappa: float,
        v_bar: float,
        T: float,
        K: float,
        nT: int,
        R: int,
        numerical_package_bundle: tp.Type[NumericalPackageBundle],
        number_of_cores: tp.Optional[int] = None) -> float:
    if number_of_cores is None:
        # number_of_cores = 5 * multiprocessing.cpu_count()
        number_of_cores = 5 * multiprocessing.cpu_count()

    kwargs = \
        dict(x0=x0,
             v0=v0,
             r=r,
             rho=rho,
             sigma_v=sigma_v,
             kappa=kappa,
             v_bar=v_bar,
             T=T,
             K=K,
             nT=nT,
             numerical_package_bundle=numerical_package_bundle)

    # actual simulation run to price plain vanilla call option
    if number_of_cores == 1:
        kwargs['R'] = R
        option_price = \
            simulate_and_compute_option_price(
                **kwargs)

    else:
        kwargs['R'] = math.ceil(R / number_of_cores)

        option_price = \
            map_reduce_multicore(f=simulate_and_compute_option_price,
                                 reduction=lambda x, y: x + y / number_of_cores,
                                 initial_value=0.0,
                                 kwargs_list=number_of_cores * [kwargs])

    return option_price


def simulate_and_compute_option_price_gpu(
        x0: float,
        v0: float,
        r: float,
        rho: float,
        sigma_v: float,
        kappa: float,
        v_bar: float,
        T: float,
        K: float,
        nT: int,
        R: int,
        compute_device_pool: tp.Optional[ComputeDevicePool] = None,
        number_of_batches: tp.Optional[int] = None) -> float:
    numerical_package_bundle = CocosBundle
    if number_of_batches is None:
        if compute_device_pool is None:
            number_of_batches = 1
        else:
            number_of_batches = compute_device_pool.number_of_devices

    kwargs = \
        dict(x0=x0,
             v0=v0,
             r=r,
             rho=rho,
             sigma_v=sigma_v,
             kappa=kappa,
             v_bar=v_bar,
             T=T,
             K=K,
             nT=nT,
             numerical_package_bundle=numerical_package_bundle)

    if compute_device_pool is None:
        kwargs['R'] = R
        print(f'computing {kwargs["R"]} paths on single GPU')

        option_price = \
            simulate_and_compute_option_price(
                **kwargs)

    else:
        kwargs['R'] = math.ceil(R / number_of_batches)
        print(f'computing {R} paths on {compute_device_pool.number_of_devices} GPUs in '
              f'{number_of_batches} batches of {kwargs["R"]} paths')

        option_price = \
            compute_device_pool.map_reduce(f=simulate_and_compute_option_price,
                                           reduction=lambda x, y: x + y / number_of_batches,
                                           initial_value=0.0,
                                           kwargs_list=number_of_batches * [kwargs])

    return option_price
