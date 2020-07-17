import math
import numpy
import time
import typing as tp

import cocos.device as cd
from cocos.numerics.numerical_package_selector import \
    get_gpu_and_num_pack_by_dtype

from cocos.numerics.random import randn_antithetic


def simulate_heston_model(T: float,
                          N: int,
                          R: int,
                          mu: float,
                          kappa: float,
                          v_bar: float,
                          sigma_v: float,
                          rho: float,
                          x0: float,
                          v0: float,
                          gpu: bool = False) -> tp.Tuple:
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
    if gpu:
        import cocos.numerics as np
    else:
        import numpy as np

    Delta_t = T / float(N - 1)

    x = [np.full((R,), x0, dtype=numpy.float32),
         np.zeros((R,), dtype=numpy.float32)]

    v = [np.full((R,), v0, dtype=numpy.float32),
         np.zeros((R,), dtype=numpy.float32)]

    sqrt_delta_t = math.sqrt(Delta_t)
    sqrt_one_minus_rho_square = math.sqrt(1 - rho ** 2)

    # m = np.array([[rho, sqrt_one_minus_rho_square]])
    m = np.zeros((2,), dtype=numpy.float32)
    m[0] = rho
    m[1] = sqrt_one_minus_rho_square
    zero_array = np.zeros((R,), dtype=numpy.float32)

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


def compute_option_price(r: float,
                         T: float,
                         K: float,
                         x_simulated,
                         num_pack: tp.Optional = None):
    """
    Compute the function of a plain-vanilla call option from simulated
    log-returns.

    :param r: the risk-free rate
    :param T: the time to expiration
    :param K: the strike price
    :param x_simulated: a numeric array of simulated log prices of the underlying
    :param num_pack: a module - either numpy or cocos.numerics
    :return: option price
    """

    if num_pack is None:
        use_gpu, num_pack = get_gpu_and_num_pack_by_dtype(x_simulated)

    return math.exp(-r * T) \
           * num_pack.mean(num_pack.maximum(num_pack.exp(x_simulated) - K, 0))


def run_benchmark(x0: float,
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
                  use_gpu: bool = True) -> tp.Tuple[float, float, float, float]:
    if use_gpu:
        import cocos.numerics as np
    else:
        import numpy as np

    # number of paths for warm-up (compile GPU kernels)
    R_warm_up = 1000
    x_simulated, v_simulated \
        = simulate_heston_model(T=T,
                                N=nT,
                                R=R_warm_up,
                                mu=r,
                                kappa=kappa,
                                v_bar=v_bar,
                                sigma_v=sigma_v,
                                rho=rho,
                                x0=x0,
                                v0=v0,
                                gpu=use_gpu)

    # actual simulation run to price plain vanilla call option
    tic = time.time()
    (x_simulated, v_simulated) \
        = simulate_heston_model(T=T,
                                N=nT,
                                R=R,
                                mu=r,
                                kappa=kappa,
                                v_bar=v_bar,
                                sigma_v=sigma_v,
                                rho=rho,
                                x0=x0,
                                v0=v0,
                                gpu=use_gpu)

    if use_gpu:
        cd.sync()
    time_in_simulation = time.time() - tic

    tic = time.time()

    # compute option price
    option_price = compute_option_price(r, T, K, x_simulated, num_pack=None)

    time_in_option_price_calculation = time.time() - tic

    total_time = time_in_simulation + time_in_option_price_calculation

    return (time_in_simulation,
            time_in_option_price_calculation,
            total_time,
            option_price)


def main():
    cd.info()

    # model parameters
    x0 = 0.0  # initial log stock price
    v0 = 0.101 ** 2  # initial volatility
    r = math.log(1.0319)  # risk-free rate
    rho = -0.7  # instantaneous correlation between Brownian motions
    sigma_v = 0.61  # variance of volatility
    kappa = 6.21  # mean reversion speed
    v_bar = 0.019  # mean variance

    # option parameters
    T = 1.0  # time to expiration
    K = 0.95  # strike price

    # simulation parameters
    nT = int(math.ceil(500 * T))  # number of time-steps to simulate
    R = 2000000  # actual number of paths to simulate for pricing

    # run benchmark on cpu
    (time_in_simulation_cpu,
     time_in_option_price_calculation_cpu,
     total_time_cpu,
     option_price_cpu) \
        = run_benchmark(x0,
                        v0,
                        r,
                        rho,
                        sigma_v,
                        kappa,
                        v_bar,
                        T,
                        K,
                        nT,
                        R,
                        use_gpu=False)

    print(f"Time in simulation on CPU: {time_in_simulation_cpu} secs")
    print(f"Time in option price calculation on CPU: "
          f"= {time_in_option_price_calculation_cpu} secs")

    print(f"Total time on CPU: "
          f"= {total_time_cpu} secs")
    print(f"Call option price on CPU: {option_price_cpu}")

    # run benchmark on gpu
    (time_in_simulation_gpu,
     time_in_option_price_calculation_gpu,
     total_time_gpu,
     option_price_gpu) \
        = run_benchmark(x0,
                        v0,
                        r,
                        rho,
                        sigma_v,
                        kappa,
                        v_bar,
                        T,
                        K,
                        nT,
                        R,
                        use_gpu=True)

    print(f"Time in simulation on GPU: {time_in_simulation_gpu} secs")
    print(f"Time in option price calculation on GPU: "
          f"= {time_in_option_price_calculation_gpu} secs")

    print(f"Total time on GPU: "
          f"= {total_time_gpu} secs")
    print(f"Call option price on GPU: {option_price_gpu}")

    # compute and print the speedup factor
    print(f'Speedup: {total_time_cpu/ total_time_gpu}')


if __name__ == "__main__":
    main()
