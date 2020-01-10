from abc import ABC, abstractmethod
import concurrent
from concurrent import futures
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy
import pandas as pd
import time
from types import ModuleType
import typing as tp

import cocos.device
from cocos.numerics.random import randn_antithetic


def loky_installed() -> bool:
    try:
        import loky
        return True
    except:
        return False


class NumericalPackageBundle(ABC):
    @classmethod
    @abstractmethod
    def is_installed(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def label(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def module(cls) -> ModuleType:
        pass

    @classmethod
    @abstractmethod
    def random_module(cls) -> ModuleType:
        pass

    @classmethod
    def synchronize(cls):
        pass


class NumpyBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        try:
            import numpy
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        return 'NumPy'

    @classmethod
    def module(cls) -> ModuleType:
        import numpy
        return numpy

    @classmethod
    def random_module(cls) -> ModuleType:
        import numpy.random
        return numpy.random


class NumpyMulticoreBundle(NumpyBundle):
    @classmethod
    def is_installed(cls) -> bool:
        return super().is_installed() & loky_installed()

    @classmethod
    def label(cls) -> str:
        return 'NumPy Multicore'


class CocosBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        try:
            import cocos
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        return 'Cocos'

    @classmethod
    def module(cls) -> ModuleType:
        import cocos.numerics
        return cocos.numerics

    @classmethod
    def random_module(cls) -> ModuleType:
        import cocos.numerics.random
        return cocos.numerics.random

    @classmethod
    def synchronize(cls):
        from cocos.device import sync
        sync()


class CuPyBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        try:
            import cupy
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        return 'CuPy'

    @classmethod
    def module(cls) -> ModuleType:
        import cupy
        return cupy

    @classmethod
    def random_module(cls) -> ModuleType:
        import cupy.random
        return cupy.random

    # @classmethod
    # def code(cls) -> NumericalPackage:
    #     return NumericalPackage.CUPY

    @classmethod
    def synchronize(cls):
        import cupy
        cupy.cuda.Stream.null.synchronize()


def get_available_numerical_packages(
        list_installed_bundles: tp.Optional[bool] = False) \
        -> tp.Tuple[tp.Type[NumericalPackageBundle], ...]:
    numerical_bundles_to_try = (NumpyBundle,
                                CocosBundle,
                                CuPyBundle)

    available_numerical_bundles \
        = [numerical_bundle
           for numerical_bundle in numerical_bundles_to_try
           if numerical_bundle.is_installed()]

    if list_installed_bundles:
        print(f'Required packages found for the following benchmarks:')
        for numerical_bundle in available_numerical_bundles:
            print(numerical_bundle.label())

        print()

    return tuple(available_numerical_bundles)


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
                               num_pack=np) * sqrt_delta_t

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
        numerical_package_bundle: tp.Type[NumericalPackageBundle]):
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

    return math.exp(-r * T) \
           * num_pack.mean(num_pack.maximum(num_pack.exp(x_simulated) - K, 0))


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
        numerical_package_bundle: tp.Type[NumericalPackageBundle]) -> float:

    # actual simulation run to price plain vanilla call option
    (x_simulated, v_simulated) \
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


ResultType = tp.TypeVar('ResultType')


def map_reduce_multicore(
        f: tp.Callable[..., ResultType],
        reduction: tp.Callable[[ResultType, ResultType], ResultType],
        initial_value: ResultType,
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None) -> ResultType:
    from loky import get_reusable_executor

    if number_of_batches is None:
        if args_list is not None:
            number_of_batches = len(args_list)
        elif kwargs_list is not None:
            number_of_batches = len(kwargs_list)
        else:
            raise ValueError('Number_of_batches must be defined if '
                             'both args_list and kwargs_list are empty')

    if args_list is None:
        args_list = [list() for i in range(number_of_batches)]
    if kwargs_list is None:
        kwargs_list = number_of_batches * [dict()]

    executor = \
        get_reusable_executor(timeout=None,
                              context='loky')

    futures = [executor.submit(f, *args, **kwargs)
               for args, kwargs
               in zip(args_list, kwargs_list)]

    result = initial_value
    for future in concurrent.futures.as_completed(futures):
        result = reduction(result, future.result())

    return result


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


class HestonBenchmarkResults:
    def __init__(self,
                 numerical_package_bundle: tp.Type[NumericalPackageBundle],
                 total_time: float,
                 option_price: float):
        self._numerical_package_bundle = numerical_package_bundle
        self._total_time = total_time
        self._option_price = option_price

    @property
    def numerical_package_bundle(self) -> tp.Type[NumericalPackageBundle]:
        return self._numerical_package_bundle

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def option_price(self) -> float:
        return self._option_price

    def print_results(self):
        label = self.numerical_package_bundle.label()
        print(f'results on {label}')
        print(
            f"Time using {label}: {self.total_time} secs")
        print(f"Call option price using {label}: {self.option_price}")


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
                  numerical_package_bundle: tp.Type[NumericalPackageBundle]) \
        -> HestonBenchmarkResults:

    # number of paths for warm-up (compile GPU kernels)
    R_warm_up = 1000
    _, _ = \
        simulate_heston_model(T=T,
                              N=nT,
                              R=R_warm_up,
                              mu=r,
                              kappa=kappa,
                              v_bar=v_bar,
                              sigma_v=sigma_v,
                              rho=rho,
                              x0=x0,
                              v0=v0,
                              numerical_package_bundle=numerical_package_bundle)

    # actual simulation run to price plain vanilla call option
    tic = time.time()
    (x_simulated, v_simulated) \
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

    # print(option_price)
    numerical_package_bundle.synchronize()
    total_time = time.time() - tic

    return HestonBenchmarkResults(numerical_package_bundle,
                                  total_time=total_time,
                                  option_price=option_price)


def run_benchmarks(
        numerical_package_bundles:
        tp.Optional[tp.Tuple[tp.Type[NumericalPackageBundle], ...]] = None) \
        -> tp.Dict[type(NumericalPackageBundle), HestonBenchmarkResults]:
    if numerical_package_bundles is None:
        numerical_package_bundles = \
            get_available_numerical_packages(list_installed_bundles=True)

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
    # R = 20000  # actual number of paths to simulate for pricing

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
             R=R)

    # single core benchmarks
    numerical_package_bundle_to_result_map = {}
    for numerical_package_bundle in numerical_package_bundles:
        kwargs['numerical_package_bundle'] = numerical_package_bundle
        # run benchmark on gpu
        heston_benchmark_results \
            = run_benchmark(**kwargs)

        numerical_package_bundle_to_result_map[numerical_package_bundle] \
            = heston_benchmark_results

        heston_benchmark_results.print_results()

    # multi core benchmarks
    if loky_installed():
        kwargs['numerical_package_bundle'] = NumpyMulticoreBundle
        # initialize Python processes
        tic = time.time()
        _ \
            = simulate_and_compute_option_price_multicore(**kwargs)
        print(f'time in first run={time.time() - tic}')

        tic = time.time()
        option_price \
            = simulate_and_compute_option_price_multicore(**kwargs)

        total_time = time.time() - tic
        numpy_multicore_results = \
            HestonBenchmarkResults(
                numerical_package_bundle=NumpyMulticoreBundle,
                total_time=total_time,
                option_price=option_price)

        numpy_multicore_results.print_results()

        numerical_package_bundle_to_result_map[NumpyMulticoreBundle] \
            = numpy_multicore_results
    else:
        print(f'Please install loky to enable multi core benchmarks')

    return numerical_package_bundle_to_result_map


def create_result_dataframe(
        numerical_package_bundle_to_result_map:
        tp.Dict[type(NumericalPackageBundle), HestonBenchmarkResults]) \
        -> pd.DataFrame:
    numpy_results = numerical_package_bundle_to_result_map[NumpyBundle]

    table_dict = {}

    for numerical_package_bundle, result \
            in numerical_package_bundle_to_result_map.items():
        entry_dict = {'Total Time in Seconds': result.total_time,
                      'Speedup Compared to NumPy':
                          numpy_results.total_time / result.total_time}
        table_dict[numerical_package_bundle.label()] = entry_dict

    return pd.DataFrame(table_dict).transpose()


def create_result_table(
        numerical_package_bundle_to_result_map:
        tp.Dict[type(NumericalPackageBundle), HestonBenchmarkResults]) -> str:
    numpy_results = numerical_package_bundle_to_result_map[NumpyBundle]

    res = "<table>\n"
    res += "<tbody>\n"
    res += "<tr>\n"
    res += "<th></th>\n"
    res += "<th>Total Time in Seconds</th>\n"
    res += "<th>Speedup Compared to NumPy</th>\n"
    res += "</tr>\n"

    for numerical_package_bundle, result \
            in numerical_package_bundle_to_result_map.items():
        res += "<tr>\n"
        res += f"<td>{numerical_package_bundle.label()}</td>\n"
        res += f"<td>{result.total_time}</td>\n"
        res += f"<td>{numpy_results.total_time / result.total_time}</td>\n"
        res += "</tr>\n"

    res += "</table>"

    return res


def create_bar_plot(
        numerical_package_bundle_to_result_map:
        tp.Dict[type(NumericalPackageBundle), HestonBenchmarkResults]):
    numpy_results = numerical_package_bundle_to_result_map[NumpyBundle]
    device_name \
       = cocos.device.ComputeDeviceManager.get_current_compute_device().name

    objects = [numerical_package_bundle.label()
               for numerical_package_bundle
               in numerical_package_bundle_to_result_map.keys()]

    y_pos = numpy.arange(len(objects))

    performance = [numpy_results.total_time / result.total_time
                   for result
                   in numerical_package_bundle_to_result_map.values()]

    plt.figure(1)
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Speedup Factor')
    plt.title('Performance Relative to NumPy \n'
              'in Monte Carlo Simulation of Heston Model \n')

    plt.show()

    plt.savefig(f'heston_pricing_benchmark_results_{device_name}')


def main():
    numerical_package_bundle_to_result_map = run_benchmarks()

    # create result dataframe
    result_df = create_result_dataframe(numerical_package_bundle_to_result_map)
    print(result_df)

    # create result table
    result_table = create_result_table(numerical_package_bundle_to_result_map)
    print(result_table)

    # plot and save result bar graph
    create_bar_plot(numerical_package_bundle_to_result_map)


if __name__ == '__main__':
    main()
