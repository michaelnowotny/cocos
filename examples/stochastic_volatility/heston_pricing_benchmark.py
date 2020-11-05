from contexttimer import Timer
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy
import pandas as pd
import time
import typing as tp

import cocos.device
from cocos.multi_processing.device_pool import ComputeDevicePool
from cocos.numerics.random import randn_antithetic
from cocos.numerics.numerical_package_bundle import (
    get_available_numerical_packages,
    NumericalPackageBundle,
    NumpyBundle,
    CocosBundle
)

from examples.stochastic_volatility.heston_utilities import (
    simulate_and_compute_option_price_gpu,
    simulate_and_compute_option_price_multicore,
    simulate_and_compute_option_price,
    simulate_heston_model,
    compute_option_price_from_simulated_paths
)


def loky_installed() -> bool:
    """
    Determine if loky is installed.

    Args:
    """
    try:
        import loky
        return True
    except:
        return False


class NumpyMulticoreBundle(NumpyBundle):
    @classmethod
    def is_installed(cls) -> bool:
        """
        Determine if the installed.

        Args:
            cls: (todo): write your description
        """
        return super().is_installed() & loky_installed()

    @classmethod
    def label(cls) -> str:
        """
        Return label.

        Args:
            cls: (callable): write your description
        """
        return 'NumPy Multicore'


class CocosMultiGPUBundle(CocosBundle):
    def __init__(self, number_of_gpus: int):
        """
        Initialize the specified number.

        Args:
            self: (todo): write your description
            number_of_gpus: (int): write your description
        """
        self.number_of_gpus = number_of_gpus

    @classmethod
    def is_installed(cls) -> bool:
        """
        Determine if the installed.

        Args:
            cls: (todo): write your description
        """
        return super().is_installed() & loky_installed()

    def label(self) -> str:
        """
        The label of the label.

        Args:
            self: (todo): write your description
        """
        if self.number_of_gpus == 1:
            return 'Cocos on 1 GPU'
        else:
            return f'Cocos on {self.number_of_gpus} GPUs'


class HestonBenchmarkResults:
    def __init__(self,
                 numerical_package_bundle: tp.Type[NumericalPackageBundle],
                 total_time: float,
                 option_price: float):
        """
        Initialize the bundle bundle.

        Args:
            self: (todo): write your description
            numerical_package_bundle: (int): write your description
            tp: (int): write your description
            Type: (str): write your description
            NumericalPackageBundle: (int): write your description
            total_time: (int): write your description
            option_price: (todo): write your description
        """
        self._numerical_package_bundle = numerical_package_bundle
        self._total_time = total_time
        self._option_price = option_price

    @property
    def numerical_package_bundle(self) -> tp.Type[NumericalPackageBundle]:
        """
        The bundle bundle bundle bundle bundle.

        Args:
            self: (todo): write your description
        """
        return self._numerical_package_bundle

    @property
    def total_time(self) -> float:
        """
        The total amount of the total time.

        Args:
            self: (todo): write your description
        """
        return self._total_time

    @property
    def option_price(self) -> float:
        """
        Returns the default price of the given option.

        Args:
            self: (todo): write your description
        """
        return self._option_price

    def print_results(self):
        """
        Print the bundle results.

        Args:
            self: (todo): write your description
        """
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
    """
    Benchmarkmarkmarkmarkmarkmarkmark

    Args:
        x0: (array): write your description
        v0: (str): write your description
        r: (str): write your description
        rho: (str): write your description
        sigma_v: (float): write your description
        kappa: (str): write your description
        v_bar: (str): write your description
        T: (str): write your description
        K: (str): write your description
        nT: (str): write your description
        R: (str): write your description
        numerical_package_bundle: (bool): write your description
        tp: (str): write your description
        Type: (str): write your description
        NumericalPackageBundle: (int): write your description
    """

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
    with Timer() as timer:
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
    total_time = timer.elapsed

    return HestonBenchmarkResults(numerical_package_bundle,
                                  total_time=total_time,
                                  option_price=option_price)


def run_benchmarks(
        numerical_package_bundles:
        tp.Optional[tp.Tuple[tp.Type[NumericalPackageBundle], ...]] = None) \
        -> tp.Dict[type(NumericalPackageBundle), HestonBenchmarkResults]:
    """
    Execute benchmarks.

    Args:
        numerical_package_bundles: (bool): write your description
        tp: (str): write your description
        Optional: (todo): write your description
        tp: (str): write your description
        Tuple: (str): write your description
        tp: (str): write your description
        Type: (str): write your description
        NumericalPackageBundle: (int): write your description
    """
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

    if loky_installed():
        # multi core benchmarks
        kwargs['numerical_package_bundle'] = NumpyMulticoreBundle
        # initialize Python processes
        with Timer() as timer:
            _ \
                = simulate_and_compute_option_price_multicore(**kwargs)
            print(f'time in first run={timer.elapsed}')

        with Timer() as timer:
            option_price \
                = simulate_and_compute_option_price_multicore(**kwargs)

        total_time = timer.elapsed

        numpy_multicore_results = \
            HestonBenchmarkResults(
                numerical_package_bundle=NumpyMulticoreBundle,
                total_time=total_time,
                option_price=option_price)

        numpy_multicore_results.print_results()

        numerical_package_bundle_to_result_map[NumpyMulticoreBundle] \
            = numpy_multicore_results

        # multi gpu benchmarks
        if 'numerical_package_bundle' in kwargs:
            kwargs.pop('numerical_package_bundle')

        compute_device_pool = ComputeDevicePool()
        if compute_device_pool.number_of_devices > 1:
            # warm up
            kwargs['R'] = 20000
            option_price = \
                simulate_and_compute_option_price_gpu(compute_device_pool=compute_device_pool,
                                                      **kwargs)

            # actual benchmark
            kwargs['R'] = R

            for number_of_gpus in range(1, compute_device_pool.number_of_devices + 1):
                cocos_multi_gpu_bundle = CocosMultiGPUBundle(number_of_gpus=number_of_gpus)

                with Timer() as timer:
                    option_price = \
                        simulate_and_compute_option_price_gpu(compute_device_pool=compute_device_pool,
                                                              number_of_batches=number_of_gpus,
                                                              **kwargs)
                    cocos.device.sync()

                total_time = timer.elapsed

                cocos_multi_gpu_results = \
                    HestonBenchmarkResults(
                        numerical_package_bundle=cocos_multi_gpu_bundle,
                        total_time=total_time,
                        option_price=option_price)

                cocos_multi_gpu_results.print_results()

                numerical_package_bundle_to_result_map[cocos_multi_gpu_bundle] \
                    = cocos_multi_gpu_results
    else:
        print(f'Please install loky to enable multi core and multi gpu benchmarks')

    return numerical_package_bundle_to_result_map


def create_result_dataframe(
        numerical_package_bundle_to_result_map:
        tp.Dict[type(NumericalPackageBundle), HestonBenchmarkResults]) \
        -> pd.DataFrame:
    """
    Creates a dataframe from the results table

    Args:
        numerical_package_bundle_to_result_map: (dict): write your description
        tp: (str): write your description
        Dict: (todo): write your description
        type: (str): write your description
        NumericalPackageBundle: (int): write your description
        HestonBenchmarkResults: (todo): write your description
    """
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
    """
    Builds a table with the results of a bundle.

    Args:
        numerical_package_bundle_to_result_map: (dict): write your description
        tp: (todo): write your description
        Dict: (todo): write your description
        type: (str): write your description
        NumericalPackageBundle: (int): write your description
        HestonBenchmarkResults: (todo): write your description
    """
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
    """
    Generate the bar plot

    Args:
        numerical_package_bundle_to_result_map: (todo): write your description
        tp: (str): write your description
        Dict: (todo): write your description
        type: (str): write your description
        NumericalPackageBundle: (int): write your description
        HestonBenchmarkResults: (todo): write your description
    """
    numpy_results = numerical_package_bundle_to_result_map[NumpyBundle]

    objects = [numerical_package_bundle.label()
               for numerical_package_bundle
               in numerical_package_bundle_to_result_map.keys()]

    y_pos = numpy.arange(len(objects))

    performance = [numpy_results.total_time / result.total_time
                   for result
                   in numerical_package_bundle_to_result_map.values()]

    plt.figure(1)
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation=45, ha="right")
    plt.ylabel('Speedup Factor')
    plt.title('Performance Relative to NumPy \n'
              'in Monte Carlo Simulation of Heston Model \n')

    plt.savefig(f'heston_pricing_benchmark_results', bbox_inches='tight')

    plt.show()


def main():
    """
    Execute the bundle.

    Args:
    """
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
