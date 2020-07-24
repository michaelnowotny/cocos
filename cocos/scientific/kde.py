#-------------------------------------------------------------------------------
#
#  Define classes for (uni/multi)-variate kernel density estimation.
#
#  Currently, only Gaussian kernels are implemented.
#
#  Copyright 2004-2005 by Enthought, Inc.
#
#  The code has been adapted by Michael Nowotny to work with GPUs
#  using Cocos from the SciPy code available at
#  https://github.com/scipy/scipy/blob/master/scipy/stats/kde.py
#
#  The open source license of the original code is reproduced below:
#
# Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#-------------------------------------------------------------------------------

# Standard library imports.
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from cached_property import cached_property
import numbers
import typing as tp
import warnings

from cocos.multi_processing.device_pool import ComputeDevicePool
from cocos.multi_processing.single_device_batch_processing import map_combine_single_device
from cocos.multi_processing.utilities import generate_slices_with_number_of_batches
import cocos.numerics as cn
from cocos.numerics.data_types import NumericArray
import cocos.device as cd
from cocos.numerics.numerical_package_selector import select_num_pack

# SciPy imports.
from scipy import linalg, special
from scipy.special import logsumexp
from scipy._lib._util import check_random_state

from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, dot, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones, cov)
import numpy as np

# Local imports.
# # from . import mvn
# from scipy.stats import mvn
# from ._stats import gaussian_kernel_estimate
#
# from scipy.stats._stats import gaussian_kernel_estimate

__all__ = ['gaussian_kde']


def _split_points_into_batches(points: NumericArray,
                               number_of_points_per_batch: int) \
        -> tp.List[tp.List[NumericArray]]:
    number_of_points = points.shape[1]

    n_begin = 0
    args_list = []

    while n_begin < number_of_points:
        n_end = min(n_begin + number_of_points_per_batch, number_of_points)
        args_list.append([points[:, n_begin:n_end]])
        n_begin = n_end

    return args_list


def _check_array_at_right_location_and_convert(array,
                                               gpu: bool,
                                               dtype: np.generic = np.float32):
    if isinstance(array, np.ndarray) and gpu:
        array = cn.array(array)

    if isinstance(array, cn.ndarray) and not gpu:
        array = np.array(array)

    if array.dtype != dtype:
        array = array.astype(dtype)

    return array


def ensure_consistent_numeric_arrays(arrays: tp.Iterable[tp.Optional[NumericArray]],
                                     gpu: bool,
                                     dtype: np.generic = np.float32):
    return tuple(_check_array_at_right_location_and_convert(array=array, gpu=gpu, dtype=dtype)
                 if array is not None
                 else None
                 for array
                 in arrays)


def _verify_and_get_shape_of_datapoints_datavalues_and_evaluation_points(points: NumericArray,
                                                                         values: NumericArray,
                                                                         xi: NumericArray) \
        -> tp.Tuple[int, int, int]:
    n = points.shape[0]

    if points.ndim > 1:
        d = points.shape[1]
    else:
        d = 1
    m = xi.shape[0]

    if values.ndim > 1:
        p = values.shape[1]
    else:
        p = 1

    if p != 1:
        raise ValueError('p != 1 is not supported')

    if xi.shape[1] != d:
        raise ValueError(f"points and xi must have same trailing dim but the shape of xi is {xi.shape}")

    return n, m, d


def gaussian_kernel_estimate_vectorized_whitened(whitening: NumericArray,
                                                 whitened_points: NumericArray,
                                                 values: NumericArray,
                                                 xi: NumericArray,
                                                 norm: float,
                                                 dtype: np.generic,
                                                 gpu: bool) -> NumericArray:
    n, m, d = \
        _verify_and_get_shape_of_datapoints_datavalues_and_evaluation_points(points=whitened_points,
                                                                             values=values,
                                                                             xi=xi)
    whitened_points, values, xi, whitening = \
        ensure_consistent_numeric_arrays((whitened_points, values, xi, whitening), gpu)

    num_pack = select_num_pack(gpu)

    whitened_points = whitened_points.astype(dtype, copy=False)
    whitened_xi = num_pack.dot(xi, whitening).astype(dtype, copy=False)
    values = values.astype(dtype, copy=False)

    # Create the result array and evaluate the weighted sum
    whitened_points = whitened_points.reshape((n, 1, d))
    whitened_xi = whitened_xi.reshape((1, m, d))
    residual = whitened_points - whitened_xi
    arg = residual * residual
    del residual
    if d > 1:
        assert arg.shape == (n, m, d)
        arg = num_pack.sum(arg, axis=2)
    else:
        arg = arg.reshape((n, m))
    if not gpu:
        assert arg.shape == (n, m)
    arg = num_pack.exp(- 0.5 * arg) * norm
    if not gpu:
        assert arg.shape == (n, m)

    # estimate = num_pack.dot(arg.T, values)
    estimate = (values * arg).sum(axis=0)
    if estimate.ndim > 1:
        estimate = estimate.squeeze()

    if gpu:
        cd.sync()

    return estimate


def gaussian_kernel_estimate_vectorized(points: NumericArray,
                                        values: NumericArray,
                                        xi: NumericArray,
                                        precision: NumericArray,
                                        dtype: np.generic,
                                        gpu: bool = False) \
        -> NumericArray:
    """
    def gaussian_kernel_estimate(points, real[:, :] values, xi, precision)
    Evaluate a multivariate Gaussian kernel estimate.
    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimenions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    precision : array_like with shape (d, d)
        Precision matrix for the Gaussian kernel.
    dtype : the result dtype
    gpu : whether to compute the gaussian kernel estimate on the gpu

    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input coordinates.
    """
    num_pack = select_num_pack(gpu)
    n, m, d = \
        _verify_and_get_shape_of_datapoints_datavalues_and_evaluation_points(points=points,
                                                                             values=values,
                                                                             xi=xi)

    # n = points.shape[0]
    #
    # if points.ndim > 1:
    #     d = points.shape[1]
    # else:
    #     d = 1
    # m = xi.shape[0]
    #
    # if values.ndim > 1:
    #     p = values.shape[1]
    # else:
    #     p = 1
    #
    # if p != 1:
    #     raise ValueError('p != 1 is not supported')
    #
    # if xi.shape[1] != d:
    #     raise ValueError("points and xi must have same trailing dim")
    # if precision.shape[0] != d or precision.shape[1] != d:
    #     raise ValueError("precision matrix must match data dims")

    points, values, xi, precision = \
        ensure_consistent_numeric_arrays((points, values, xi, precision), gpu)

    print(f'type(points) = {type(points)}')
    print(f'type(values) = {type(values)}')
    print(f'type(xi) = {type(xi)}')
    print(f'type(precision) = {type(precision)}')

    # Rescale the data
    whitening = num_pack.linalg.cholesky(precision).astype(dtype, copy=False)
    points = num_pack.dot(points, whitening).astype(dtype, copy=False)
    # xi = num_pack.dot(xi, whitening).astype(dtype, copy=False)
    values = values.astype(dtype, copy=False)

    # Evaluate the normalisation
    norm = (2 * np.pi) ** (- d / 2) * num_pack.prod(num_pack.diag(whitening))

    # # Create the result array and evaluate the weighted sum
    # points = points.reshape((n, 1, d))
    # xi = xi.reshape((1, m, d))
    # residual = points - xi
    # arg = residual * residual
    # del residual
    # if d > 1:
    #     assert arg.shape == (n, m, d)
    #     arg = num_pack.sum(arg, axis=2)
    # else:
    #     arg = arg.reshape((n, m))
    # assert arg.shape == (n, m)
    # arg = num_pack.exp(- 0.5 * arg) * norm
    # assert arg.shape == (n, m)
    #
    # estimate = num_pack.dot(arg.T, values)
    #
    # if gpu:
    #     cd.sync()
    #
    # return estimate.squeeze()

    return gaussian_kernel_estimate_vectorized_whitened(whitening=whitening,
                                                        whitened_points=points,
                                                        xi=xi,
                                                        values=values,
                                                        norm=norm,
                                                        dtype=dtype,
                                                        gpu=gpu)


def gaussian_kernel_estimate(points, values, xi, precision, dtype):
    """
    def gaussian_kernel_estimate(points, real[:, :] values, xi, precision)
    Evaluate a multivariate Gaussian kernel estimate.
    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimenions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    precision : array_like with shape (d, d)
        Precision matrix for the Gaussian kernel.
    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input coordinates.
    """
    n = points.shape[0]
    d = points.shape[1]
    m = xi.shape[0]
    p = values.shape[1]

    if p != 1:
        raise ValueError('p != 1 is not supported')

    if xi.shape[1] != d:
        raise ValueError("points and xi must have same trailing dim")
    if precision.shape[0] != d or precision.shape[1] != d:
        raise ValueError("precision matrix must match data dims")

    # Rescale the data
    whitening = np.linalg.cholesky(precision).astype(dtype, copy=False)
    points_ = np.dot(points, whitening).astype(dtype, copy=False)
    xi_ = np.dot(xi, whitening).astype(dtype, copy=False)
    values_ = values.astype(dtype, copy=False)

    # Evaluate the normalisation
    norm = (2 * np.pi) ** (- d / 2)
    for i in range(d):
        norm *= whitening[i, i]

    # Create the result array and evaluate the weighted sum
    estimate = np.zeros((m, p), dtype)
    for i in range(n):
        for j in range(m):
            arg = 0
            for k in range(d):
                residual = (points_[i, k] - xi_[j, k])
                arg += residual * residual

            arg = np.exp(-arg / 2) * norm
            for k in range(p):
                estimate[j, k] += values_[i, k] * arg

    return np.asarray(estimate)


@dataclass(frozen=True)
class GaussianKDEInformation:
    points: np.ndarray  # (d, n) shaped array of datapoints
    weights: np.ndarray  # (d, n) shaped array of weights, optional
    dimension: int  # data dimension
    n: int  # number of data points
    neff: float  # effective sample size


CovarianceFactorFunctionType = tp.Callable[[GaussianKDEInformation], float]


SCOTTS_FACTOR_STRING = 'scotts'
SILVERMAN_FACTOR_STRING = 'silverman'


def compute_scotts_factor(kde_info: GaussianKDEInformation) -> float:
    return power(kde_info.neff, -1.0 / (kde_info.dimension + 4))


def compute_silverman_factor(kde_info: GaussianKDEInformation) -> float:
    d = kde_info.dimension
    neff = kde_info.neff
    return power(neff * (d + 2.0) / 4.0, -1.0 / (d + 4))


# class CovarianceFactor(ABC):
#     @abstractmethod
#     def compute_covariance_factor(self, kde_info: GaussianKDEInformation) -> float:
#         pass
#
#
# class ScottsFactor(CovarianceFactor):
#     def compute_covariance_factor(self, kde_info: GaussianKDEInformation) -> float:
#         return power(kde_info.neff, -1.0 / (kde_info.dimension + 4))
#
#
# class SilvermanFactor(CovarianceFactor):
#     def compute_covariance_factor(self, kde_info: GaussianKDEInformation) -> float:
#         d = kde_info.dimension
#         neff = kde_info.neff
#         return power(neff * (d + 2.0) / 4.0, -1.0 / (d + 4))
#
#
# class LambdaCovarianceFactor(CovarianceFactor):
#     def __init__(self, covariance_factor_fun: tp.Callable[[GaussianKDEInformation], float]):
#         self._covariance_factor_fun = covariance_factor_fun
#
#     def compute_covariance_factor(self, kde_info: GaussianKDEInformation) -> float:
#         return self._covariance_factor_fun(kde_info)


class gaussian_kde:
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data. It
    includes automatic bandwidth determination. The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, optional
        weights of datapoints. This must be the same shape as dataset.
        If None (default), the samples are assumed to be equally weighted
    gpu: whether to evaluate the kernel density estimate on the gpu

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : int
        Effective number of datapoints.

        .. versionadded:: 1.2.0
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    evaluate
    __call__
    integrate_gaussian
    integrate_box_1d
    integrate_box
    integrate_kde
    pdf
    logpdf
    resample

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    In the case of unequally weighted points, `scotts_factor` becomes::

        neff**(-1./(d+4)),

    with ``neff`` the effective number of datapoints.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    or in the case of unequally weighted points::

        (neff * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    With a set of weighted samples, the effective number of datapoints ``neff``
    is defined by::

        neff = sum(weights)^2 / sum(weights^2)

    as detailed in [5]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.
    .. [5] Gray P. G., 1969, Journal of the Royal Statistical Society.
           Series A (General), 132, 272

    Examples
    --------
    Generate some random two-dimensional data:

    >>> from scipy import stats
    >>> def measure(n):
    ...     "Measurement model, return two coupled measurements."
    ...     m1 = np.random.normal(size=n)
    ...     m2 = np.random.normal(scale=0.5, size=n)
    ...     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    """

    def __init__(self,
                 dataset: NumericArray,
                 bw_method: tp.Optional[tp.Union[CovarianceFactorFunctionType,
                                                 str,
                                                 tp.Callable,
                                                 numbers.Number]] = None,
                 weights: tp.Optional[NumericArray] = None,
                 gpu: bool = False):

        self._num_pack = select_num_pack(gpu)
        self._gpu = gpu
        self.dataset = atleast_2d(asarray(dataset))
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            weights = atleast_1d(weights).astype(float)
            weights /= np.sum(weights)
            if weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1.0/np.sum(weights**2)
        else:
            weights = ones(self.n) / self.n

        if gpu:
            dtype = np.float32
            weights = weights.astype(dtype)
            self.dataset = self.dataset.astype(dtype)

        self._weights = weights
        self._covariance_factor = \
            self._get_covariance_factor_function_from_bandwidth_type(bw_method)

        self._compute_covariance()

    def _check_and_adjust_dimensions_of_points(self, points: np.ndarray) \
            -> np.ndarray:
        points = atleast_2d(asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                raise ValueError(f"points have dimension {d}, "
                                 f"dataset has dimension {self.d}")

        return points

    def evaluate(self, points):
        """
        Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = self._check_and_adjust_dimensions_of_points(points)
        output_dtype = np.common_type(self.covariance, points)

        if True:
            # result = gaussian_kernel_estimate_vectorized(points=self.dataset.T,
            #                                              values=self.weights[:, None],
            #                                              xi=points.T,
            #                                              precision=self.inv_cov,
            #                                              dtype=output_dtype,
            #                                              gpu=self._gpu)

            result = gaussian_kernel_estimate_vectorized_whitened(
                        whitening=self.whitening,
                        whitened_points=self.whitened_points,
                        values=self.weights[:, None],
                        xi=points.T,
                        norm=self.normalization_constant,
                        dtype=output_dtype,
                        gpu=self._gpu)

            return result
        else:
            result = gaussian_kernel_estimate(points=self.dataset.T,
                                              values=self.weights[:, None],
                                              xi=points.T,
                                              precision=self.inv_cov,
                                              dtype=output_dtype)
            return result[:, 0]

    __call__ = evaluate

    def evaluate_in_batches(self,
                            points: NumericArray,
                            maximum_number_of_elements_per_batch: int) \
            -> np.ndarray:
        """
        Evaluates a Gaussian KDE in batches and stores the results in main memory.

        Args:
            points:
                numeric array with shape (d, m) containing the points at which to evaluate the kernel
                density estimate
            maximum_number_of_elements_per_batch:
                maximum number of data points times evaluation points to process in a single batch

        Returns:
            a m-dimensional NumPy array of kernel density estimates
        """
        points_per_batch = math.floor(maximum_number_of_elements_per_batch / (self.n * self.d))

        args_list = _split_points_into_batches(points, points_per_batch)

        result = \
            map_combine_single_device(f=self.evaluate,
                                      combination=lambda x: np.hstack(x),
                                      args_list=args_list)

        return result

    def evaluate_in_batches_on_multiple_devices(self,
                                                points: NumericArray,
                                                maximum_number_of_elements_per_batch: int,
                                                compute_device_pool: ComputeDevicePool) \
            -> np.ndarray:
        """
        Evaluates a Gaussian KDE in batches on multiple gpus and stores the results in main memory.

        Args:
            points:
                numeric array with shape (d, m) containing the points at which to evaluate the kernel
                density estimate
            maximum_number_of_elements_per_batch:
                maximum number of data points times evaluation points to process in a single batch

        Returns:
            a m-dimensional NumPy array of kernel density estimates
        """
        if self.gpu:
            raise ValueError('Multi GPU evaluation requires gaussian_kde.gpu = False.')

        points = self._check_and_adjust_dimensions_of_points(points)

        number_of_points = points.shape[1]

        args_list = []
        for begin_index, end_index in generate_slices_with_number_of_batches(number_of_points,
                                                                             compute_device_pool.number_of_devices):
            args_list.append([points[:, begin_index:end_index]])

        # points_per_device = math.floor(number_of_points / gpu_pool.number_of_devices)
        # args_list = _split_points_into_batches(points, points_per_device)
        kwargs_list = compute_device_pool.number_of_devices * \
                      [
                          {'maximum_number_of_elements_per_batch': maximum_number_of_elements_per_batch,
                           'n': self.n,
                           'd': self.d}
                      ]

        def f(points_internal,
              maximum_number_of_elements_per_batch: int,
              n: int,
              d: int):
            points_per_batch = math.floor(maximum_number_of_elements_per_batch / (n * d))
            args_list_internal = _split_points_into_batches(points_internal, points_per_batch)

            def f_internal(points_internal_internal):
                return gaussian_kernel_estimate_vectorized_whitened(
                        whitening=self.whitening,
                        whitened_points=self.whitened_points,
                        values=self.weights[:, None],
                        xi=points_internal_internal.T,
                        norm=self.normalization_constant,
                        dtype=np.float32,
                        gpu=True)

            result = \
                map_combine_single_device(f=f_internal,
                                          combination=lambda x: np.hstack(x),
                                          args_list=args_list_internal)

            return result

        result = \
            compute_device_pool.map_combine(f=f,
                                            combination=lambda x: np.hstack(x),
                                            args_list=args_list,
                                            kwargs_list=kwargs_list)

        return result

    # def evaluate_in_batches_on_multiple_gpus(self,
    #                                          points: NumericArray,
    #                                          maximum_number_of_elements_per_batch: int,
    #                                          gpu_pool: ComputeDevicePool) \
    #         -> np.ndarray:
    #     """
    #     Evaluates a Gaussian KDE in batches on multiple gpus and stores the results in main memory.
    #
    #     Args:
    #         points:
    #             numeric array with shape (d, m) containing the points at which to evaluate the kernel
    #             density estimate
    #         maximum_number_of_elements_per_batch:
    #             maximum number of data points times evaluation points to process in a single batch
    #
    #     Returns:
    #         a m-dimensional NumPy array of kernel density estimates
    #     """
    #     if self.gpu:
    #         raise ValueError('Multi GPU evaluation requires gaussian_kde.gpu = False.')
    #
    #     points = self._check_and_adjust_dimensions_of_points(points)
    #
    #     # number_of_points = points.shape[1]
    #     points_per_batch = math.floor(maximum_number_of_elements_per_batch / (self.n * self.d))
    #
    #     args_list = _split_points_into_batches(points, points_per_batch)
    #
    #     def f(x):
    #         result = gaussian_kernel_estimate_vectorized_whitened(
    #             whitening=self.whitening,
    #             whitened_points=self.whitened_points,
    #             values=self.weights[:, None],
    #             xi=x.T,
    #             norm=self.normalization_constant,
    #             dtype=np.float32,
    #             gpu=True)
    #
    #         return result
    #
    #     result = \
    #         gpu_pool.map_combine(f=f,
    #                              combination=lambda x: np.hstack(x),
    #                              args_list=args_list)
    #
    #     return result

    def integrate_gaussian(self, mean, cov):
        """
        Multiply estimated density by a multivariate Gaussian and integrate
        over the whole space.

        Parameters
        ----------
        mean : aray_like
            A 1-D array, specifying the mean of the Gaussian.
        cov : array_like
            A 2-D array, specifying the covariance matrix of the Gaussian.

        Returns
        -------
        result : scalar
            The value of the integral.

        Raises
        ------
        ValueError
            If the mean or covariance of the input Gaussian differs from
            the KDE's dimensionality.

        """
        mean = atleast_1d(squeeze(mean))
        cov = atleast_2d(cov)

        if mean.shape != (self.d,):
            raise ValueError("mean does not have dimension %s" % self.d)
        if cov.shape != (self.d, self.d):
            raise ValueError("covariance does not have dimension %s" % self.d)

        # make mean a column vector
        mean = mean[:, newaxis]

        sum_cov = self.covariance + cov

        # This will raise LinAlgError if the new cov matrix is not s.p.d
        # cho_factor returns (ndarray, bool) where bool is a flag for whether
        # or not ndarray is upper or lower triangular
        sum_cov_chol = linalg.cho_factor(sum_cov)

        diff = self.dataset - mean
        tdiff = linalg.cho_solve(sum_cov_chol, diff)

        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        energies = sum(diff * tdiff, axis=0) / 2.0
        result = sum(exp(-energies)*self.weights, axis=0) / norm_const

        return result

    def integrate_box_1d(self, low, high):
        """
        Computes the integral of a 1D pdf between two bounds.

        Parameters
        ----------
        low : scalar
            Lower bound of integration.
        high : scalar
            Upper bound of integration.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDE is over more than one dimension.

        """
        if self.d != 1:
            raise ValueError("integrate_box_1d() only handles 1D pdfs")

        stdev = ravel(sqrt(self.covariance))[0]

        normalized_low = ravel((low - self.dataset) / stdev)
        normalized_high = ravel((high - self.dataset) / stdev)

        value = np.sum(self.weights*(
                        special.ndtr(normalized_high) -
                        special.ndtr(normalized_low)))
        return value

    # def integrate_box(self, low_bounds, high_bounds, maxpts=None):
    #     """Computes the integral of a pdf over a rectangular interval.
    #
    #     Parameters
    #     ----------
    #     low_bounds : array_like
    #         A 1-D array containing the lower bounds of integration.
    #     high_bounds : array_like
    #         A 1-D array containing the upper bounds of integration.
    #     maxpts : int, optional
    #         The maximum number of points to use for integration.
    #
    #     Returns
    #     -------
    #     value : scalar
    #         The result of the integral.
    #
    #     """
    #     if maxpts is not None:
    #         extra_kwds = {'maxpts': maxpts}
    #     else:
    #         extra_kwds = {}
    #
    #     value, inform = mvn.mvnun_weighted(low_bounds, high_bounds,
    #                                        self.dataset, self.weights,
    #                                        self.covariance, **extra_kwds)
    #     if inform:
    #         msg = ('An integral in mvn.mvnun requires more points than %s' %
    #                (self.d * 1000))
    #         warnings.warn(msg)
    #
    #     return value

    def integrate_kde(self, other):
        """
        Computes the integral of the product of this  kernel density estimate
        with another.

        Parameters
        ----------
        other : gaussian_kde instance
            The other kde.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDEs have different dimensionality.

        """
        if other.d != self.d:
            raise ValueError("KDEs are not the same dimensionality")

        # we want to iterate over the smallest number of points
        if other.n < self.n:
            small = other
            large = self
        else:
            small = self
            large = other

        sum_cov = small.covariance + large.covariance
        sum_cov_chol = linalg.cho_factor(sum_cov)
        result = 0.0
        for i in range(small.n):
            mean = small.dataset[:, i, newaxis]
            diff = large.dataset - mean
            tdiff = linalg.cho_solve(sum_cov_chol, diff)

            energies = sum(diff * tdiff, axis=0) / 2.0
            result += sum(exp(-energies)*large.weights, axis=0)*small.weights[i]

        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        result /= norm_const

        return result

    def resample(self, size=None, seed=None):
        """
        Randomly sample a dataset from the estimated pdf.

        Parameters
        ----------
        size : int, optional
            The number of samples to draw.  If not provided, then the size is
            the same as the effective number of samples in the underlying
            dataset.
        seed : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            This parameter defines the object to use for drawing random
            variates.
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.
            Specify `seed` for reproducible drawing of random variates.

        Returns
        -------
        resample : (self.d, `size`) ndarray
            The sampled dataset.

        """
        if size is None:
            size = int(self.neff)

        random_state = check_random_state(seed)
        norm = transpose(random_state.multivariate_normal(
            zeros((self.d,), float), self.covariance, size=size
        ))
        indices = random_state.choice(self.n, size=size, p=self.weights)
        means = self.dataset[:, indices]

        return means + norm

    @staticmethod
    def _get_covariance_factor_function_from_bandwidth_type(
            bw_method: tp.Optional[tp.Union[CovarianceFactorFunctionType,
                                            str,
                                            tp.Callable,
                                            numbers.Number]] = None) \
            -> CovarianceFactorFunctionType:
        """
        Infers the bandwidth selection method from
        Args:
            bw_method: either 'scotts' or 'silverman' or a scalar or a function returning a float

        Returns:
        covariance factor function
        """
        if bw_method is None:
            return compute_scotts_factor
        elif isinstance(bw_method, str):
            if bw_method == SCOTTS_FACTOR_STRING:
                return compute_scotts_factor
            elif bw_method == SILVERMAN_FACTOR_STRING:
                return compute_silverman_factor
            else:
                raise ValueError(f'bw_method={bw_method} is not supported')
        elif callable(bw_method):
            return bw_method
        elif np.isscalar(bw_method):
            return lambda kde_info: bw_method
        else:
            raise ValueError(f'bw_method {bw_method} is not supported')

    def _compute_covariance(self):
        """
        Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        kde_info = GaussianKDEInformation(dimension=self.d,
                                          n=self.n,
                                          neff=self.neff,
                                          points=self.dataset,
                                          weights=self.weights)

        self.factor = self._covariance_factor(kde_info)

        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = \
                atleast_2d(cov(self.dataset,
                               rowvar=True,
                               bias=False,
                               aweights=self.weights))
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = sqrt(linalg.det(2*pi*self.covariance))

    def pdf(self, x: np.ndarray) -> NumericArray:
        """
        Evaluate the estimated pdf on a provided set of points.

        Notes
        -----
        This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``
        docstring for more details.

        """
        return self.evaluate(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the log of the estimated pdf on a provided set of points.
        """

        points = atleast_2d(x)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        if m >= self.n:
            # there are more points than data, so loop over data
            energy = zeros((self.n, m), dtype=float)
            for i in range(self.n):
                diff = self.dataset[:, i, newaxis] - points
                tdiff = dot(self.inv_cov, diff)
                energy[i] = sum(diff*tdiff, axis=0) / 2.0
            result = logsumexp(-energy.T,
                               b=self.weights / self._norm_factor, axis=1)
        else:
            # loop over points
            result = zeros((m,), dtype=float)
            for i in range(m):
                diff = self.dataset - points[:, i, newaxis]
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff * tdiff, axis=0) / 2.0
                result[i] = logsumexp(-energy, b=self.weights /
                                      self._norm_factor)

        return result

    @property
    def gpu(self) -> bool:
        return self._gpu

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @cached_property
    def neff(self) -> float:
        return 1.0/np.sum(self.weights*self.weights)

    @cached_property
    def whitening(self) -> NumericArray:
        gpu = self.gpu
        num_pack = select_num_pack(gpu)
        precision = \
            ensure_consistent_numeric_arrays((self.inv_cov, ), gpu)[0]

        return num_pack.linalg.cholesky(precision)

    @cached_property
    def whitened_points(self) -> NumericArray:
        gpu = self.gpu
        num_pack = select_num_pack(gpu)
        points = \
            ensure_consistent_numeric_arrays((self.dataset.T, ), gpu)[0]

        return num_pack.dot(points, self.whitening)

    @cached_property
    def normalization_constant(self) -> float:
        gpu = self.gpu
        num_pack = select_num_pack(gpu)

        return (2 * np.pi) ** (- self.d / 2) * num_pack.prod(num_pack.diag(self.whitening))


# def evaluate_gaussian_kde_in_batches(kde: gaussian_kde,
#                                      points: NumericArray,
#                                      maximum_number_of_elements_per_batch: int) \
#         -> np.ndarray:
#     """
#     Evaluates a Gaussian KDE in batches and stores the results in main memory.
#
#     Args:
#         kde: a gaussian_kde object
#         points:
#             numeric array with shape (d, m) containing the points at which to evaluate the kernel
#             density estimate
#         maximum_number_of_elements_per_batch:
#             maximum number of data points times evaluation points to process in a single batch
#
#     Returns:
#         a m-dimensional NumPy array of kernel density estimates
#     """
#     number_of_points = points.shape[1]
#     points_per_batch = math.floor(maximum_number_of_elements_per_batch / (kde.n * kde.d))
#
#     n_begin = 0
#
#     output_array = np.zeros((number_of_points, ), dtype=points.dtype)
#
#     while n_begin < number_of_points:
#         n_end = min(n_begin + points_per_batch, number_of_points)
#         output_array[n_begin:n_end] = np.array(kde.evaluate(points[:, n_begin:n_end]))
#         n_begin = n_end
#
#     return output_array


def evaluate_gaussian_kde_in_batches(kde: gaussian_kde,
                                     points: NumericArray,
                                     maximum_number_of_elements_per_batch: int) \
        -> np.ndarray:
    """
    Evaluates a Gaussian KDE in batches and stores the results in main memory.

    Args:
        kde: a gaussian_kde object
        points:
            numeric array with shape (d, m) containing the points at which to evaluate the kernel
            density estimate
        maximum_number_of_elements_per_batch:
            maximum number of data points times evaluation points to process in a single batch

    Returns:
        a m-dimensional NumPy array of kernel density estimates
    """
    points_per_batch = math.floor(maximum_number_of_elements_per_batch / (kde.n * kde.d))

    args_list = _split_points_into_batches(points, points_per_batch)

    result = \
        map_combine_single_device(f=kde.evaluate,
                                  combination=lambda x: np.hstack(x),
                                  args_list=args_list)

    return result
