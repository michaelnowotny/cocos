import arrayfire as af
from arrayfire.library import Dtype
import collections
import math
import numpy as np
from types import ModuleType
import typing as tp
from ._arith import \
    exp, \
    log, \
    minimum, \
    sqrt, \
    minimum

from ._array import asscalar, ndarray
from ._conversion import \
    convert_numpy_to_af_type, \
    convert_af_to_numpy_type

from cocos.numerics.linalg import cholesky

from cocos.options import \
    GPUOptions, \
    RandomNumberGenerator


SIZE_TYPE = tp.Optional[tp.Union[int, tp.Sequence]]


def map_rng_to_random_engine(rng: RandomNumberGenerator):
    if rng == RandomNumberGenerator.PHILOX_4X32_10:
        return af.random.RANDOM_ENGINE.PHILOX_4X32_10
    elif rng == RandomNumberGenerator.THREEFRY_2X32_16:
        return af.random.RANDOM_ENGINE.THREEFRY_2X32_16
    elif rng == RandomNumberGenerator.MERSENNE_GP11213:
        return af.random.RANDOM_ENGINE.MERSENNE_GP11213
    elif rng == RandomNumberGenerator.PHILOX:
        return af.random.RANDOM_ENGINE.PHILOX
    elif rng == RandomNumberGenerator.THREEFRY:
        return af.random.RANDOM_ENGINE.THREEFRY
    elif rng == RandomNumberGenerator.DEFAULT:
        return af.random.RANDOM_ENGINE.DEFAULT
    else:
        raise ValueError("The requested random number generator "
                         "is not supported.")


# initialized default random number engine
random_engine \
    = af.random.Random_Engine(
        engine_type=map_rng_to_random_engine(GPUOptions.default_rng))


################################################################################
# functions to get and set the seed
################################################################################

def seed(seed: tp.Optional[int] = None):
    """
    Seed the generator.
    """

    if seed is None:
        seed = 0
    af.set_seed(seed)


def get_seed() -> int:
    """
    Returns the current seed of the generator.
    """

    return af.get_seed()


################################################################################
# supporting functions for antithetic random numbers
################################################################################

def get_antithetic_slices(shape: tp.Sequence[int],
                          antithetic_dimension: int) \
        -> tp.Tuple[slice, ...]:
    """
    This function generates a tuple of slices to index the original array of
    random numbers to take either half or one less than half of the the original
    random numbers along the antithetic dimension and all of the random numbers
    along the other dimensions.
    """

    slices = []

    for axis, dimension in enumerate(shape):
        if axis == antithetic_dimension:
            s = slice(0, math.floor(dimension / 2), 1)
        else:
            s = slice(0, dimension, 1)

        slices.append(s)

    return tuple(slices)


def verify_shape_and_antithetic_dimension(shape: tp.Sequence[int],
                                          antithetic_dimension: tp.Optional[
                                              int] = None):
    """
    This function makes sure that the length shape argument is between 1 and 4
    and checks that the antithetic dimension is one of the dimensions in the
    shape argument.
    """

    if len(shape) > 4:
        raise ValueError('arrays with more than 4 axes are not supported')

    if len(shape) < 1:
        raise ValueError('array must have at least one axis')

    if antithetic_dimension < 0 or antithetic_dimension > len(shape) - 1:
        raise ValueError(
            f'antithetic dimension must be None or between 0 and {len(shape)}')


################################################################################
# Basic continuous random number generators
################################################################################

def rand(d0: int,
         d1: tp.Optional[int] = None,
         d2: tp.Optional[int] = None,
         d3: tp.Optional[int] = None,
         dtype: np.generic = np.float32) -> ndarray:
    """
    Random values in a given shape.
    """

    af_type = convert_numpy_to_af_type(dtype)
    af_array = af.data.randu(d0, d1, d2, d3, dtype=af_type)

    return ndarray(af_array)


def randn(d0: int,
          d1: tp.Optional[int] = None,
          d2: tp.Optional[int] = None,
          d3: tp.Optional[int] = None,
          dtype: np.generic = np.float32):
    """
    Return a sample (or samples) from the “standard normal” distribution.
    """

    af_type = convert_numpy_to_af_type(dtype)
    af_array = af.data.randn(d0, d1, d2, d3, dtype=af_type)
    return ndarray(af_array)


def _random_with_dtype_internal(shape: tp.Sequence[int],
                                rng_function: tp.Callable,
                                dtype: np.generic = np.float32,
                                num_pack: ModuleType = np):
    draw_shape = list(shape)

    if num_pack == np:
        x = rng_function(*draw_shape)
        if x.dtype != dtype:
            x = x.astype(dtype)
    else:
        x = rng_function(*draw_shape, dtype=dtype)

    return x


def rand_with_dtype(shape: tp.Sequence[int],
                     dtype: np.generic = np.float32,
                     num_pack: ModuleType = np):
    return _random_with_dtype_internal(shape=shape,
                                       rng_function=num_pack.random.rand,
                                       dtype=dtype,
                                       num_pack=num_pack)


def randn_with_dtype(shape: tp.Sequence[int],
                     dtype: np.generic = np.float32,
                     num_pack: ModuleType = np):
    return _random_with_dtype_internal(shape=shape,
                                       rng_function=num_pack.random.randn,
                                       dtype=dtype,
                                       num_pack=num_pack)


def randn_antithetic(shape: tp.Sequence[int],
                     antithetic_dimension: tp.Optional[int] = None,
                     dtype: np.generic = np.float32,
                     num_pack: ModuleType = np):
    verify_shape_and_antithetic_dimension(shape, antithetic_dimension)
    draw_shape = list(shape)

    if antithetic_dimension is not None:
        # adjust dimension over which antithetic random numbers are to be drawn
        draw_shape[antithetic_dimension] \
            = math.ceil(shape[antithetic_dimension] / 2)

    # draw original random numbers
    if num_pack == np:
        z = num_pack.random.randn(*draw_shape)
        if z.dtype != dtype:
            z = z.astype(dtype)
    else:
        z = num_pack.random.randn(*draw_shape, dtype=dtype)

    if antithetic_dimension is not None:
        # reflect random numbers at 0 and concatenate to original random numbers
        slices = get_antithetic_slices(shape, antithetic_dimension)
        z = num_pack.concatenate((z, -z[slices]),
                                 axis=antithetic_dimension)

    return z


def rand_antithetic(shape: tp.Sequence[int],
                    antithetic_dimension: tp.Optional[int] = None,
                    dtype: np.generic = np.float32,
                    num_pack: ModuleType = np):
    verify_shape_and_antithetic_dimension(shape, antithetic_dimension)
    draw_shape = list(shape)

    if antithetic_dimension is not None:
        # adjust dimension over which antithetic random numbers are to be drawn
        draw_shape[antithetic_dimension] \
            = math.ceil(shape[antithetic_dimension] / 2)

    # draw original random numbers
    if num_pack == np:
        u = num_pack.random.rand(*draw_shape)
        if u.dtype != dtype:
            u = u.astype(dtype)
    else:
        u = num_pack.random.rand(*draw_shape, dtype=dtype)

    if antithetic_dimension is not None:
        # reflect random numbers at 0 and concatenate to original random numbers
        slices = get_antithetic_slices(shape, antithetic_dimension)
        u = num_pack.concatenate((u, 1.0 - u[slices]),
                                 axis=antithetic_dimension)

    return u


################################################################################
# Basic discrete random number generators
################################################################################

def randint(low: int,
            high: tp.Optional[int] = None,
            size: tp.Optional[tp.Union[tp.Tuple[int, ...], int]] = None,
            dtype: np.generic = np.int32) \
        -> ndarray:
    """
    Draws an array of random integers ranging from low to high-1 of the
    specified shape.

    :param low: lowest number to draw
    :param high: highest integer to draw (excluding)
    :param size: shape of output array
    :param dtype: data type of integer to be generated
    :return: an ndarray of random integers
    """
    if not high:
        high = low
        low = 0

    if not size:
        size = (1,)
    elif isinstance(size, int):
        size = (size,)

    n = np.prod(size)
    divisor = 1.0 / (high - low)

    u = rand(n)
    u = minimum(u, 1.0 - np.finfo(np.float32).eps)
    if dtype != np.int32:
        i = (u / divisor).astype(np.int64) + low
        i = i.astype(dtype)
    else:
        i = (u / divisor).astype(np.int32) + low

    return i.reshape(size)


def choice(a: ndarray,
           size: tp.Optional[tp.Union[tp.Tuple[int, ...], int]] = None,
           replace: bool = True,
           p: tp.Optional[ndarray] = None) -> ndarray:
    if p:
        raise ValueError('p != None is not supported')

    if not replace:
        raise ValueError('replace=False is not supported')

    i = randint(0, a.size, size=size)

    if not isinstance(size, int):
        return a[i].reshape(size)
    else:
        return a[i]


def _draw_and_reshape(size: SIZE_TYPE,
                      rng_func: tp.Callable[[int], ndarray]) \
        -> ndarray:
    if not size:
        n = 1
    elif isinstance(size, int):
        n = size
    elif isinstance(size, (list, tuple)):
        n = np.prod(size)
    else:
        raise TypeError("size must be either of type int or tuple")

    random_numbers = rng_func(n)

    if size is None:
        random_numbers = asscalar(random_numbers)
    elif not isinstance(size, int):
        random_numbers = random_numbers.reshape(size)

    return random_numbers


def uniform(low: float = 0.0,
            high: float = 1.0,
            size: tp.Optional[SIZE_TYPE] = None):
    """
    Draw samples from a uniform distribution.
    """

    if high < low:
        raise ValueError("high must not be less than low")

    u = _draw_and_reshape(size, rand)
    return u * (high - low) + low


def _exponential_internal(scale: float,
                          n: int,
                          antithetic: bool = False) -> ndarray:
    u = rand(n)
    u = minimum(u, 1.0 - np.finfo(np.float32).eps)
    x: ndarray = log(1.0 - u) * (-scale)
    return x


def exponential(scale: float=1.0,
                size: tp.Optional[SIZE_TYPE] = None,
                antithethic: bool = False) -> ndarray:
    return _draw_and_reshape(size,
                             lambda n: _exponential_internal(
                                            scale=scale,
                                            n=n,
                                            antithetic=antithethic))


def standard_exponential(size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return exponential(size=size)


################################################################################
# gamma random number generator by Marsaglia and Tsang
# using Cocos vectorization
################################################################################

def gamma_rand_marsaglia_and_tsang_arrayfire(alpha: float,
                                             lambda_: float,
                                             n: int) \
        -> af.array:
    random_numbers = af.constant(0, n, dtype=Dtype.f32)
    # Gamma(alpha, lambda) generator using Marsaglia and Tsang method
    # Algorithm 4.33
    if alpha >= 1.0:
        d = alpha - 1 / 3
        c = 1.0 / np.sqrt(9.0 * d)

        number_generated = 0
        number_generated_total = 0

        while number_generated < n:
            number_left = n - number_generated

            z = af.randn(number_left, dtype=Dtype.f32)
            y = (1.0 + c * z)
            v = y * y * y

            accept_index_1 = ((z >= -1.0 / c) & (v > 0.0))
            z_accept_1 = z[accept_index_1]
            # del z
            v_accept_1 = v[accept_index_1]
            # del v
            u_accept_1 = af.randu(v_accept_1.elements(), dtype=Dtype.f32)
            # del U

            accept_index_2 = \
                u_accept_1 < af.exp((0.5 * z_accept_1 * z_accept_1 + d - d * v_accept_1 + d * af.log(v_accept_1)))

            x_accept = d * v_accept_1[accept_index_2] / lambda_
            number_accept = x_accept.elements()

            random_numbers[number_generated:np.minimum(n, number_generated + number_accept)] = \
                x_accept[0:np.minimum(number_left, number_accept)]

            number_generated += number_accept
            number_generated_total += number_left

        if GPUOptions.verbose:
            print(f"Acceptance ratio = {n/number_generated_total}")
    else:
        random_numbers = gamma_rand_marsaglia_and_tsang_arrayfire(alpha + 1, lambda_, n)
        random_numbers *= af.randu(n, dtype=Dtype.f32) ** (1.0 / alpha)

    return random_numbers


def gamma(shape: float,
          scale: float = 1.0,
          size: tp.Optional[SIZE_TYPE] = None) \
        -> ndarray:
    def fun(n: int):
        return ndarray(gamma_rand_marsaglia_and_tsang_arrayfire(
                            alpha=shape,
                            lambda_=1.0/scale, n=n))

    return _draw_and_reshape(size, lambda n: fun(n))


def standard_gamma(shape: float,
                   size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return gamma(shape, size=size)


def chisquare(df, size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return gamma(df / 2.0, 2.0, size)


def _beta_internal(a: float,
                   b: float,
                   n: int) -> ndarray:
    X = gamma(a, 1.0, n)
    Y = gamma(b, 1.0, n)
    return X / (X + Y)


def beta(a: float,
         b: float,
         size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return _draw_and_reshape(size, lambda n: _beta_internal(a, b, n))


def _wald_internal(mu: float,
                   LAMBDA: float,
                   n: int) -> ndarray:
    v = randn(n)
    u = rand(n)

    y = v * v
    del v
    x = mu + mu ** 2 / (2.0 * LAMBDA) * y - mu / (2.0 * LAMBDA) * sqrt(4.0 * mu * LAMBDA * y + mu ** 2.0 * y * y)
    reject_index = u > (mu / (mu + x))
    x[reject_index] = mu ** 2 / x[reject_index]
    return x


def wald(mean: float,
         scale: float,
         size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return _draw_and_reshape(size, lambda n: _wald_internal(mean, scale, n))


def normal(loc: float = 0.0,
           scale: float = 1.0,
           size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return _draw_and_reshape(size, lambda n: loc + scale * randn(n))


def standard_normal(size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return _draw_and_reshape(size, randn)


def lognormal(mean: float = 0.0,
              sigma: float = 1.0,
              size: tp.Optional[SIZE_TYPE] = None) -> ndarray:
    return exp(normal(mean, sigma, size))


def _logistic_internal(loc: float,
                       scale: float,
                       n: int) -> ndarray:
    u = rand(n)
    u = minimum(u, 1.0 - np.finfo(np.float32).eps)
    x: ndarray = loc - scale * log(1.0 / u - 1.0)
    return x


def logistic(loc: float = 0.0,
             scale: float = 1.0,
             size: tp.Optional[SIZE_TYPE] = None):
    return _draw_and_reshape(size, lambda n: _logistic_internal(loc, scale, n))


def multivariate_normal(mean, cov, size: tp.Sequence[int]) \
        -> ndarray:
    d = len(mean)
    if not isinstance(size, collections.abc.Iterable):
        size = [size]

    if not isinstance(size, collections.abc.Sequence):
        size = list(size)

    if not cov.shape == (d, d):
        raise ValueError('mean and cov must be a arrays with shapes (d, ) and '
                         '(d, d) respectively')

    draw_shape = list(size)
    draw_shape.append(d)
    n = int(np.prod(size))
    if mean.dtype != cov.dtype:
        raise ValueError('the dtypes of mean and cov mjust match')

    z = randn(n, d, dtype=cov.dtype)
    cov_cholesky = cholesky(cov).T
    x = z @ cov_cholesky

    return x.reshape(draw_shape)
