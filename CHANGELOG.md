# Changelog

## 0.3.0 (2026-10-25)

Cocos 0.3 brings cocos up to date with current GPUs and Python. It requires
ArrayFire 3.10, which supports current NVIDIA GPUs, as well as Python 3.12 or
newer and NumPy 2. It also fixes several functions that returned wrong
results or failed.

### Fixed: results that change

*   `var` and `std` now match NumPy. With ArrayFire 3.8 or newer, cocos 0.2.2
    returned wrong variances: with the 2018 release of the `arrayfire` Python
    wrapper, `var` returned the sample variance for the default `ddof=0` and
    the population variance for `ddof=1`, because ArrayFire 3.8 reversed the
    meaning of the flag cocos passed. With the wrapper 3.8.0, which pip
    installed for cocos 0.2.2 since 2022, `var` raised a `TypeError`.
    Independently of the ArrayFire version, `std(..., ddof=1)` ignored `ddof`
    and returned the population standard deviation. For complex arrays,
    `var` returned a complex number and `std` raised an error; both now
    return NumPy's real-valued result. `ndarray.std` now also accepts `ddof`.
*   `cov` and `corrcoef` never worked: they raised a `TypeError` or crashed
    the Python interpreter. They now follow NumPy, including the parameters
    `y`, `rowvar`, `bias` and `ddof` (but not `fweights` and `aweights`),
    except that single-precision inputs give single-precision results.
*   `sort_by_keys` sorted by the values instead of the keys and returned the
    two arrays swapped.
*   `cocos.scientific.kde.gaussian_kernel_estimate_vectorized` failed on the
    GPU with an ArrayFire type error, and for one-dimensional data or
    evaluation points. It also no longer prints debugging output.

### New requirements

*   ArrayFire 3.10 or newer. Importing cocos now checks the ArrayFire version
    and explains how to install or upgrade it. ArrayFire 3.10 includes the
    CUDA 12.8 libraries it uses, so no CUDA toolkit is needed. Its CUDA
    backend needs an NVIDIA GPU of the Maxwell generation or newer and an
    NVIDIA driver 525.60 or newer (528.33 on Windows).
*   Python 3.12, 3.13, 3.14 or 3.15.
*   NumPy 2.0, SciPy 1.13, SymPy 1.12 and loky 3.4 or newer, and the
    `arrayfire` Python wrapper 3.8.

### No longer supported

*   ArrayFire 3.9 and older. ArrayFire 3.8 cannot compile kernels for recent
    NVIDIA GPUs such as the RTX 40 series. If you must stay on an older
    ArrayFire, keep using cocos 0.2.2.
*   Python 3.11 and older, and NumPy 1.
*   macOS, for which there is no ArrayFire 3.10 release, and NVIDIA GPUs
    older than the Maxwell generation, which CUDA 12 does not support.
*   Windows is untested. ArrayFire provides a Windows installer and cocos
    should work there, but CI and the release tests run on Linux only.

### Other changes

*   cocos no longer installs pytest, contexttimer and cached_property.
    pathos, an alternative to the default loky for multi-core processing, is
    now an optional extra: `pip install cocos[pathos]`. It does not work on
    Python 3.15 yet because its dependency dill does not. CuPy, used by the
    benchmarks, is available as `cocos[cupy]`.
*   `cocos.device.init` warns when it falls back to the CPU backend instead
    of doing so silently.
*   `cocos.__version__` gives the installed version.
*   cocos arrays support NumPy 2's `copy` and `dtype` arguments when they are
    converted to NumPy arrays.
*   The Docker image is rebuilt on Ubuntu 24.04 with ArrayFire 3.10,
    JupyterLab and the examples' dependencies, and is published as
    `ghcr.io/michaelnowotny/cocos`. It needs no CUDA installation on the host,
    only the NVIDIA driver and the NVIDIA Container Toolkit.
*   The project is built with uv and `pyproject.toml` instead of `setup.py`.
*   The test suite selects a backend with `pytest --backend cuda` (or
    `opencl`, `cpu`) and seeds the random number generators. The tests are
    installed with cocos, so you can check an installation with
    `pytest -p cocos.tests.pytest_plugin --pyargs cocos.tests --backend cuda`.
    Several tests used to pass without checking anything: the chi-square
    and Wald generator tests, the kernel density estimate tests, one slicing
    test and the `sort_by_keys` test. They now check their results.
*   GitHub Actions run the tests on Python 3.12 to 3.15 on the CPU backend.
    Releases are published to PyPI by GitHub Actions.
*   The examples work with current Matplotlib and SciPy.

### Tested with

The release was tested on Ubuntu 24.04 (in Docker) with an NVIDIA RTX 4070 Ti
and driver 580: the full test suite passes on the CUDA, OpenCL and CPU
backends with Python 3.12 to 3.15, and with the lowest supported versions of
the direct dependencies. The examples (except the CuPy ones) and the Getting Started
notebook run without errors.

## 0.2.2 and earlier

See the [commit history](https://github.com/michaelnowotny/cocos/commits/master).
