"""
pytest plugin for the cocos test suite: choose the ArrayFire backend and seed
the random number generators.

Select the backend with ``pytest --backend cuda`` (or cpu, opencl), or with the
environment variable COCOS_TEST_BACKEND. Without either, ArrayFire picks the
first available backend in the order CUDA, OpenCL, CPU.

pyproject.toml loads this plugin for runs from the repository. To test an
installed copy of cocos, load it explicitly:

    pytest -p cocos.tests.pytest_plugin --pyargs cocos.tests --backend cuda
"""

import os

import arrayfire as af
import numpy as np
import pytest

import cocos
import cocos.device
import cocos.numerics as cn

BACKENDS = ("cpu", "cuda", "opencl")


def pytest_addoption(parser):
    parser.addoption("--backend",
                     choices=BACKENDS,
                     default=os.environ.get("COCOS_TEST_BACKEND") or None,
                     help="ArrayFire backend to run the tests on "
                          "(default: $COCOS_TEST_BACKEND or ArrayFire's choice)")


def pytest_configure(config):
    backend = config.getoption("--backend")
    if backend:
        # Unlike cocos.device.init, this fails if the backend is unavailable.
        af.set_backend(backend)


def pytest_report_header(config):
    return f"cocos {cocos.__version__} on {cocos.device.build_and_backend()}"


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    """
    Make every test that draws random numbers reproducible. Set the
    environment variable COCOS_TEST_SEED to use a seed other than 1.
    """
    seed = int(os.environ.get("COCOS_TEST_SEED", "1"))
    cn.random.seed(seed)
    np.random.seed(seed)
