"""Cocos: scientific GPU computing in Python, built on ArrayFire."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("cocos")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

MINIMUM_ARRAYFIRE_VERSION = (3, 10, 0)

_INSTALLATION_HELP = ("See https://github.com/michaelnowotny/cocos#installation "
                      "for installation instructions.")

try:
    import arrayfire as _af
except RuntimeError as error:  # the wrapper raises this when no library loads
    raise ImportError(
        "Cocos could not load the ArrayFire library. Install ArrayFire "
        f"{'.'.join(map(str, MINIMUM_ARRAYFIRE_VERSION[:2]))} or newer and, if "
        "it is not installed in /opt/arrayfire, set the environment variable "
        "AF_PATH to its installation directory. " + _INSTALLATION_HELP) \
        from error

ARRAYFIRE_VERSION = tuple(_af.get_version())

if ARRAYFIRE_VERSION < MINIMUM_ARRAYFIRE_VERSION:
    raise ImportError(
        f"Cocos {__version__} requires ArrayFire "
        f"{'.'.join(map(str, MINIMUM_ARRAYFIRE_VERSION))} or newer, but found "
        f"ArrayFire {'.'.join(map(str, ARRAYFIRE_VERSION))}. Older versions "
        "cannot compile kernels for recent NVIDIA GPUs and compute variances "
        "differently. Upgrade ArrayFire, or keep the older ArrayFire with "
        "cocos 0.2.2. " + _INSTALLATION_HELP)
