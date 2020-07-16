from enum import Enum

try:
    import pathos
    _IS_PATHOS_INSTALLED = True
except:
    _IS_PATHOS_INSTALLED = False


try:
    import loky
    _IS_LOKY_INSTALLED = True
except:
    _IS_LOKY_INSTALLED = False


def is_loky_installed() -> bool:
    return _IS_LOKY_INSTALLED


def is_pathos_installed() -> bool:
    return _IS_PATHOS_INSTALLED


class MultiprocessingPoolType(Enum):
    LOKY = 1
    PATHOS = 2

    @staticmethod
    def default() -> 'MultiprocessingPoolType':
        if is_pathos_installed():
            return MultiprocessingPoolType.PATHOS
        elif is_loky_installed():
            return MultiprocessingPoolType.LOKY
