import math
import typing as tp
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
        if is_loky_installed():
            return MultiprocessingPoolType.LOKY
        elif is_pathos_installed():
            return MultiprocessingPoolType.PATHOS
        else:
            raise ValueError('No suitable multiprocessing package found. '
                             'Please install either loky or pathos.')


def generate_slices_with_batch_size(n: int, batch_size: int) \
        -> tp.Tuple[tp.Tuple[int, int]]:
    """
    Splits range(0, n) into a partition of sub-ranges with length <= batch_size.

    Args:
        n: length of the source range
        batch_size: maximal length of each range in the partition

    Returns:
        tuple of two-element tuples following the pattern (begin_index, end_index)
    """
    begin_index = 0
    result = []

    while begin_index < n:
        end_index = min(begin_index + batch_size, n)
        result.append((begin_index, end_index))
        begin_index += batch_size

    return tuple(result)


def generate_slices_with_number_of_batches(n: int, number_of_batches: int) \
        -> tp.Tuple[tp.Tuple[int, int]]:
    """
    Splits range(0, n) into a partition of number_of_batches sub-ranges of equal length except for the last one.

    Args:
        n: length of the source range
        number_of_batches: the number of sub-ranges in the partition

    Returns:
        tuple of two-element tuples following the pattern (begin_index, end_index)
    """
    batch_size = math.ceil(n/number_of_batches)
    return generate_slices_with_batch_size(n=n, batch_size=batch_size)


ResultType = tp.TypeVar('ResultType')

ParameterTransferFunction = tp.Callable[[tp.Sequence, tp.Dict[str, tp.Any]],
                                        tp.Tuple[tp.Sequence, tp.Dict[str, tp.Any]]]


def _extract_arguments_and_number_of_batches(
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None):
    if number_of_batches is None:
        if args_list is not None:
            number_of_batches = len(args_list)
        elif kwargs_list is not None:
            number_of_batches = len(kwargs_list)
        else:
            raise ValueError('Number_of_batches must be defined if '
                             'both args_list and kwargs_list are empty')

    if args_list is None:
        args_list = number_of_batches * [list()]
    if kwargs_list is None:
        kwargs_list = number_of_batches * [dict()]

    return args_list, kwargs_list, number_of_batches
