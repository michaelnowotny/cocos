import typing as tp

from cocos.device import sync
from cocos.multi_processing.utilities import (
    MultiprocessingPoolType,
    ResultType,
    ParameterTransferFunction,
    _extract_arguments_and_number_of_batches
)


def map_reduce_single_gpu(
        f: tp.Callable[..., ResultType],
        reduction: tp.Callable[[ResultType, ResultType], ResultType],
        initial_value: ResultType,
        host_to_device_transfer_function:
        tp.Optional[ParameterTransferFunction] = None,
        device_to_host_transfer_function:
        tp.Optional[tp.Callable[[ResultType], ResultType]] = None,
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None) \
        -> ResultType:
    args_list, kwargs_list, number_of_batches = \
        _extract_arguments_and_number_of_batches(
            args_list=args_list,
            kwargs_list=kwargs_list,
            number_of_batches=number_of_batches)

    result = initial_value

    for args, kwargs in zip(args_list, kwargs_list):
        if host_to_device_transfer_function is not None:
            args, kwargs = host_to_device_transfer_function(*args, **kwargs)
        sync()
        new_part = f(*args, **kwargs)
        if device_to_host_transfer_function is not None:
            new_part = device_to_host_transfer_function(new_part)
        sync()

        result = reduction(result, new_part)

    return result


def map_combine_single_gpu(
        f: tp.Callable[..., ResultType],
        combination: tp.Callable[[tp.Iterable[ResultType]], ResultType],
        host_to_device_transfer_function:
        tp.Optional[ParameterTransferFunction] = None,
        device_to_host_transfer_function:
        tp.Optional[tp.Callable[[ResultType], ResultType]] = None,
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None) \
        -> ResultType:
    args_list, kwargs_list, number_of_batches = \
        _extract_arguments_and_number_of_batches(
            args_list=args_list,
            kwargs_list=kwargs_list,
            number_of_batches=number_of_batches)

    # def synced_f(index, *args, **kwargs) -> ResultType:
    #     return index, f(*args, **kwargs)

    results = []

    for args, kwargs in zip(args_list, kwargs_list):
        if host_to_device_transfer_function is not None:
            args, kwargs = host_to_device_transfer_function(*args, **kwargs)
        sync()
        result = f(*args, **kwargs)
        if device_to_host_transfer_function is not None:
            result = device_to_host_transfer_function(result)
        sync()
        results.append(result)

    return combination(results)
