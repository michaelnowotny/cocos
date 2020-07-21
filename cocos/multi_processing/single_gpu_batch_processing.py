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
    """
    This method function the function 'f' on elements of 'args_list' and 
    'kwargs_list' sequentially on a single device and performs the reduction 
    by calling the function 'reduction' on the result and the result of the 
    reductions so far to eventually produce one final result of type 
    'ResultType'. The reduce step is performed from the left and results are 
    being processed in the same order as they appear in `args_list` and 
    `kwargs_list`. 

    Input data to the function f must initially reside in host memory and 
    the user must provide functions 'host_to_device_transfer_function' and 
    'device_to_host_transfer_function' to transfer the data to and results 
    from device memory respectively.

    If the arguments for each run of 'f' are identical and they have already 
    been applied to the function that is passed then 'args_list' and 
    'kwargs_list' may both be None but the argument 'number_of_batches' must 
    be specified so the method knows how many times to run the function 'f'.
    
    Args:
        f: The map function to be evaluated over elements of 'args_list' and 
           'kwargs_list'.
           
        reduction: The reduction to be performed on the results of 'f'. 
                   This is done on the host (not the device).
                   
        initial_value: The initial value of the reduction 
                       (i.e. the neutral element).
                       
        host_to_device_transfer_function: 
            A function that transfers elements of args_list and kwargs_list 
            from host memory to device memory.
            
        device_to_host_transfer_function: 
            A function that transfers results from device to host memory.
            
        args_list: A sequence of sequences of positional arguments.
        kwargs_list: A sequence of dictionaries of keyword arguments.
        number_of_batches: 
            The number of function evaluations is required if 'args_list' 
            and 'kwargs_list' are both empty.
    """
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
    """
    This function evaluates the function `f` on elements of `args_list` and 
    `kwargs_list` sequentially on a single device and aggregates results 
    in a single step by calling the function `combination` with a list of all 
    results. Results provided to `combination` are in the same order as 
    they appear in `args_list` and `kwargs_list`. 
    
    Input data to the function f must initially reside in host memory and 
    the user must provide functions 'host_to_device_transfer_function' and 
    'device_to_host_transfer_function' to transfer the data to and results 
    from device memory respectively.
    
    If the arguments for each run of 'f' are identical and they have already 
    been applied to the function that is passed then 'args_list' and 
    'kwargs_list' may both be None but the argument 'number_of_batches' must 
    be specified so the method knows how many times to run the function 'f'.
    
    Args:
        f: The map function to be evaluated over elements of 'args_list' and 
           'kwargs_list'.
           
        combination: 
            A function that aggregates a list of all results in a single step
            
        host_to_device_transfer_function: 
            A function that transfers elements of args_list and kwargs_list 
            from host memory to device memory.
            
        device_to_host_transfer_function:
             A function that transfers results from device to host memory.
             
        args_list: A sequence of sequences of positional arguments.
        kwargs_list: A sequence of dictionaries of keyword arguments.
        number_of_batches: 
            The number of function evaluations is required if 'args_list' 
            and 'kwargs_list' are both empty.
    """
    args_list, kwargs_list, number_of_batches = \
        _extract_arguments_and_number_of_batches(
            args_list=args_list,
            kwargs_list=kwargs_list,
            number_of_batches=number_of_batches)

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
