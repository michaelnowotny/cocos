import typing as tp

from cocos.multi_processing.utilities import (
    ResultType,
    MultiprocessingPoolType,
    _extract_arguments_and_number_of_batches
)


def map_reduce_multicore(
        f: tp.Callable[..., ResultType],
        reduction: tp.Callable[[ResultType, ResultType], ResultType],
        initial_value: ResultType,
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None,
        multiprocessing_pool_type: MultiprocessingPoolType = MultiprocessingPoolType.default()) \
        -> ResultType:
    """
    This function evaluates the function 'f' on elements of 'args_list' and
    'kwargs_list' in parallel on multiple cpu cores and performs the reduction
    by calling the function 'reduction' on the result and the result of the
    reductions so far to eventually produce one final result of type
    'ResultType'. The reduce step is performed from the left and results are
    being processed in the same order as they appear in `args_list` and
    `kwargs_list`.

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

        args_list: A sequence of sequences of positional arguments.
        kwargs_list: A sequence of dictionaries of keyword arguments.
        number_of_batches:
            The number of function evaluations is required if 'args_list'
            and 'kwargs_list' are both empty.

        multiprocessing_pool_type:
            the type of multi-processing pool (see class MultiprocessingPoolType)
    """
    args_list, kwargs_list, number_of_batches = \
        _extract_arguments_and_number_of_batches(
            args_list=args_list,
            kwargs_list=kwargs_list,
            number_of_batches=number_of_batches)

    def wrapped_f(index, *args, **kwargs) -> ResultType:
        return index, f(*args, **kwargs)

    if multiprocessing_pool_type == MultiprocessingPoolType.LOKY:
        from concurrent.futures import as_completed
        from loky import get_reusable_executor

        executor = \
            get_reusable_executor(timeout=None,
                                  context='loky')

        futures = [executor.submit(wrapped_f, i, *args, **kwargs)
                   for i, (args, kwargs)
                   in enumerate(zip(args_list, kwargs_list))]

        result_from_future = lambda x: x.result()
    elif multiprocessing_pool_type == MultiprocessingPoolType.PATHOS:
        from pathos.pools import ProcessPool
        pool = ProcessPool()

        futures = [pool.apipe(wrapped_f, i, *args, **kwargs)
                   for i, (args, kwargs)
                   in enumerate(zip(args_list, kwargs_list))]

        result_from_future = lambda x: x.get()
    else:
        raise ValueError(f'Multiprocessing pool type {multiprocessing_pool_type} not supported')

    results = [result_from_future(future) for future in futures]
    results = sorted(results, key=lambda x: x[0])
    results = [result[1] for result in results]

    result = initial_value
    for new_result in results:
        result = reduction(result, new_result)

    return result


def map_combine_multicore(
        f: tp.Callable[..., ResultType],
        combination: tp.Callable[[tp.Iterable[ResultType]], ResultType],
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None,
        multiprocessing_pool_type: MultiprocessingPoolType = MultiprocessingPoolType.default()) \
        -> ResultType:
    """
    This function evaluates the function `f` on elements of `args_list` and
    `kwargs_list` in parallel on multiple cpu cores and aggregates results
    in a single step by calling the function `combination` with a list of all
    results. Results provided to `combination` are in the same order as
    they appear in `args_list` and `kwargs_list`.

    If the arguments for each run of 'f' are identical and they have already
    been applied to the function that is passed then 'args_list' and
    'kwargs_list' may both be None but the argument 'number_of_batches' must
    be specified so the method knows how many times to run the function 'f'.

    Args:
        f: The map function to be evaluated over elements of 'args_list' and
               'kwargs_list'.

        combination: A function that aggregates a list of all results in a single step
        args_list: A sequence of sequences of positional arguments.
        kwargs_list: A sequence of dictionaries of keyword arguments.
        number_of_batches:
            The number of function evaluations is required if 'args_list'
            and 'kwargs_list' are both empty.

        multiprocessing_pool_type:
            the type of multi-processing pool (see class MultiprocessingPoolType)
    """
    args_list, kwargs_list, number_of_batches = \
        _extract_arguments_and_number_of_batches(
            args_list=args_list,
            kwargs_list=kwargs_list,
            number_of_batches=number_of_batches)

    def wrapped_f(index, *args, **kwargs) -> ResultType:
        return index, f(*args, **kwargs)

    results = []
    if multiprocessing_pool_type == MultiprocessingPoolType.LOKY:
        from concurrent.futures import as_completed
        from loky import get_reusable_executor

        executor = \
            get_reusable_executor(timeout=None,
                                  context='loky')

        futures = [executor.submit(wrapped_f, i, *args, **kwargs)
                   for i, (args, kwargs)
                   in enumerate(zip(args_list, kwargs_list))]

        for future in as_completed(futures):
            results.append(future.result())
    elif multiprocessing_pool_type == MultiprocessingPoolType.PATHOS:
        from pathos.pools import ProcessPool
        pool = ProcessPool()

        futures = [pool.apipe(wrapped_f, i, *args, **kwargs)
                   for i, (args, kwargs)
                   in enumerate(zip(args_list, kwargs_list))]

        for future in futures:
            results.append(future.get())
    else:
        raise ValueError(f'Multiprocessing pool type {multiprocessing_pool_type} not supported')

    results = sorted(results, key=lambda x: x[0])
    results = [result[1] for result in results]

    # print(f'len(results)={len(results)}')
    # for result in results:
    #     print(result.shape)

    return combination(results)


# class MultiCoreMapReduce:
#     def __init__(self, number_of_cores: tp.Optional[int] = None):
#         self._number_of_cores = number_of_cores
