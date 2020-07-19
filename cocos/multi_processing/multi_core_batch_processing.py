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
    args_list, kwargs_list, number_of_batches = \
        _extract_arguments_and_number_of_batches(
            args_list=args_list,
            kwargs_list=kwargs_list,
            number_of_batches=number_of_batches)

    # if number_of_batches is None:
    #     if args_list is not None:
    #         number_of_batches = len(args_list)
    #     elif kwargs_list is not None:
    #         number_of_batches = len(kwargs_list)
    #     else:
    #         raise ValueError('Number_of_batches must be defined if '
    #                          'both args_list and kwargs_list are empty')
    #
    # if args_list is None:
    #     args_list = number_of_batches * [list()]
    # if kwargs_list is None:
    #     kwargs_list = number_of_batches * [dict()]

    result = initial_value
    if multiprocessing_pool_type == MultiprocessingPoolType.LOKY:
        from concurrent.futures import as_completed
        from loky import get_reusable_executor

        executor = \
            get_reusable_executor(timeout=None,
                                  context='loky')

        futures = [executor.submit(f, *args, **kwargs)
                   for args, kwargs
                   in zip(args_list, kwargs_list)]

        result_from_future = lambda x: x.result()
    elif multiprocessing_pool_type == MultiprocessingPoolType.PATHOS:
        from pathos.pools import ProcessPool
        pool = ProcessPool()

        futures = [pool.apipe(f, *args, **kwargs)
                   for args, kwargs
                   in zip(args_list, kwargs_list)]

        result_from_future = lambda x: x.get()
    else:
        raise ValueError(f'Multiprocessing pool type {multiprocessing_pool_type} not supported')

    for future in futures:
        result = reduction(result, result_from_future(future))

    return result


def map_combine_multicore(
        f: tp.Callable[..., ResultType],
        combination: tp.Callable[[tp.Iterable[ResultType]], ResultType],
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None,
        multiprocessing_pool_type: MultiprocessingPoolType = MultiprocessingPoolType.default()) \
        -> ResultType:
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
