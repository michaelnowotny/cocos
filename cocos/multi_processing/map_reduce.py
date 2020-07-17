from cocos.multi_processing.utilities import MultiprocessingPoolType, ResultType, reduce_with_none

import typing as tp


def map_reduce_multicore(
        f: tp.Callable[..., ResultType],
        reduction: tp.Callable[[ResultType, ResultType], ResultType],
        initial_value: tp.Optional[ResultType] = None,
        args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
        kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
        number_of_batches: tp.Optional[int] = None,
        multiprocessing_pool_type: MultiprocessingPoolType = MultiprocessingPoolType.default()) \
        -> ResultType:

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

        for future in as_completed(futures):
            result = reduce_with_none(result, future.result(), reduction)

    elif multiprocessing_pool_type == MultiprocessingPoolType.PATHOS:
        from pathos.pools import ProcessPool
        pool = ProcessPool()
        futures = [pool.apipe(f, *args, **kwargs)
                   for args, kwargs
                   in zip(args_list, kwargs_list)]

        for future in futures:
            result = reduce_with_none(result, future.get(), reduction)
    else:
        raise ValueError(f'Multiprocessing pool type {multiprocessing_pool_type} not supported')

    return result


# class MultiCoreMapReduce:
#     def __init__(self, number_of_cores: tp.Optional[int] = None):
#         self._number_of_cores = number_of_cores
