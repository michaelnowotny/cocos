import time
import typing as tp

from cocos.device import (
    ComputeDeviceManager,
    ComputeDevice,
    sync
)

from cocos.multi_processing.utilities import (
    MultiprocessingPoolType,
    ResultType,
    ParameterTransferFunction,
    _extract_arguments_and_number_of_batches
)


def _init_gpu_in_process(device_id: int):
    # print(f'initializing device {device_id}')
    time.sleep(0.1)
    ComputeDeviceManager.set_compute_device(compute_device=device_id)
    # print(f'device {device_id} initialized')


def _get_set_of_compute_devices_from_iterable(
        compute_devices: tp.Iterable[tp.Union[int, ComputeDevice]]) \
        -> tp.FrozenSet[ComputeDevice]:
    result_compute_devices = []

    for i, compute_device in enumerate(compute_devices):
        if isinstance(compute_device, int):
            compute_device \
                = ComputeDeviceManager.get_compute_device(compute_device)
        elif not isinstance(compute_device, ComputeDevice):
            raise TypeError(f"Every element in compute_devices must be of "
                            f"type ComputeDevice or of type int. Entry {i} "
                            f"is of type {type(compute_device)}.")

        result_compute_devices.append(compute_device)

    return frozenset(result_compute_devices)


ComputeDeviceFilter = tp.Callable[[ComputeDevice], bool]


def exclude_intel_devices(compute_device: ComputeDevice) -> bool:
    """
    Some Intel processors feature a GPU integrated on the chip of the CPU. The
    integrated GPU is typically less performant than discrete GPUs in the
    system. This parameter can be used to automatically exclude any device whose
    name contains 'Intel'.
    """

    return 'intel' not in compute_device.name.lower()


class ComputeDevicePool:
    def __init__(self,
                 compute_devices:
                 tp.Optional[tp.Iterable[tp.Union[int, ComputeDevice]]] = None,
                 compute_device_filter:
                 tp.Optional[ComputeDeviceFilter] = exclude_intel_devices,
                 multiprocessing_pool_type: MultiprocessingPoolType = MultiprocessingPoolType.default()) \
            -> None:
        """
        This method constructs a compute device pool from a collection of
        individual devices.

        :param compute_devices: a collection of device ids or compute devices
        :param compute_device_filter: provide a predicate used to filter devices 
                                      to include in the pool
        :param multiprocessing_pool_type: the type of multi-processing pool 
                                          (see class MultiprocessingPoolType)

        """
        if compute_devices is None:
            compute_devices = ComputeDeviceManager.get_compute_devices()

        self._compute_devices \
            = _get_set_of_compute_devices_from_iterable(compute_devices)

        if compute_device_filter is not None:
            compute_devices = \
                    filter(compute_device_filter,
                           [compute_device
                            for compute_device
                            in self._compute_devices])
            self._compute_devices = frozenset(compute_devices)

        if exclude_intel_devices:
            compute_devices = \
                filter(lambda x: 'intel' not in x.name.lower(),
                       [compute_device
                        for compute_device
                        in self._compute_devices])
            self._compute_devices = frozenset(compute_devices)

        # ctx = multiprocessing.get_context("spawn")
        # self._executor = ProcessPoolExecutor(max_workers=self._n_gpus,
        #                                      mp_context=ctx)

        if multiprocessing_pool_type == MultiprocessingPoolType.LOKY:
            from loky import get_reusable_executor, wait

            self._executor = get_reusable_executor(max_workers=self.number_of_devices,
                                                   timeout=None,
                                                   context='loky')

            futures = [self._executor.submit(_init_gpu_in_process,
                                             device_id=compute_device.id)
                       for compute_device
                       in self._compute_devices]

            wait(futures)

            [future.result() for future in futures]
        elif multiprocessing_pool_type == MultiprocessingPoolType.PATHOS:
            from pathos.pools import ProcessPool

            self._executor = ProcessPool(nodes=self.number_of_devices)
            futures = [self._executor.apipe(_init_gpu_in_process, device_id=compute_device.id)
                       for compute_device
                       in self._compute_devices]

            for future in futures:
                while not future.ready():
                    pass
        else:
            raise ValueError(f'Multiprocessing pool type {multiprocessing_pool_type} not supported')

        self._multiprocessing_pool_type = multiprocessing_pool_type

    @property
    def compute_devices(self) -> tp.FrozenSet[ComputeDevice]:
        return self._compute_devices

    @property
    def number_of_devices(self) -> int:
        return len(self.compute_devices)

    @property
    def multiprocessing_pool_type(self) -> MultiprocessingPoolType:
        return self._multiprocessing_pool_type

    def sync(self):
        for compute_device in self._compute_devices:
            compute_device.sync()

    def map_reduce(
            self,
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
        This method evaluates the function 'f' on elements of 'args_list' and 
        'kwargs_list' in parallel on multiple devices and performs the reduction 
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

        def synced_f(index, *args, **kwargs) -> ResultType:
            if host_to_device_transfer_function is not None:
                args, kwargs = host_to_device_transfer_function(*args, **kwargs)
            sync()
            result = f(*args, **kwargs)
            if device_to_host_transfer_function is not None:
                result = device_to_host_transfer_function(result)
            sync()
            return index, result

        results = []
        if self.multiprocessing_pool_type == MultiprocessingPoolType.LOKY:
            from loky import as_completed

            futures = [self._executor.submit(synced_f, i, *args, **kwargs)
                       for i, (args, kwargs)
                       in enumerate(zip(args_list, kwargs_list))]

            for future in as_completed(futures):
                results.append(future.result())
                # result = reduction(result, future.result())
        elif self.multiprocessing_pool_type == MultiprocessingPoolType.PATHOS:
            futures = [self._executor.apipe(synced_f, i, *args, **kwargs)
                       for i, (args, kwargs)
                       in enumerate(zip(args_list, kwargs_list))]

            for future in futures:
                results.append(future.get())
                # result = reduction(result, future.get())
        else:
            raise ValueError(f'Multiprocessing pool type {self.multiprocessing_pool_type} not supported')

        results = sorted(results, key=lambda x: x[0])
        results = [result[1] for result in results]

        result = initial_value
        for new_result in results:
            result = reduction(result, new_result)

        return result

    def map_combine(self,
                    f: tp.Callable[..., ResultType],
                    combination: tp.Callable[[tp.Iterable[ResultType]], ResultType],
                    host_to_device_transfer_function:
                    tp.Optional[ParameterTransferFunction] = None,
                    device_to_host_transfer_function:
                    tp.Optional[tp.Callable[[ResultType], ResultType]] = None,
                    args_list: tp.Optional[tp.Sequence[tp.Sequence]] = None,
                    kwargs_list: tp.Optional[tp.Sequence[tp.Dict[str, tp.Any]]] = None,
                    number_of_batches: tp.Optional[int] = None) -> ResultType:
        """
        This method evaluates the function `f` on elements of `args_list` and 
        `kwargs_list` in parallel on multiple devices and aggregates results 
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

        def synced_f(index, *args, **kwargs) -> ResultType:
            if host_to_device_transfer_function is not None:
                args, kwargs = host_to_device_transfer_function(*args, **kwargs)
            sync()
            result = f(*args, **kwargs)
            if device_to_host_transfer_function is not None:
                result = device_to_host_transfer_function(result)
            sync()
            return index, result

        results = []
        if self.multiprocessing_pool_type == MultiprocessingPoolType.LOKY:
            from loky import as_completed

            futures = [self._executor.submit(synced_f, i, *args, **kwargs)
                       for i, (args, kwargs)
                       in enumerate(zip(args_list, kwargs_list))]

            for future in as_completed(futures):
                results.append(future.result())
        elif self.multiprocessing_pool_type == MultiprocessingPoolType.PATHOS:
            futures = [self._executor.apipe(synced_f, i, *args, **kwargs)
                       for i, (args, kwargs)
                       in enumerate(zip(args_list, kwargs_list))]

            for future in futures:
                results.append(future.get())
        else:
            raise ValueError(f'Multiprocessing pool type {self.multiprocessing_pool_type} not supported')

        results = sorted(results, key=lambda x: x[0])
        results = [result[1] for result in results]

        return combination(results)
