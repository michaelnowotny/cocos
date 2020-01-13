import time
import typing as tp

from cocos.device import ComputeDeviceManager, ComputeDevice, sync
from loky import get_reusable_executor, wait, as_completed

ResultType = tp.TypeVar('ResultType')
ParameterTransferFunction = tp.Callable[[tp.Sequence,
                                     tp.Dict[str, tp.Any]],
                                    tp.Tuple[tp.Sequence,
                                             tp.Dict[str, tp.Any]]]


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
    name containts 'Intel'.
    """

    return 'intel' not in compute_device.name.lower()


class ComputeDevicePool:
    def __init__(self,
                 compute_devices:
                 tp.Optional[tp.Iterable[tp.Union[int, ComputeDevice]]] = None,
                 compute_device_filter:
                 tp.Optional[ComputeDeviceFilter] = exclude_intel_devices) \
            -> None:
        """
        This method constructs a compute device pool from a collection of
        individual devices.

        :param compute_devices: a collection of device ids or compute devices
        :param compute_device_filter: provide a predicate used to filter devices 
                                      to include in the pool
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

        self._executor = get_reusable_executor(max_workers=self.number_of_devices,
                                               timeout=None,
                                               context='spawn')

        futures = [self._executor.submit(_init_gpu_in_process,
                                         device_id=compute_device.id)
                   for compute_device
                   in self._compute_devices]

        wait(futures)

        [future.result() for future in futures]

    @property
    def compute_devices(self) -> tp.FrozenSet[ComputeDevice]:
        return self._compute_devices

    @property
    def number_of_devices(self) -> int:
        return len(self.compute_devices)

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
        'ResultType'. 
    
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

        if number_of_batches is None:
            if args_list is not None:
                number_of_batches = len(args_list)
            elif kwargs_list is not None:
                number_of_batches = len(kwargs_list)
            else:
                raise ValueError('Number_of_batches must be defined if '
                                 'both args_list and kwargs_list are empty')

        if args_list is None:
            args_list = [list() for i in range(number_of_batches)]
        if kwargs_list is None:
            kwargs_list = number_of_batches * [dict()]

        def synced_f(*args, **kwargs) -> ResultType:
            if host_to_device_transfer_function is not None:
                args, kwargs = host_to_device_transfer_function(*args, **kwargs)
            sync()
            result = f(*args, **kwargs)
            if device_to_host_transfer_function is not None:
                result = device_to_host_transfer_function(result)
            sync()
            return result

        futures = [self._executor.submit(synced_f, *args, **kwargs)
                   for i, (args, kwargs)
                   in enumerate(zip(args_list, kwargs_list))]

        result = initial_value
        for future in as_completed(futures):
            result = reduction(result, future.result())

        return result