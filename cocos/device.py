import arrayfire as af
import typing as tp


class DeviceMemoryInfo:
    def __init__(self,
                 allocated_buffers: int,
                 allocated_bytes: int,
                 locked_buffers: int,
                 locked_bytes: int) -> None:
        self._allocated_buffers = allocated_buffers
        self._allocated_bytes = allocated_bytes
        self._locked_buffers = locked_buffers
        self._locked_bytes = locked_bytes

    @property
    def allocated_buffers(self) -> int:
        return self._allocated_buffers

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    @property
    def locked_buffers(self) -> int:
        return self._locked_buffers

    @property
    def locked_bytes(self) -> int:
        return self._locked_bytes

    def __str__(self):
        return f"allocated buffers = {self.allocated_buffers}\n" \
               f"allocated bytes = {self.allocated_bytes}\n" \
               f"locked buffers = {self.locked_buffers}\n" \
               f"locked bytes = {self.locked_bytes}"


class ComputeDevice:
    def __init__(self,
                 id: int,
                 name: str,
                 backend: str,
                 toolkit_version: str,
                 compute_version: str) -> None:
        self._id = id
        self._name = name
        self._backend = backend
        self._toolkit_version = toolkit_version
        self._compute_version = compute_version

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def toolkit_version(self) -> str:
        return self._toolkit_version

    @property
    def compute_version(self) -> str:
        return self._compute_version

    def sync(self):
        sync(self.id)

    def __str__(self):
        return f"{self.id}: {self.name} | " \
               f"{self.backend} - {self.toolkit_version} | " \
               f"compute version {self.compute_version}"


def get_current_device_memory_info() \
        -> DeviceMemoryInfo:
    device_mem_info = af.device_mem_info()
    allocated_buffers = device_mem_info['alloc']['buffers']
    allocated_bytes = device_mem_info['alloc']['bytes']
    locked_buffers = device_mem_info['lock']['buffers']
    locked_bytes = device_mem_info['lock']['bytes']

    return DeviceMemoryInfo(allocated_buffers=allocated_buffers,
                            allocated_bytes=allocated_bytes,
                            locked_buffers=locked_buffers,
                            locked_bytes=locked_bytes)


class ComputeDevicePool:
    def __init__(self,
                 compute_devices: tp.Sequence[tp.Union[int, ComputeDevice]]) \
            -> None:
        self._compute_devices = []

        for i, compute_device in enumerate(compute_devices):
            if isinstance(compute_device, int):
                compute_device \
                    = ComputeDeviceManager.get_compute_device(compute_device)
            elif not isinstance(compute_device, ComputeDevice):
                raise TypeError(f"Every element in compute_devices must be of "
                                f"type ComputeDevice or of type int. Entry {i} "
                                f"is of type {type(compute_device)}")

            self._compute_devices.append(compute_device)

        self._compute_devices: tp.Tuple[ComputeDevice] \
            = tuple(self._compute_devices)

    def compute_devices(self) -> tp.Tuple[ComputeDevice]:
        return self._compute_devices

    def sync(self):
        for compute_device in self._compute_devices:
            compute_device.sync()


class ComputeDeviceManager:
    _compute_devices = None

    @staticmethod
    def _get_compute_device_internal(id: int) -> ComputeDevice:
        af.set_device(id)
        device_info = af.device_info()
        name = device_info['device']
        backend = device_info['backend']
        toolkit_version = device_info['toolkit']
        compute_version = device_info['compute']

        return ComputeDevice(id,
                             name,
                             backend,
                             toolkit_version,
                             compute_version)

    @staticmethod
    def get_number_of_compute_devices() -> int:
        return len(ComputeDeviceManager.get_compute_devices())

    @classmethod
    def get_compute_devices(cls) -> tp.Sequence[ComputeDevice]:
        if ComputeDeviceManager._compute_devices is None:
            saved_device_id = cls.get_current_compute_device_id()
            n = af.get_device_count()
            ComputeDeviceManager._compute_devices = []

            for id in range(n):
                (ComputeDeviceManager
                 ._compute_devices
                 .append(ComputeDeviceManager._get_compute_device_internal(id)))

            af.set_device(saved_device_id)

        return ComputeDeviceManager._compute_devices

    @staticmethod
    def get_compute_devices_by_name(name_contains: str) \
            -> tp.Sequence[ComputeDevice]:
        compute_devices = ComputeDeviceManager.get_compute_devices()
        return [compute_device for compute_device in compute_devices
                if name_contains in compute_device.name]

    @staticmethod
    def get_compute_device(id: int) -> ComputeDevice:
        return ComputeDeviceManager.get_compute_devices()[id]

    @classmethod
    def get_current_compute_device_id(cls) -> int:
        return af.get_device()

    @classmethod
    def get_current_compute_device(cls) -> ComputeDevice:
        return cls.get_compute_device(cls.get_current_compute_device_id())

    @staticmethod
    def set_compute_device(compute_device: tp.Union[int, ComputeDevice]) \
            -> None:
        if isinstance(compute_device, int):
            compute_device \
                = ComputeDeviceManager.get_compute_device(compute_device)
        elif not isinstance(compute_device, ComputeDevice):
            raise TypeError(f"The argument compute_device must be of "
                            f"type ComputeDevice or of type int. The argument "
                            f"provided is of type {type(compute_device)}")

        af.set_device(compute_device.id)


def init(backend: tp.Optional[str] = None):
    try:
        if backend:
            af.set_backend(backend)
        af.get_device()
    except:
        af.set_backend('cpu')


def info():
    print(build_and_backend())
    selected_device_id = ComputeDeviceManager.get_current_compute_device_id()
    for compute_device in ComputeDeviceManager.get_compute_devices():
        device_id = compute_device.id
        if device_id == selected_device_id:
            device_string = "[" + str(device_id) + "]"
        else:
            device_string = "-" + str(device_id) + "-"
        device_name = compute_device.name.replace('_', ' ')
        print(
            f'{device_string} {compute_device.toolkit_version}: {device_name} '
            f'| {compute_device.backend} '
            f'| compute version {compute_device.compute_version}')


def build_and_backend() -> str:
    return af.info_str().split('\n')[0]


def sync(compute_device: tp.Optional[tp.Union[int, ComputeDevice]] = None):
    if isinstance(compute_device, ComputeDevice):
        device = compute_device.id
    elif not compute_device or isinstance(compute_device, int):
        device = compute_device
    else:
        raise TypeError(f"The argument compute_device must be of "
                        f"type ComputeDevice or of type int. The argument "
                        f"provided is of type {type(compute_device)}")

    af.sync(device=device)


def is_dbl_supported(
        compute_device: tp.Optional[tp.Union[int, ComputeDevice]] = None) \
        -> bool:
    if isinstance(compute_device, ComputeDevice):
        device = compute_device.id
    elif not compute_device or isinstance(compute_device, int):
        device = compute_device
    else:
        raise TypeError(f"The argument compute_device must be of "
                        f"type ComputeDevice or of type int. The argument "
                        f"provided is of type {type(compute_device)}")

    return af.is_dbl_supported(device)


def selected_backend() -> str:
    device_info = af.device_info()
    return device_info['backend'].lower()
