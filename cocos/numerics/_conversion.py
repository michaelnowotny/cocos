import numpy as np
from arrayfire import Dtype


def convert_numpy_to_af_type(numpy_type):
    if numpy_type == np.float32:
        return Dtype.f32
    elif numpy_type == np.complex64:
        return Dtype.c32
    elif numpy_type == np.float64:
        return Dtype.f64
    elif numpy_type == np.complex128:
        return Dtype.c64
    elif numpy_type == np.bool8 or \
            numpy_type == np.bool or \
            numpy_type == np.bool_:
        return Dtype.b8
    elif numpy_type == np.int32:
        return Dtype.s32
    elif numpy_type == np.uint32:
        return Dtype.u32
    elif numpy_type == np.uint8:
        return Dtype.u8
    elif numpy_type == np.int64:
        return Dtype.s64
    elif numpy_type == np.uint64:
        return Dtype.u64
    elif numpy_type == np.int or numpy_type == int:
        return Dtype.s32
    elif numpy_type == np.long:
        return Dtype.s64
    elif numpy_type == np.float or numpy_type == float:
        return Dtype.f32
    elif numpy_type == np.complex or numpy_type == complex:
        return Dtype.c32
    elif numpy_type == np.double:
        return Dtype.f64
    else:
        raise ValueError(f"Type {numpy_type} not supported.")


def convert_af_to_numpy_type(af_type):
    if af_type == Dtype.f32:
        return np.float32
    elif af_type == Dtype.c32:
        return np.complex64
    elif af_type == Dtype.f64:
        return np.float64
    elif af_type == Dtype.c64:
        return np.complex128
    elif af_type == Dtype.b8:
        return np.bool8
    # elif (af_type == Dtype.b8):
    #     return bool
    elif af_type == Dtype.s32:
        return np.int32
    elif af_type == Dtype.u32:
        return np.uint32
    elif af_type == Dtype.u8:
        return np.uint8
    elif af_type == Dtype.s64:
        return np.int64
    elif af_type == Dtype.u64:
        return np.uint64
    else:
        raise ValueError(f"Type {af_type} not supported.")
