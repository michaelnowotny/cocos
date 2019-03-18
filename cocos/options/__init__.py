from enum import Enum


class MixedComputationErrorLevel(Enum):
    NONE = 1
    WARNING = 2
    ERROR = 3


class RandomNumberGenerator(Enum):
    PHILOX_4X32_10 = 1
    THREEFRY_2X32_16 = 2
    MERSENNE_GP11213 = 3
    PHILOX = 1
    THREEFRY = 2
    DEFAULT = 1


class GPUOptions:
    def __init__(self):
        print("running GPUOptions.__init__")
    default_rng = RandomNumberGenerator.DEFAULT
    mixed_computation_error_level = MixedComputationErrorLevel.ERROR
    str_via_numpy = True
    use_gpu = True
    verbose = False
