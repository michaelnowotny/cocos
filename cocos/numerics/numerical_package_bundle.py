from abc import ABC, abstractmethod
from types import ModuleType
import typing as tp


class NumericalPackageBundle(ABC):
    @classmethod
    @abstractmethod
    def is_installed(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def label(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def module(cls) -> ModuleType:
        pass

    @classmethod
    @abstractmethod
    def random_module(cls) -> ModuleType:
        pass

    @classmethod
    def synchronize(cls):
        pass


# class AFNumpyBundle(NumericalPackageBundle):
#     @classmethod
#     def is_installed(cls) -> bool:
#         try:
#             import afnumpy
#             return True
#         except:
#             return False
#
#     @classmethod
#     def label(cls) -> str:
#         return 'afnumpy'
#
#     @classmethod
#     def module(cls) -> ModuleType:
#         import afnumpy
#         return afnumpy
#
#     @classmethod
#     def random_module(cls) -> ModuleType:
#         import afnumpy.random
#         return afnumpy.random
#
#     @classmethod
#     def synchronize(cls):
#         from arrayfire import sync
#         sync()


class NumpyBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        try:
            import numpy
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        return 'NumPy'

    @classmethod
    def module(cls) -> ModuleType:
        import numpy
        return numpy

    @classmethod
    def random_module(cls) -> ModuleType:
        import numpy.random
        return numpy.random


class CocosBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        try:
            import cocos
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        return 'Cocos'

    @classmethod
    def module(cls) -> ModuleType:
        import cocos.numerics
        return cocos.numerics

    @classmethod
    def random_module(cls) -> ModuleType:
        import cocos.numerics.random
        return cocos.numerics.random

    @classmethod
    def synchronize(cls):
        from cocos.device import sync
        sync()


class CuPyBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        try:
            import cupy
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        return 'CuPy'

    @classmethod
    def module(cls) -> ModuleType:
        import cupy
        return cupy

    @classmethod
    def random_module(cls) -> ModuleType:
        import cupy.random
        return cupy.random

    @classmethod
    def synchronize(cls):
        import cupy
        cupy.cuda.Stream.null.synchronize()


def get_available_numerical_packages(
        list_installed_bundles: tp.Optional[bool] = False) \
        -> tp.Tuple[tp.Type[NumericalPackageBundle], ...]:
    numerical_bundles_to_try = (NumpyBundle,
                                CocosBundle,
                                CuPyBundle,
                                # AFNumpyBundle
                                )

    available_numerical_bundles \
        = [numerical_bundle
           for numerical_bundle in numerical_bundles_to_try
           if numerical_bundle.is_installed()]

    if list_installed_bundles:
        print(f'Required packages found for the following benchmarks:')
        for numerical_bundle in available_numerical_bundles:
            print(numerical_bundle.label())

        print()

    return tuple(available_numerical_bundles)
