from abc import ABC, abstractmethod
from types import ModuleType
import typing as tp


class NumericalPackageBundle(ABC):
    @classmethod
    @abstractmethod
    def is_installed(cls) -> bool:
        """
        Return true if any of - installed packages are installed.

        Args:
            cls: (todo): write your description
        """
        pass

    @classmethod
    @abstractmethod
    def label(cls) -> str:
        """
        Create a label.

        Args:
            cls: (callable): write your description
        """
        pass

    @classmethod
    @abstractmethod
    def module(cls) -> ModuleType:
        """
        Returns the module to the given type.

        Args:
            cls: (todo): write your description
        """
        pass

    @classmethod
    @abstractmethod
    def random_module(cls) -> ModuleType:
        """
        Returns a random random module.

        Args:
            cls: (todo): write your description
        """
        pass

    @classmethod
    def synchronize(cls):
        """
        Syncs the given cls from the given cls.

        Args:
            cls: (todo): write your description
        """
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
        """
        Return true if a boolean is installed.

        Args:
            cls: (todo): write your description
        """
        try:
            import numpy
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        """
        Return label.

        Args:
            cls: (callable): write your description
        """
        return 'NumPy'

    @classmethod
    def module(cls) -> ModuleType:
        """
        Returns a module object from the module.

        Args:
            cls: (todo): write your description
        """
        import numpy
        return numpy

    @classmethod
    def random_module(cls) -> ModuleType:
        """
        Return a random random module.

        Args:
            cls: (todo): write your description
        """
        import numpy.random
        return numpy.random


class CocosBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        """
        Determine if the current executable is installed.

        Args:
            cls: (todo): write your description
        """
        try:
            import cocos
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        """
        Return label.

        Args:
            cls: (callable): write your description
        """
        return 'Cocos'

    @classmethod
    def module(cls) -> ModuleType:
        """
        Returns the module associated with the given name.

        Args:
            cls: (todo): write your description
        """
        import cocos.numerics
        return cocos.numerics

    @classmethod
    def random_module(cls) -> ModuleType:
        """
        Return a random module.

        Args:
            cls: (todo): write your description
        """
        import cocos.numerics.random
        return cocos.numerics.random

    @classmethod
    def synchronize(cls):
        """
        Synchronize the device.

        Args:
            cls: (todo): write your description
        """
        from cocos.device import sync
        sync()


class CuPyBundle(NumericalPackageBundle):
    @classmethod
    def is_installed(cls) -> bool:
        """
        Determine whether or false otherwise.

        Args:
            cls: (todo): write your description
        """
        try:
            import cupy
            return True
        except:
            return False

    @classmethod
    def label(cls) -> str:
        """
        Return label.

        Args:
            cls: (callable): write your description
        """
        return 'CuPy'

    @classmethod
    def module(cls) -> ModuleType:
        """
        Return a module from the given type.

        Args:
            cls: (todo): write your description
        """
        import cupy
        return cupy

    @classmethod
    def random_module(cls) -> ModuleType:
        """
        Returns a random random module.

        Args:
            cls: (todo): write your description
        """
        import cupy.random
        return cupy.random

    @classmethod
    def synchronize(cls):
        """
        Synchronize the image.

        Args:
            cls: (todo): write your description
        """
        import cupy
        cupy.cuda.Stream.null.synchronize()


def get_available_numerical_packages(
        list_installed_bundles: tp.Optional[bool] = False) \
        -> tp.Tuple[tp.Type[NumericalPackageBundle], ...]:
    """
    Return a list of tuples.

    Args:
        list_installed_bundles: (str): write your description
        tp: (todo): write your description
        Optional: (todo): write your description
        bool: (todo): write your description
    """
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
