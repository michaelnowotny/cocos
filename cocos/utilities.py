import collections
import arrayfire as af
import sympy as sym
import typing as tp


def convert_trans_to_af_matprop(trans: int) -> af.MATPROP:
    """
    Convert trans_matprop to matprop.

    Args:
        trans: (todo): write your description
    """
    if trans == 0:
        return af.MATPROP.NONE
    elif trans == 1:
        return af.MATPROP.TRANS
    elif trans == 2:
        return af.MATPROP.CTRANS
    else:
        raise ValueError("trans must be 0, 1, or 2")


def is_broadcastable(shp1: tp.Tuple, shp2: tp.Tuple) -> bool:
    """
    Determine if two iterables in the same.

    Args:
        shp1: (str): write your description
        tp: (str): write your description
        Tuple: (todo): write your description
        shp2: (str): write your description
        tp: (str): write your description
        Tuple: (todo): write your description
    """
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def check_and_make_sequence(sequence_candidate,
                            item_class) -> tp.Sequence:
    """
    Check that sequence_sequence.

    Args:
        sequence_candidate: (todo): write your description
        item_class: (todo): write your description
    """
    if not isinstance(sequence_candidate, collections.abc.Sequence):
        if isinstance(sequence_candidate, item_class):
            sequence_candidate = [sequence_candidate]
        else:
            raise TypeError("sequence_candidate must be of of type "
                            "Sequence[item_class]")

    return sequence_candidate
