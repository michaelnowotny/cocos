import collections
import arrayfire as af
import typing as tp


def convert_trans_to_af_matprop(trans: int) -> af.MATPROP:
    if trans == 0:
        return af.MATPROP.NONE
    elif trans == 1:
        return af.MATPROP.TRANS
    elif trans == 2:
        return af.MATPROP.CTRANS
    else:
        raise ValueError("trans must be 0, 1, or 2")


def is_broadcastable(shp1: tp.Tuple, shp2: tp.Tuple) -> bool:
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def check_and_make_sequence(sequence_candidate,
                            item_class) -> tp.Sequence:
    if not isinstance(sequence_candidate, collections.Sequence):
        if isinstance(sequence_candidate, item_class):
            sequence_candidate = [sequence_candidate]
        else:
            raise TypeError("sequence_candidate must be of of type "
                            "Sequence[item_class]")

    return sequence_candidate
