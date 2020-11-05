import numpy as np
import pytest
import typing as tp

from cocos.numerics._array import _complete_shape

two_by_two = np.array([[1, 2], [3, 4]])
two_by_two_by_two = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

test_data = [(two_by_two, (1, -1), (1, 4)),
             (two_by_two, (2, -1), (2, 2)),
             (two_by_two, (4, -1), (4, 1)),
             (two_by_two, (-1, 1), (4, 1)),
             (two_by_two, (-1, 2), (2, 2)),
             (two_by_two, (-1, 4), (1, 4)),
             (two_by_two_by_two, (-1, 2, 2), (2, 2, 2)),
             (two_by_two_by_two, (-1, 1, 1), (8, 1, 1)),
             (two_by_two_by_two, (-1, 2), (4, 2)), ]


@pytest.mark.parametrize("a, input_shape, intended_output_shape", test_data)
def test_complete_shape(a,
                        input_shape: tp.Tuple[int, ...],
                        intended_output_shape: tp.Tuple[int, ...]):
    """
    Test if a is an input shape.

    Args:
        a: (todo): write your description
        input_shape: (list): write your description
        tp: (todo): write your description
        Tuple: (todo): write your description
        int: (todo): write your description
        intended_output_shape: (todo): write your description
        tp: (todo): write your description
        Tuple: (todo): write your description
        int: (todo): write your description
    """

    output_shape = _complete_shape(a, input_shape)
    assert output_shape == intended_output_shape
    # y = x[index]
    # computed_length = _compute_slice_length(index, len(x))
    # if not len(y) == computed_length:
    #     print(f'computed length = {computed_length}, actual length = {len(y)}')
    # assert len(y) == computed_length
