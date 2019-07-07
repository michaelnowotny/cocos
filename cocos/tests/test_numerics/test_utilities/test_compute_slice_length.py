import numpy as np
import pytest

from cocos.numerics._utilities import _compute_slice_length


test_data = [(slice(0, 3, 2), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
             (slice(1, 3, 2), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
             (slice(2, 3, 2), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
             (slice(3, 3, 2), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
             (slice(1, 3, 3), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
             (slice(1, 2, 2), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
             (slice(3, 3, 2), np.array([1, 2, 3, 4, 5, 6, 7, 8]))]


@pytest.mark.parametrize("index, x", test_data)
def test_compute_slice_length(index: slice, x: np.ndarray):
    y = x[index]
    computed_length = _compute_slice_length(index, len(x))
    if not len(y) == computed_length:
        print(f'computed length = {computed_length}, actual length = {len(y)}')
    assert len(y) == computed_length
