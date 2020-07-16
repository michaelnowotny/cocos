import pytest
import typing as tp

from cocos.multi_processing.utilities import (
    generate_slices_with_batch_size,
    generate_slices_with_number_of_batches
)


@pytest.mark.parametrize('n, batch_size, result', [(10, 2, ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))),
                                                   (10, 3, ((0, 3), (3, 6), (6, 9), (9, 10))),
                                                   (10, 4, ((0, 4), (4, 8), (8, 10)))])
def test_generate_slices_with_batch_size(n: int, batch_size: int, result: tp.Tuple[tp.Tuple[int, int]]):
    assert result == generate_slices_with_batch_size(n=n, batch_size=batch_size)


@pytest.mark.parametrize('n, number_of_batches, result', [(10, 2, ((0, 5), (5, 10))),
                                                          (10, 3, ((0, 4), (4, 8), (8, 10))),
                                                          (10, 4, ((0, 3), (3, 6), (6, 9), (9, 10)))])
def test_generate_slices_with_batch_size(n: int, number_of_batches: int, result: tp.Tuple[tp.Tuple[int, int]]):
    assert result == generate_slices_with_number_of_batches(n=n, number_of_batches=number_of_batches)
