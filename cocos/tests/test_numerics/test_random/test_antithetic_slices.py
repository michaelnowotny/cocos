from cocos.numerics.random import get_antithetic_slices


def test_antithetic_slices():
    expected_output_1 = (slice(0, 1, 1), slice(0, 10, 1))
    output_1 = get_antithetic_slices((2, 10), antithetic_dimension=0)
    assert output_1 == expected_output_1

    expected_output_2 = (slice(0, 2, 1), slice(0, 5, 1))
    output_2 = get_antithetic_slices((2, 10), antithetic_dimension=1)
    assert output_2 == expected_output_2

    expected_output_3 = (slice(0, 2, 1), slice(0, 4, 1))
    output_3 = get_antithetic_slices((2, 9), antithetic_dimension=1)
    assert output_3 == expected_output_3