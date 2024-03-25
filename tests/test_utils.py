import numpy as np

from geokde._utils import (
    adjust_bounds,
    create_array,
    validate_transform,
)


def test_validate_transform(geodataframe):
    points = geodataframe("points")
    result = validate_transform("radius", "radius", points)
    assert isinstance(result, np.ndarray)  # nosec
    assert len(result) == len(points)  # nosec


def test_adjust_bounds():
    radius = 10
    bounds = [-10, -10, 10, 10]
    expected = [-20, -20, 20, 20]
    result = adjust_bounds(*bounds, radius)
    assert result == expected  # nosec


def test_create_array():
    resolution = 1
    bounds = [-10, -10, 10, 10]
    array = create_array(*bounds, resolution)
    assert array.shape == (20, 20)  # nosec

# TODO: final three _utils fn tests, though all covered by test_geokde
