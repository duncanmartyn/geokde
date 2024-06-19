import numpy as np

from geokde._utils import (
    create_array,
    validate_transform,
)


def test_validate_transform(geodataframe):
    points = geodataframe("points")
    result = validate_transform("radius", "radius", points)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(points)


def test_create_array():
    radius = 10
    resolution = 1
    bounds = [-10.0, -10.0, 10.0, 10.0]
    expected = [-20, -20, 20, 20]
    array, bounds = create_array(bounds, radius, resolution)
    assert array.shape == (40, 40)
    assert bounds == expected


# TODO: final three _utils fn tests, though all covered by test_geokde
