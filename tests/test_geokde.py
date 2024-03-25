import numpy as np
import pytest

from geokde.geokde import kde


@pytest.mark.parametrize(
        ["geom_type", "radius", "resolution", "kernel", "weight", "scale", "error"],
        [
            # all valid
            ("points", 1, 0.1, "quartic", 1, False, None),
            # invalid geoms
            ("polygons", 1, 0.1, "quartic", 1, False, TypeError),
            # invalid kernel
            ("points", 1, 0.1, "invalid_kernel", 1, False, ValueError),
            # invalid scale
            ("points", 1, 0.1, "quartic", 1, "invalid_scale", TypeError),
            # all valid, gdf column "radius" as KDE radii
            ("points", "radius", 0.1, "quartic", 1, False, None),
            # gdf column of type object as radii
            ("points", "radius", 0.1, "quartic", 1, False, TypeError),
            # incomplete gdf column as radii
            ("points", "radius", 0.1, "quartic", 1, False, ValueError),
            # needs test for max radius lt resolution
        ],
)
def test_kde(geodataframe, geom_type, radius, resolution, kernel, weight, scale, error):
    gdf = geodataframe(geom_type)
    if error == TypeError and isinstance(radius, str):
        gdf.radius = gdf.radius.astype(str)
    if error == ValueError and isinstance(radius, str):
        gdf.at[0, "radius"] = None
    if error:
        with pytest.raises(error):
            kde(gdf, radius, resolution, kernel, weight, scale)
    else:
        array, bounds = kde(gdf, radius, resolution, kernel, weight, scale)
        assert isinstance(array, np.ndarray)  # nosec
        assert isinstance(bounds, list)  # nosec
        # TODO: more detailed assertions
