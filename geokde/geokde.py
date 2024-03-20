import geopandas as gpd
import numpy as np
import rasterio as rio

from _utils import calculate_bounds, calculate_kde, create_array, get_array_coordinates


def kernel_density_estimation(
        points: gpd.GeoDataFrame,
        radius: int | float,
        resolution: int | float,
        kernel: str = "quartic",
) -> tuple[np.ndarray, rio.Affine]:
    """Estimate kernel density with a given radius, resolution, and kernel from point
    geometries in a given `GeoDataFrame`.

    Parameters
    ----------
    points : geopandas.GeoDataFrame
    radius : int | float
    resolution : int | float
    kernel : str, default = "quartic"

    Returns
    -------
    arr : numpy.ndarray
    transform : rasterio.Affine
    """
    bounds = calculate_bounds(*points.total_bounds, radius)
    array, transform = create_array(*bounds, resolution)
    get_array_coordinates(points, transform)
    calculate_kde(points, array, radius, resolution)
    return array, transform


# TODO:
# implement different kernels
# check gdf for weight column
# check gdf for radius column
# vectorise point kde estimation
