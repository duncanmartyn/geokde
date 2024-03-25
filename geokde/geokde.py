import geopandas as gpd
import numpy as np

from geokde._kernels import VALID_KERNELS
from geokde._utils import (
    adjust_bounds,
    calculate_kde,
    create_array,
    get_points,
    validate_transform,
)


def kde(
        points: gpd.GeoDataFrame | gpd.GeoSeries,
        radius: int | float | str,
        resolution: int | float,
        kernel: str = "quartic",
        weight: int | float | str = 1.0,
        scale: bool = False,
) -> tuple[np.ndarray, list[int | float]]:
    """Estimate raw or scaled kernel density with a given radius (or radii), resolution,
    kernel, and weight(s) from point geometries in a given GeoDataFrame or GeoSeries.

    Parameters
    ----------
    points : geopandas.GeoDataFrame | geopandas.GeoSeries
        A GeoDataFrame or GeoSeries of point geometries.
    radius : int | float | str
        Value(s) to use as the search radius.Can be a str, int, or float corresponding
        to a column name in the GeoDataFrame (used preferentially if present), or a
        single int or float value. Should be in the same units as the `points`
        GeoDataFrame's CRS.
    resolution : int | float
        Spatial resolution of the output KDE array. Should be in the same units as the
        `points` GeoDataFrame's CRS.
    kernel : str, default = "quartic"
        Kernel type to use in density estimation. Must be one of "epanechnikov",
        "quartic", or "triweight". Default is "quartic".
    weight : int | float | str, default = 1.0
        Value(s) to use as weights. Can be a str, int, or float corresponding to a
        column name in the GeoDataFrame (used preferentially if present), or a single
        int or float value.
    scale : bool, default = False
        Whether to calculate raw or scaled KDE values, defaults to False (raw).

    Returns
    -------
    array : numpy.ndarray
        Array of KDE values.
    bounds : list[int | float]
         The array's bounding coordinates in minx, miny, maxx, maxy format.

    Raises
    ------
    ValueError
        If the specified kernel is invalid or resolution is greater than the maximum
        specified radius.
    TypeError
        If any geometries are not a point or scale is not boolean.

    Examples
    --------
    Writing results with rasterio:

    >>> gdf = geopandas.read_file("vector_points.geojson")
    >>> kde_array, array_bounds = geokde.kde(gdf, 1, 0.1)
    >>> transform = rasterio.transform.from_bounds(
        *array_bounds,
        kde_array.shape[1],
        kde_array.shape[0],
    )

    >>> with rasterio.open(
        fp="raster.tif",
        mode="w",
        driver="GTiff",
        width=kde_array.shape[1],
        height=kde_array.shape[0],
        count=1,
        crs=gdf.crs,
        transform=transform,
        dtype=kde_array.dtype,
        nodata=0.0,
    ) as dst:
        dst.write(kde_array, 1)
    """
    if not all(points.geometry.geom_type == "Point"):
        raise TypeError("all geometries must be points.")
    if kernel not in VALID_KERNELS:
        raise ValueError(f"kernel must be one of {VALID_KERNELS}, not: {kernel}")
    if not isinstance(scale, bool):
        raise TypeError(f"scale must be bool, not: {type(scale)}")
    radius = validate_transform(radius, "radius", points)
    weight = validate_transform(weight, "weight", points)
    if resolution > radius.max():
        raise ValueError("resolution must be less than radius.")

    bounds = adjust_bounds(*points.total_bounds, radius.max())
    array = create_array(*bounds, resolution)
    points = get_points(points, bounds[0], bounds[3], radius, weight, resolution)
    calculate_kde(points, array, kernel, scale)
    return array, bounds
