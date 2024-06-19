import geopandas as gpd
import numpy as np
import pandas as pd

from geokde._kernels import (
    epanechnikov_raw,
    epanechnikov_scaled,
    quartic_raw,
    quartic_scaled,
    triweight_raw,
    triweight_scaled,
)


def validate_transform(
    arg: int | float | str,
    arg_name: str,
    points: gpd.GeoDataFrame | gpd.GeoSeries,
) -> np.ndarray:
    """Validation and transformation function for weight and radius arguments.

    If `arg` is present in the GeoDataFrame's columns, said column will be used
    preferentially and converted to ndarray, assuming it is complete and of numeric
    dtype.
    If `arg` is not present in the GeoDataFrame's columns but is int or float, an
    ndarray of equal length to the GeoDataFrame will be populated with the given value.

    Parameters
    ----------
    arg : int | float | str
        Value of the argument to validate and transform.
    arg_name : str
        Name of the argument to validate and transform.
    points : geopandas.GeoDataFrame | geopandas.GeoSeries
        A GeoDataFrame or GeoSeries of point geometries to validate against.

    Returns
    -------
    arg : numpy.ndarray
        ndarray-transformation of `arg`.

    Raises
    ------
    ValueError
        If the column in which weight values are present is not complete or the value
        for `arg` is not appropriate.
    TypeError
        If the column in which weight values are present is not of a numeric dtype.
    """
    if arg in points.columns:
        if not pd.api.types.is_numeric_dtype(points[arg]):
            raise TypeError(f"{arg_name} column must be numeric dtype.")
        if points[arg].isnull().any():
            raise ValueError(f"{arg_name} column must be complete.")
        arg = points[arg].to_numpy()
    elif isinstance(arg, (int, float)):
        arg = np.full(len(points), arg)
    else:
        raise ValueError(
            f"{arg_name} must correspond to a GeoDataFrame column or be int or float.",
        )
    return arg


def create_array(
    bounds: np.ndarray,
    radius: int | float,
    resolution: int | float,
) -> tuple[np.ndarray, list[int | float]]:
    """Generate an ndarray of shape derived from bounding coordinates and spatial
    resolution and calculate the radius- and shape-adjusted bounds thereof.

    Parameters
    ----------
    bounds : numpy.ndarray
        Bounding coordinates to adjust.
    radius : int | float
        Radius with which to adjust coordinates.
    resolution : int | float
        Spatial resolution for the output array.

    Returns
    -------
    array : numpy.ndarray
        Array of shape determined using bounding coordinates and resolution.
    list[int | float]
        List of minx, miny, maxx, and maxy bounding coordinates for the array.
    """
    minx, miny, maxx, maxy = bounds
    minx -= radius
    maxy += radius
    width = round((maxx + radius - minx) / resolution)
    height = round((maxy - (miny - radius)) / resolution)
    maxx = minx + width * resolution
    miny = maxy - height * resolution
    array = np.full((height, width), 0.0)
    return array, [minx, miny, maxx, maxy]


def get_points(
    points: gpd.GeoDataFrame | gpd.GeoSeries,
    minx: int | float,
    maxy: int | float,
    radius: np.ndarray,
    weight: np.ndarray,
    resolution: int | float,
) -> np.ndarray:
    """Generate an ndarray of point X and Y array coordinates, search window radii, and
    weights.

    Parameters
    ----------
    points : geopandas.GeoDataFrame | geopandas.GeoSeries
        A GeoDataFrame or GeoSeries of point geometries from which to derive array
        coordinates.
    minx : int | float
        Minimum X bounding coordinate of the array for which coordinates are generated.
    maxy : int | float
        Maximum Y bounding coordinate of the array for which coordinates are generated.
    radius : numpy.ndarray
        Radii with which to calculate a point's search window radius.
    weight : numpy.ndarray
        Values with which a point's KDE values will be weighted.
    resolution : int | float
        Spatial resolution of the array to which KDE values will be added.

    Returns
    -------
    array_points : numpy.ndarray
        Points comprising array X and Y coordinates, search window radius, and weight.
    """
    x_idx = (points.geometry.x - minx) / resolution
    y_idx = (maxy - points.geometry.y) / resolution
    window = radius / resolution
    array_points = np.column_stack((x_idx, y_idx, window, weight))
    return array_points


def calculate_kde(
    points: np.ndarray,
    array: np.ndarray,
    kernel: str,
    scale: bool,
) -> None:
    """Iterate over elements of an ndarray comprising point properties.

    Parameters
    ----------
    points : numpy.ndarray
        Points comprising array X and Y coordinates, search window radius, and weight
        for which KDE will be calculated.
    array : numpy.ndarray
        Array to which KDE values will be added.
    kernel : str
        Kernel with which to perform KDE.
    scale : bool
        Whether to calculate raw or scaled KDE values.
    """
    kernel_vfuncs = {
        "epanechnikov": epanechnikov_scaled if scale else epanechnikov_raw,
        "quartic": quartic_scaled if scale else quartic_raw,
        "triweight": triweight_scaled if scale else triweight_raw,
    }
    # challenge to vectorise as needs to operate on >1 array element
    for point in points:
        add_point_kde(point, array, kernel_vfuncs[kernel])


def add_point_kde(
    point: np.ndarray,
    array: np.ndarray,
    kernel_vfunc: np.vectorize,
) -> None:
    """Perform KDE for a given point, adding the result to an array.

    Parameters
    ----------
    point : numpy.ndarray
        Properties of the point for which KDE will be calculated, including array X and
        Y coordinates, search window radius, and weight.
    array : numpy.ndarray
        Array to which KDE values will be added.
    kernel_vfunc : numpy.vectorize
        Vectorised kernel function with which to perform KDE.
    """
    x, y, window, weight = point
    minx = round(x - window)
    miny = round(y - window)
    maxx = round(x + window)
    maxy = round(y + window)
    y_idx, x_idx = np.ogrid[miny + 0.5 : maxy + 0.5, minx + 0.5 : maxx + 0.5]
    dist_array = np.sqrt(pow(x_idx - x, 2) + pow(y_idx - y, 2))
    kde_array = kernel_vfunc(dist_array, window, weight)
    array[miny:maxy, minx:maxx] += kde_array
