import geopandas as gpd
import numpy as np
import rasterio as rio

from _kernels import quartic


def calculate_bounds(
        minx: int | float,
        miny: int | float,
        maxx: int | float,
        maxy: int | float,
        radius: int | float,
) -> tuple[int | float, int | float, int | float, int | float]:
    """Calculate bounds for

    Parameters
    ----------
    minx : int | float
    miny : int | float
    maxx : int | float
    radius : int | float
    """
    minx -= radius
    miny -= radius
    maxx += radius
    maxy += radius
    return minx, miny, maxx, maxy


def create_array(
        minx: int | float,
        miny: int | float,
        maxx: int | float,
        maxy: int | float,
        resolution: int | float,
) -> tuple[np.ndarray, rio.Affine]:
    """Generate an array and associated affine transformation from bounding coordinates
    and spatial resolution.

    Parameters
    ----------
    minx : int | float
    miny : int | float
    maxx : int | float
    resolution : int | float

    Returns
    -------
    array : numpy.ndarray
    tf : rasterio.Affine
    """
    dim_x = round((maxx - minx) / resolution)  # round() no?
    dim_y = round((maxy - miny) / resolution)
    array = np.full((dim_y, dim_x), 0.0)
    tf = rio.transform.from_bounds(minx, miny, maxx, maxy, dim_x, dim_y)
    return array, tf


def get_array_coordinates(gdf: gpd.GeoDataFrame, transform: rio.Affine) -> None:
    """
    """
    gdf["arr_idx"] = gdf.geometry.apply(
        lambda point: rio.transform.rowcol(
            transform, point.x, point.y, op=transform_rowcol_return),
    )


def transform_rowcol_return(value: float) -> float:
    """Function that returns the value it's passed. Used with `op` kwarg of
    `rasterio.transform.rowcol()` to ensure decimal array index return."""
    return value


def calculate_kde(
        gdf: gpd.GeoDataFrame,
        array: np.ndarray,
        radius: int | float,
        resolution: int | float,
) -> None:
    """Iterate over GeoDataFrame records, calculating KDE for a given radius and
    resolution and adding the result to a given array.
    """
    window = radius / resolution
    for point in gdf.to_dict(orient="records"):
        add_point_kde(array, point["arr_idx"], window)


def add_point_kde(
        array: np.ndarray,
        point: tuple[int | float, int | float],
        window: int | float,
        weight: int | float = 1
) -> None:
    """Perform KDE for a given point, window, and weight, adding the result to an array.

    Parameters
    ----------
    array : numpy.ndarray
    point : tuple[int | float, int | float]
    window : int | float
    weight: int | float, default = 1
    """
    minx = round(point[1] - window)
    miny = round(point[0] - window)
    maxx = round(point[1] + window)
    maxy = round(point[0] + window)
    y_idx, x_idx = np.ogrid[miny + .5:maxy + .5, minx + .5:maxx + .5]
    dist_array = np.sqrt((x_idx - point[1]) ** 2 + (y_idx - point[0]) ** 2)
    dist_array[dist_array >= window] = 0.0
    kde_array = quartic(dist_array, window, weight)
    array[miny:maxy, minx:maxx] += kde_array
