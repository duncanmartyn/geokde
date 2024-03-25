![pypi version](https://img.shields.io/pypi/v/geokde)
![pypi downloads](https://img.shields.io/pypi/dm/geokde)
[![publish](https://github.com/duncanmartyn/geokde/actions/workflows/publish.yaml/badge.svg?branch=main)](https://github.com/duncanmartyn/geokde/actions/workflows/publish.yaml)
[![test](https://github.com/duncanmartyn/geokde/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/duncanmartyn/geokde/actions/workflows/test.yaml)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

# GeoKDE
Package for geospatial kernel density estimation (KDE).

Written in Python 3.10.11 (though compatible with 3.10.11+), GeoKDE depends on the following:
- `geopandas`
- `numpy` (itself a dependency of `geopandas`)

# Examples
Perform KDE on a GeoJSON of point geometries and write the result to a GeoTIFF raster file with `rasterio`:
```
gdf = geopandas.read_file("vector_points.geojson")
kde_array, array_bounds = geokde.kde(gdf, 1, 0.1)
transform = rasterio.transform.from_bounds(
    *array_bounds,
    kde_array.shape[1],
    kde_array.shape[0],
)

with rasterio.open(
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
```

# Roadmap
- Add more kernels.
- Finish tests - coverage is >=95% for _utils.py and geokde.py as is.
- Implement other methods of distance measurement, e.g. haversine, Manhattan.
- Investigate alternatives to iterating over points.
- Enable use of single radius and weight values without filling array of the same length as the points GeoDataFrame/GeoSeries. Results in marginal speed up but the current approach may become an issue with very large point datasets.
- Integrate mypy in pre-commit, possibly also linter and formatter though flake8 and black used locally.

# Contributions
Feel free to raise any issues, especially bugs and feature requests!
