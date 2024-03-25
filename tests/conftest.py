from typing import Callable

import geopandas as gpd
import pytest
import shapely


@pytest.fixture
def geodataframe() -> Callable:
    def _create(geom_type: str) -> gpd.GeoDataFrame:
        if geom_type == "points":
            points = [(0, 0), (10, 10)]
            geoms = [shapely.Point(*coords) for coords in points]
        if geom_type == "polygons":
            polygons = [(-180, 0, 0, 90), (0, 0, 90, 180)]
            geoms = [shapely.box(*coords) for coords in polygons]

        gdf = gpd.GeoDataFrame(geometry=geoms, crs=4326)
        gdf["radius"] = 1
        gdf["weight"] = 1
        return gdf
    return _create
