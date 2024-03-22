## Roadmap
-------
- Add more kernels.
- Implement other methods of distance measurement, e.g. haversine, manhattan.
- Investigate possible alternatives to iterating over points.
- Enable use of single radius and weight values without filling array of the same length as the points GeoDataFrame/GeoSeries. Results in marginal speed up but the current approach may become an issue with large point datasets.
