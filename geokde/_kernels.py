"""Kernels for density estimation.

All kernels accept the following parameters:

Parameters
----------
distance : int | float
    Euclidean distance value for a given point from the reference point.
radius : int | float
    Search window radius for the reference point.
weight : int | float
    Value with which the KDE value for a point will be weighted.

And provide the following return:

Returns
-------
value : float
    Specified kernel's density estimation value.
"""

import numpy as np


VALID_KERNELS = [
    "epanechnikov",
    "quartic",
    "triweight",
]


@np.vectorize
def quartic_raw(
        distance: int | float,
        radius: int | float,
        weight: int | float,
) -> float:
    """Raw Quartic kernel."""
    if distance < radius:
        value = weight * pow(1 - pow(distance / radius, 2), 2)
    else:
        value = 0.0
    return value


@np.vectorize
def quartic_scaled(
        distance: int | float,
        radius: int | float,
        weight: int | float,
) -> float:
    """Scaled Quartic kernel."""
    if distance < radius:
        norm_const = 116 / (5 * np.pi * pow(radius, 2))
        value = weight * (norm_const * (15 / 16) * pow(1 - pow(distance / radius, 2), 2))
    else:
        value = 0.0
    return value


@np.vectorize
def epanechnikov_raw(
        distance: int | float,
        radius: int | float,
        weight: int | float,
) -> float:
    """Raw Epanechnikov kernel."""
    if distance < radius:
        value = weight * (1 - pow(distance / radius, 2))
    else:
        value = 0.0
    return value


@np.vectorize
def epanechnikov_scaled(
        distance: int | float,
        radius: int | float,
        weight: int | float,
) -> float:
    """Scaled Epanechnikov kernel."""
    if distance < radius:
        norm_const = 8 / (3 * np.pi * pow(radius, 2))
        value = weight * (norm_const * (3 / 4) * (1 - pow(distance / radius, 2)))
    else:
        value = 0.0
    return value


@np.vectorize
def triweight_raw(
        distance: int | float,
        radius: int | float,
        weight: int | float,
) -> float:
    """Raw triweight kernel."""
    if distance < radius:
        value = weight * pow(1 - pow(distance / radius, 2), 3)
    else:
        value = 0.0
    return value


@np.vectorize
def triweight_scaled(
        distance: int | float,
        radius: int | float,
        weight: int | float,
) -> float:
    """Scaled triweight kernel."""
    if distance < radius:
        norm_const = 128 / (35 * np.pi * pow(radius, 2))
        value = weight * (norm_const * (35 / 32) * pow(1 - pow(distance / radius, 2), 3))
    else:
        value = 0.0
    return value
