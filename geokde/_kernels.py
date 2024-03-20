import numpy as np


@np.vectorize
def quartic(distance: int | float, radius: int | float, weight: int | float) -> float:
    """Quartic kernel.

    Parameters
    ----------
    distance : int | float
    radius : int | float
    weight : int | float

    Returns
    -------
    value : float
    """
    if distance:
        value = weight * ((1 - ((distance / radius) ** 2)) ** 2)
    else:
        value = 0.0
    return value
