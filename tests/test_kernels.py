import pytest

from geokde import _kernels

KERNEL_VFUNCS = [
    _kernels.epanechnikov_raw,
    _kernels.epanechnikov_scaled,
    _kernels.quartic_raw,
    _kernels.quartic_scaled,
    _kernels.triweight_raw,
    _kernels.triweight_scaled,
]


@pytest.mark.parametrize(
        ["distance", "radius", "weight", "expected"],
        [
            (0, 10, 1, 1),
            (0, 10, 2, 2),
            (10, 10, 1, 0),
            (20, 10, 1, 0),
        ]
)
def test_kernel(distance, radius, weight, expected):
    for kernel_vfunc in KERNEL_VFUNCS:
        result = kernel_vfunc(distance, radius, weight)
        assert result <= expected  # nosec
