"""Utilities for manipulating k-space data representations.

This module provides core library functions for converting between dense
(gridded) k-space arrays and sparse, coordinate-based representations.
These utilities are used throughout NFRECON for efficient storage, batching,
and reconstruction of k-space data.
"""

import numpy as np


def flatten_kspace(
    kspace: np.ndarray, dynamic: bool
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Convert a dense k-space array into a sparse, coordinate-based representation.

    This function identifies nonzero k-space locations (based on coil index 0)
    and extracts the corresponding values across all coils. The output is a
    time-indexed sparse representation consisting of integer coordinates and
    complex k-space samples.

    Both dynamic and static k-space data are returned in a *uniform* format:
    a list of per-time-frame coordinate arrays and a list of per-time-frame
    value arrays. Static k-space data is treated as a degenerate dynamic case
    with a single time frame.

    Parameters
    ----------
    kspace : np.ndarray
        Complex-valued k-space data. Expected shapes are:

        - ``(nc, nt, *spatial_dims)`` for time-resolved or binned (dynamic) data
        - ``(nc, *spatial_dims)`` for static data

        where ``nc`` is the number of coils and ``nt`` the number of time frames.
    dynamic : bool
        Indicates how the input should be interpreted with respect to the time
        axis if present. Static data is returned as a single-frame dynamic
        representation.

    Returns
    -------
    kspace_coords : list[np.ndarray]
        List of coordinate arrays, one per time frame. Each array has shape
        ``(num_nonzero, n)``, where ``n`` is the number of spatial dimensions.
    kspace_nonzero: list[np.ndarray]
        List of k-space value arrays, one per time frame. Each array has shape
        ``(nc, num_nonzero)``.

    Notes
    -----
    - The sampling pattern is inferred from coil index 0 and is assumed identical
      across coils.
    - Zero-valued entries are treated as unsampled.
    - Static k-space data is returned as a list of length one, corresponding to
      a single time frame. This guarantees a stable return type for downstream
      processing.
    """
    if not isinstance(kspace, np.ndarray):
        raise TypeError("kspace must be a numpy array")

    if dynamic:
        if kspace.ndim < 3:
            raise ValueError("Dynamic k-space must have shape (nc, nt, *spatial_dims)")

        num_times = kspace.shape[1]
        kspace_coords: list[np.ndarray] = []
        kspace_nonzero: list[np.ndarray] = []

        for t in range(num_times):
            nonzero_ind = np.where(np.abs(kspace[0, t]) > 0)
            kspace_nonzero.append(kspace[:, t, *nonzero_ind])
            kspace_coords.append(np.array(nonzero_ind, dtype=int).T)
    else:
        if kspace.ndim < 2:
            raise ValueError("Static k-space must have shape (nc, *spatial_dims)")
        nonzero_ind = np.where(np.abs(kspace[0]) > 0)
        kspace_nonzero = [kspace[:, *nonzero_ind]]
        kspace_coords = [np.array(nonzero_ind, dtype=int).T]

    return kspace_coords, kspace_nonzero


def unflatten_kspace(
    kspace_coords: list[np.ndarray],
    kspace_vals: list[np.ndarray],
    gridsize: np.ndarray,
    squeeze_time_axis: bool = False,
) -> np.ndarray:
    """
    Reconstruct a dense k-space array from a sparse, coordinate-based representation.

    This function is the inverse of :func:`flatten_kspace`. It reconstructs a
    dense, gridded k-space array by placing sparse k-space samples at their
    corresponding integer coordinates for each time frame.

    Parameters
    ----------
    kspace_coords : list[np.ndarray]
        List of coordinate arrays, one per time frame. Each array has shape
        ``(num_nonzero, n)``, where ``n`` is the number of spatial dimensions.
    kspace_vals : list[np.ndarray]
        List of k-space value arrays, one per time frame. Each array has shape
        ``(nc, num_nonzero)``, where ``nc`` is the number of coils.
    gridsize : np.ndarray
        Spatial grid size of the target k-space array, e.g. ``(ny, nx)`` or
        ``(nz, ny, nx)``.
    squeeze_time_axis : bool
        If ``True`` and only a single time frame is present (nt == 1), the
        time dimension is removed from the returned array.

    Returns
    -------
    np.ndarray
        Dense k-space array of shape ``(nc, nt, *gridsize)``, where ``nt`` is the
        number of time frames. If ``nt=1`` and ``squeeze_time_axis=True``
        time dimension is squeezed out.
    """
    if len(kspace_coords) != len(kspace_vals):
        raise ValueError("kspace_coords and kspace_vals must have the same length")

    nt = len(kspace_coords)
    if nt == 0:
        raise ValueError("kspace_coords must contain at least one time frame")

    nc = kspace_vals[0].shape[0]
    dtype = kspace_vals[0].dtype
    kspace_arr = np.zeros((nc, nt, *gridsize), dtype=dtype)

    for t, (coords, vals) in enumerate(zip(kspace_coords, kspace_vals)):
        kspace_arr[:, t, *coords.T] = vals

    if nt == 1 and squeeze_time_axis:
        kspace_arr = kspace_arr.squeeze(axis=1)

    return kspace_arr
