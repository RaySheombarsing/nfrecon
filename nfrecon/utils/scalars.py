"""Scalar conversion utilities.

This module provides helper functions for converting PyTorch tensors
to native Python scalar types. Such conversions are commonly required
at *API boundaries*, for example when logging values, reporting metrics,
or interfacing with libraries that expect standard Python types rather
than framework-specific objects.
"""

import torch


def to_scalar_dict(d: dict[str, torch.Tensor | float]) -> dict[str, float]:
    """Convert a dictionary of scalar tensors to native Python floats.

    This function takes a dict whose values are either PyTorch tensors
    representing scalar quantities (i.e., zero-dimensional tensors) or
    Python numeric types, and returns a new dictionary in which all values
    are converted to Python ``float`` objects.

    Parameters
    ----------
    d : dict[str, torch.Tensor or float]
        Dictionary mapping names to scalar-valued tensors or numeric
        Python values. Tensor values must contain exactly one element.

    Returns
    -------
    dict[str, float]
        Dictionary with the same keys as the input, where all values have
        been converted to native Python floats.
    """
    return {k: v.item() if torch.is_tensor(v) else float(v) for k, v in d.items()}
