"""Utilities for working with Hydra / OmegaConf configurations.

This module provides helper functions for manipulating Hydra configuration
objects, such as flattening nested configs for logging or serialization.
"""

from typing import Any, Mapping
from omegaconf import DictConfig


def flatten_dict(
    data_dict: DictConfig | Mapping[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """
    Flatten a nested mapping or Hydra DictConfig into a single-level dictionary.

    Nested keys are concatenated using the specified separator. This function
    is primarily intended for logging and inspection purposes (e.g. flattening
    Hydra configs for experiment tracking).

    Parameters
    ----------
    data : DictConfig or Mapping[str, Any]
        Input configuration or dictionary. Nested mappings will be flattened.
    parent_key : str, optional
        Prefix to prepend to all keys (used internally for recursion).
    sep : str, optional
        Separator used to join nested key names.

    Returns
    -------
    dict[str, Any]
        A flattened dictionary mapping composite keys to leaf values.

    Notes
    -----
    - OmegaConf interpolations are **not resolved** by this function; values
      are returned as stored in the config.
    - Sequences (lists, tuples) are treated as leaf values and are not flattened.
    - This function is intended for inspection/logging, not for round-trip
      transformation of configs.
    """
    if not hasattr(data_dict, "items"):
        raise TypeError(
            "flatten_dict expects a mapping or DictConfig with an .items() method"
        )

    flat: dict[str, Any] = {}

    for key, value in data_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, (DictConfig, Mapping)):
            flat.update(flatten_dict(value, new_key, sep=sep))
        else:
            flat[new_key] = value

    return flat
