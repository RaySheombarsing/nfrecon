"""Input/output utilities for NFRECON.

This module provides lightweight helpers for loading numerical datasets
(e.g. k-space, sensitivity maps) and restoring trained MRI reconstruction
models from disk.
"""

import torch
import numpy as np

from scipy import io as sio
from pathlib import Path
from torch import Tensor
from typing import Any, Mapping

from nfrecon.forward_models.multicoil import MulticoilModel


def load_array_data(path: Path) -> Mapping[str, np.ndarray]:
    """Load numerical array data from disk.

    Supported formats are MATLAB ``.mat`` files (via :mod:`scipy.io`) and
    NumPy ``.npz`` archives. The function is intentionally conservative and
    does **not** attempt to guess semantics of the returned arrays.

    Parameters
    ----------
    path : pathlib.Path
        Path to the data file. Must have suffix ``.mat`` or ``.npz``.

    Returns
    -------
    mapping : Mapping[str, numpy.ndarray]
        Mapping from variable names to NumPy arrays as stored in the file.

    Raises
    ------
    ValueError
        If the file suffix is not supported.

    Notes
    -----
    * MATLAB files are loaded with ``simplify_cells=True`` to reduce nested
      cell structures.
    * ``.npz`` files are loaded with ``allow_pickle=True`` to accommodate
      heterogeneous research data, but users should be aware of the security
      implications.
    """
    suffix = path.suffix.lower()

    if suffix == ".mat":
        return sio.loadmat(path, simplify_cells=True)

    if suffix == ".npz":
        return np.load(path, allow_pickle=True)

    raise ValueError(
        f"Unsupported file format '{suffix}'. Only '.mat' and '.npz' are supported."
    )


def load_multicoil_model(
    model_path: Path,
    device: torch.device | str,
    model: MulticoilModel | None = None,
) -> tuple[Tensor | None, Tensor | None, list[Tensor], MulticoilModel]:
    """Load a trained multicoil MRI reconstruction model from disk.

    This function reconstructs a :class:`~nfrecon.forward_models.multicoil.MulticoilModel`
    form a serialized trained-model file produced by NFRECON training utilities. It
    optionally loads weights into an existing model instance, which can be
    useful for advanced workflows (e.g. wrapping models or parameter surgery).

    Parameters
    ----------
    model_path: Path
        Path to the serialized model checkpoint (``.pt`` or ``.pth``).
    device : torch.device or str
        Device on which the model parameters should be materialized.
    model : MulticoilModel, optional
        Existing model instance into which the checkpoint weights are loaded.
        If ``None``, a new model is instantiated from the checkpoint metadata.

    Returns
    -------
    time_scale : torch.Tensor or None
        Temporal scaling factor for dynamic models. ``None`` for static models.
    time_domain : torch.Tensor or None
        Time-domain interval definition for dynamic models. ``None`` for
        static reconstructions.
    spatial_domain : list[torch.Tensor]
        Spatial coordinate intervals for each reconstruction axis.
    model : MulticoilModel
        Restored multicoil forward model in evaluation-ready state.

    Raises
    ------
    KeyError
        If required fields are missing from the checkpoint.
    RuntimeError
        If the stored model parameters are incompatible with the provided
        ``model`` instance.

    Notes
    -----
    The checkpoint is assumed to follow NFRECON's internal serialization
    convention and must contain at least the following keys:

    ``model_cfg``, ``num_coils``, ``dynamic``, ``model_state_dict``,
    ``spatial_domain``. Optional fields such as ``time_scale`` and
    ``time_domain`` may be ``None`` for static models.
    """
    state: dict[str, Any] = torch.load(
        model_path, map_location=torch.device(device), weights_only=False
    )

    if model is None:
        model = MulticoilModel(
            state["model_cfg"],
            state["num_coils"],
            state["dynamic"],
        ).to(device)

    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return (
        state["time_scale"],
        state["time_domain"],
        Tensor(state["spatial_domain"]),
        model,
    )
