"""Multicoil loss functions for parallel imaging.

This module implements a concrete loss function for solving the multi-coil
parallel imaging problem using neural field representations. The loss
combines data consistency in k-space with regularization terms on the
magnetization and coil sensitivity fields.
"""

import torch
import numpy as np

from torch import Tensor
from typing import Callable, Mapping

from nfrecon.loss.loss_base import BaseLoss
from nfrecon.forward_models.multicoil import MulticoilModel
from nfrecon.samplers.grids import CoordPatch
from nfrecon.data.dataset import CoilData
from nfrecon.utils import fourier


class MulticoilLoss(BaseLoss):
    """Composite loss for multi-coil parallel imaging.

    The loss consists of several components:

    - A data consistency term enforcing agreement between predicted and
      observed k-space data.
    - Spatial total-variation-like regularization on the magnetization field.
    - Temporal regularization of the magnetization (dynamic case only).
    - A smoothness penalty on the coil sensitivity maps.

    Parameters
    ----------
    weights : Mapping[str, float]
        Mapping from loss component names to their corresponding weights.
    threshold_kspace_obs : float, optional
        Threshold below which k-space observations are not weighted in the
        data consistency term.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        threshold_kspace_obs: float = 1e-05,
    ) -> None:
        super().__init__(weights)
        self._threshold_kspace_obs = threshold_kspace_obs

    def __call__(
        self,
        kspace_obs: Tensor,
        kspace_pred: Tensor,
        dmdx: Tensor | None = None,
        dmdt: Tensor | None = None,
        jac_coil_sensitivity: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Evaluate multicoil loss components.

        Parameters
        ----------
        kspace_obs : complex-valued Tensor
            Observed k-space data.
            If ``dynamic == True``, shape is
            ``(num_coils, num_times, num_kspace_obs)``.
            Otherwise, shape is ``(num_coils, num_kspace_obs)``.
        kspace_pred : complex-valued Tensor
            Predicted k-space data with the same shape as ``kspace_obs``.
        dmdx : complex-valued Tensor, optional
            Spatial derivatives of the magnetization field.
            If ``dynamic == True``, shape is
            ``(num_times, *spatial_grid_size, spatial_dim)``.
            Otherwise, shape is ``(*spatial_grid_size, spatial_dim)``.
        dmdt : complex-valued Tensor, optional
            Temporal derivative of the magnetization field.
            Only present in the dynamic case.
            Shape is ``(num_times, *spatial_grid_size)``.
        jac_coil_sensitivity : complex-valued Tensor, optional
            Spatial derivatives of the coil sensitivity maps.
            Shape is ``(*spatial_grid_size, num_coils, spatial_dim)``.

        Returns
        -------
        dict[str, Tensor]
            Dictionary mapping loss component names to scalar loss values.
            The entry ``"complete_loss"`` contains the full weighted loss.
        """
        device = kspace_obs.device
        loss = self._zero_init_loss()

        # Data consistency
        diff = kspace_obs - kspace_pred
        diff_abs_sqr = torch.real(diff * torch.conj(diff))

        weights = torch.sqrt(torch.real(kspace_obs * torch.conj(kspace_obs)))
        weights[weights < self._threshold_kspace_obs] = 1.0
        weights.reciprocal_()

        loss["data_consistency"] = torch.mean(
            torch.sqrt(
                torch.sum(
                    diff_abs_sqr * torch.sqrt(weights),
                    dim=-1,
                )
            )
        )

        # Temporal-TV regularization on magnetization
        if dmdt is not None:
            loss["dmdt"] = torch.mean(torch.abs(dmdt))

        # Spatial-TV regularization on magnetization
        if dmdx is not None:
            loss["dmdx"] = torch.mean(
                torch.sqrt(torch.sum(torch.real(dmdx * torch.conj(dmdx)), dim=-1))
            )

        # Spatial (smoothness) regularization on coil sensitivities
        if jac_coil_sensitivity is not None:
            loss["jac_coil_sense"] = torch.mean(
                torch.sum(
                    torch.real(torch.conj(jac_coil_sensitivity) * jac_coil_sensitivity),
                    dim=-1,
                )
            )

        loss["complete_loss"] = self._compute_full_loss(loss, device)

        return loss


def eval_loss_static(
    model: MulticoilModel,
    loss: MulticoilLoss,
    dataset: CoilData,
    coord_patch: CoordPatch,
    coil_indices: list[int],
    device: str,
) -> dict[str, Tensor]:
    """Evaluate the multicoil loss in the static setting.

    Parameters
    ----------
    model : MulticoilModel
        Multicoil forward model.
    loss : MulticoilLoss
        Multicoil loss instance.
    dataset : CoilData
        Dataset containing k-space observations.
    coord_patch : CoordPatch
        Coordinate patch defining the sampled spatial locations.
    coil_indices : list[int]
        Indices of coils to include in the loss evaluation.
    device : str or torch.device
        Device on which computations are performed.

    Returns
    -------
    dict[str, Tensor]
        Dictionary of loss components evaluated on the static dataset.
    """
    full_obs_grid = [g.to(device) for g in dataset.spatial_partitions]

    # Compute coil images
    m_pred = model.magnetization(*full_obs_grid)
    coil_sense = model.coil_sensitivity(*full_obs_grid)[..., coil_indices]
    m_pred_coils = torch.einsum("...,...c->c...", m_pred, coil_sense)

    # Fourier Transform coil images: loop over coils to save memory.
    kspace_pred_coils = torch.stack(
        [
            fourier.cft(y, model.spatial_dim)[*(dataset.kspace_coords.T)]
            for y in m_pred_coils
        ],
        axis=0,
    )

    # Regularization terms
    jac_coil_sense = model.coil_sensitivity.partial_jac(
        model.coil_spatial_indices,
        *coord_patch.rand_mixed_grid_per_dim,
    )
    dmdx = model.magnetization.partial_jac(
        model.magnetization_spatial_indices, *coord_patch.rand_mixed_grid_per_dim
    )

    return loss(
        dataset.kspace[coil_indices].to(device),
        kspace_pred_coils,
        dmdx=dmdx,
        jac_coil_sensitivity=jac_coil_sense,
    )


def eval_loss_dynamic(
    model: MulticoilModel,
    loss: MulticoilLoss,
    dataset: CoilData,
    coord_patch: CoordPatch,
    coil_indices: list[int],
    device: str,
) -> dict[str, Tensor]:
    """Evaluate the multicoil loss in the dynamic (time-dependent) setting.

    Parameters
    ----------
    model : MulticoilModel
        Multicoil forward model.
    loss : MulticoilLoss
        Multicoil loss instance.
    dataset : CoilData
        Dataset containing k-space observations.
    coord_patch : CoordPatch
        Randomly sampled space-time coordinate patch.
    coil_indices : list[int]
        Indices of coils to include in the loss evaluation.
    device : str or torch.device
        Device on which computations are performed.

    Returns
    -------
    dict[str, Tensor]
        Dictionary of loss components evaluated on the dynamic dataset.
    """
    time_subgrid_ind = coord_patch.grid_ind_per_dim[0]
    time_subgrid = coord_patch.rand_subgrid_per_dim[0]
    time_mixed_grid = coord_patch.rand_mixed_grid_per_dim[0]
    spatial_mixed_grid = coord_patch.rand_mixed_grid_per_dim[1:]
    full_obs_spatial_grid = [g.to(device) for g in dataset.spatial_partitions]

    m_pred = model.magnetization(time_subgrid, *full_obs_spatial_grid)
    coil_sense = model.coil_sensitivity(*full_obs_spatial_grid)[..., coil_indices]
    m_pred_coils = torch.einsum("t...,...c->ct...", m_pred, coil_sense)

    kspace_mask = dataset.kspace_mask[time_subgrid_ind].to(device)
    num_times, num_obs = len(time_subgrid), kspace_mask.shape[1]
    kspace_coords = dataset.kspace_coords[time_subgrid_ind].view(
        num_times * num_obs, model.spatial_dim
    )
    kspace_time_indices = (
        torch.arange(num_times).unsqueeze(1).repeat(1, num_obs)
    ).flatten()

    # Fourier Transform coil images: loop over coils to save memory.
    kspace_pred_coils = torch.stack(
        [
            fourier.cft(y, model.spatial_dim)[
                kspace_time_indices, *(kspace_coords.T)
            ].view(num_times, num_obs)
            * kspace_mask
            for y in m_pred_coils
        ],
        axis=0,
    )

    jac_coil_sense = model.coil_sensitivity.partial_jac(
        model.coil_spatial_indices,
        *spatial_mixed_grid,
    )

    dmdx = model.magnetization.partial_jac(
        model.magnetization_spatial_indices, time_mixed_grid, *spatial_mixed_grid
    )

    dmdt = model.magnetization.partial_jac(
        (0,), time_mixed_grid, *spatial_mixed_grid
    ).squeeze(-1)

    return loss(
        dataset.kspace[coil_indices][:, time_subgrid_ind].to(device),
        kspace_pred_coils,
        dmdx=dmdx,
        dmdt=dmdt,
        jac_coil_sensitivity=jac_coil_sense,
    )


# TODO: create and move to general utilities module.
def init_time_batches(
    time_grid: Tensor,
    num_times: int,
    batch_size: int,
) -> tuple[list[np.ndarray], int]:
    """Split time indices into batches.

    Parameters
    ----------
    time_grid : Tensor
        One-dimensional time grid.
    num_times : int
        Number of time points to sample from the full grid.
    batch_size : int
        Batch size (over time).

    Returns
    -------
    time_indices_per_batch : list[np.ndarray]
        time subgrid indices, one per batch.
    num_batches : int
        Number of batches.
    """
    num_times_full = len(time_grid)

    if num_times_full <= num_times:
        time_sub_indices = np.arange(num_times_full)
        num_times = num_times_full
    else:
        time_sub_indices = np.random.choice(
            np.arange(num_times_full), size=num_times, replace=False
        )

    num_batches = num_times // batch_size
    time_indices_per_batch = [
        time_sub_indices[k * batch_size : (k + 1) * batch_size]
        for k in range(num_batches)
    ]

    remainder = num_times % batch_size
    if remainder > 0:
        time_indices_per_batch.append(time_sub_indices[num_batches * batch_size :])
        num_batches += 1

    return (
        time_indices_per_batch,
        num_batches,
    )


def eval_mean_loss_dynamic(
    model: MulticoilModel,
    loss: MulticoilLoss,
    dataset: CoilData,
    coil_sampler: Callable[[], list[int]],
    num_times: int = 256,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Evaluate the mean multicoil loss over randomly sampled time batches.

    Parameters
    ----------
    model : MulticoilModel
        Multicoil forward model.
    loss : MulticoilLoss
        Multicoil loss instance.
    dataset : CoilData
        Dataset containing k-space observations.
    coil_sampler : Callable[[], list[int]]
        Sampler returning a list of coil indices.
    num_times : int, optional
        Number of time points to sample from the full time grid.
    batch_size : int, optional
        Batch size (over time).
    device : str or torch.device, optional
        Device on which computations are performed.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing averaged loss components.
    """
    spatial_grid_ind_per_dim = [list(range(len(p))) for p in dataset.spatial_partitions]
    full_spatial_grid = [g.to(device) for g in dataset.spatial_partitions]

    time_indices_per_batch, num_batches = init_time_batches(
        dataset.time_partition,
        num_times,
        batch_size,
    )

    mean_loss = {name: torch.tensor(0.0, device=device) for name in loss.names}
    mean_loss["complete_loss"] = torch.tensor(0.0, device=device)

    # TODO: add attribute to MulticoilLoss indicating whether loss component
    # involves an average over time to make routine general purpose. For now
    # hardcode keys ...
    loss_keys_time = ["data_consistency", "dmdt", "dmdx"]
    loss_keys_no_time = ["jac_coil_sense"]

    with torch.no_grad():
        for curr_time_indices in time_indices_per_batch:
            curr_bsize_time = len(curr_time_indices)

            coil_indices = coil_sampler()
            time_subgrid = dataset.time_partition[curr_time_indices].to(device)
            spacetime_grid = [time_subgrid] + full_spatial_grid
            coord_patch = CoordPatch(
                grid_ind_per_dim=[curr_time_indices] + spatial_grid_ind_per_dim,
                rand_subgrid_per_dim=spacetime_grid,
                rand_mixed_grid_per_dim=spacetime_grid,
            )

            loss_batch = eval_loss_dynamic(
                model, loss, dataset, coord_patch, coil_indices, device
            )

            for key in loss_keys_time:
                mean_loss[key] += curr_bsize_time * loss_batch[key] / num_times

            for key in loss_keys_no_time:
                mean_loss[key] += loss_batch[key] / num_times

        for name, weight in loss.weights.items():
            mean_loss["complete_loss"] += weight * mean_loss[name]

    return mean_loss
