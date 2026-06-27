"""Utilities for constructing and sampling grids.

This module provides helper functions and data structures to construct
Cartesian grids on rectangular domains and to sample coordinate patches
from both discrete grids and continuous domains.
"""

import torch

from torch import Tensor
from dataclasses import dataclass

from nfrecon.samplers.samplers import GridSampler, DomainSampler


def construct_grid(
    domain: Tensor, npts_per_dim: tuple[int], device: str | torch.device = "cpu"
) -> tuple[Tensor, list[Tensor]]:
    """Construct a Cartesian grid on a rectangular domain.

    Parameters
    ----------
    domain : Tensor
        Tensor of shape ``(spatial_dim, 2)`` defining the lower and upper
        bounds of the rectangular domain.
    npts_per_dim : tuple of int
        Number of grid points per coordinate direction.
    device : str or torch.device, optional
        Device on which to allocate the grid.

    Returns
    -------
    mesh_size : Tensor
        Mesh size per coordinate direction, with shape ``(spatial_dim,)``.
    grid_per_dim : list[Tensor]
        One-dimensional coordinate grids for each dimension.
    """
    domain = domain.to(device)
    grid_indices = [torch.arange(d, device=device) for d in npts_per_dim]

    mesh_size = (domain[:, 1] - domain[:, 0]) / (
        torch.Tensor(npts_per_dim).to(device) - 1
    )
    grid_per_dim = [
        delta * indices + a
        for delta, indices, a in zip(mesh_size, grid_indices, domain[:, 0])
    ]

    return mesh_size, grid_per_dim


@dataclass
class CoordPatch:
    """Randomly sampled coordinate patch.

    This data structure groups together information from two sampling
    strategies:

    - Discrete sampling from a fixed grid (``grid_ind_per_dim`` and
      ``rand_subgrid_per_dim``).
    - Mixed sampling combining discrete grid points with continuous
      domain samples (``rand_mixed_grid_per_dim``).
    """

    grid_ind_per_dim: list[list[int]]
    rand_subgrid_per_dim: list[Tensor]
    rand_mixed_grid_per_dim: list[Tensor]


def sample_coordinate_patch(
    grid_sampler: GridSampler, domain_sampler: DomainSampler, device: str | torch.device
) -> CoordPatch:
    """Sample a discrete and a mixed discrete/continuous coordinate patch.

    Parameters
    ----------
    grid_sampler : GridSampler
        Sampler for discrete grid points.
    domain_sampler : DomainSampler
        Sampler for continuous domain points.
    device : str or torch.device
        Device on which to place the sampled tensors.

    Returns
    -------
    CoordPatch
        Sampled coordinate patch combining discrete grid points and
        continuous domain samples.
    """
    # Sample uniformly from grid (discrete)
    grid_ind_per_dim, subgrid_per_dim = grid_sampler(device=device)

    # Sample uniformly from full domain (continuous)
    rand_grid_per_dim = domain_sampler(device=device)

    # Take refinement of the two sampled partitions
    refinements = [
        torch.cat((p1, p2), dim=0) for p1, p2 in zip(subgrid_per_dim, rand_grid_per_dim)
    ]

    return CoordPatch(
        grid_ind_per_dim=grid_ind_per_dim,
        rand_subgrid_per_dim=subgrid_per_dim,
        rand_mixed_grid_per_dim=refinements,
    )
