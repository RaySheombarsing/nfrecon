"""Utilities for initializing samplers.

This module provides helper routines to initialize sampling objects used
throughout the training pipeline, including samplers for coil indices,
discrete grids, and continuous domains.
"""

import torch

from omegaconf import DictConfig
from typing import Callable

from nfrecon.samplers.samplers import GridSampler, DomainSampler
from nfrecon.data.dataset import CoilData


def init_samplers(cfg_sampler: DictConfig, dataset: CoilData) -> tuple[
    Callable[[], list[int]],
    GridSampler,
    DomainSampler,
]:
    """Initialize coil, discrete-grid, and continuous-domain samplers.

    Parameters
    ----------
    cfg_sampler : DictConfig
        Sampler configuration. Expected to define ``num_coils`` and
        ``num_samples_per_dim`` entries for discrete and continuous sampling.
    dataset : CoilData
        Dataset providing spatial and (optionally) temporal grids.

    Returns
    -------
    coil_sampler : Callable[[], list[int]]
        Sampler for coil indices.
    grid_sampler : GridSampler
        Sampler for the discrete (observed) space(time) grid.
    domain_sampler : DomainSampler
        Sampler for the continuous space(time) domain.
    """
    if dataset.dynamic:
        partitions = [dataset.time_partition] + dataset.spatial_partitions
        domain = [dataset.time_domain] + dataset.spatial_domain
    else:
        partitions = dataset.spatial_partitions
        domain = dataset.spatial_domain

    if cfg_sampler.num_coils:
        coil_batch_size = cfg_sampler.num_coils
    else:
        coil_batch_size = dataset.num_coils

    coil_grid_sampler = GridSampler(
        (torch.arange(dataset.num_coils),), (coil_batch_size,)
    )

    # For the coil sampler we only need the indices of the coils
    def coil_sampler() -> list[int]:
        _, coil_indices = coil_grid_sampler()
        return coil_indices[0].tolist()

    grid_sampler = GridSampler(partitions, cfg_sampler.num_samples_per_dim.discrete)
    domain_sampler = DomainSampler(domain, cfg_sampler.num_samples_per_dim.continuous)

    return coil_sampler, grid_sampler, domain_sampler
