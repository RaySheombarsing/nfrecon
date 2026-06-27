"""Grid and domain samplers.

This module provides simple sampling utilities for two common settings:

- Sampling from a discrete grid, defined as the Cartesian product of
  finite sets in each coordinate direction.
- Sampling from a continuous rectangular domain, defined by intervals
  in each coordinate direction.

All samplers use uniform distributions.
"""

import torch
import numpy as np

from torch import Tensor


class GridSampler:
    """Random sampler from a fixed discrete grid (uniform distribution).

    Parameters
    ----------
    partitions : tuple[Tensor, ...]
        Finite set of points for each coordinate direction.
    num_samples_per_dim : tuple[int, ...]
        Number of points to sample (without replacement) per dimension.
    """

    def __init__(
        self,
        partitions: tuple[Tensor, ...],
        num_samples_per_dim: tuple[int, ...],
    ) -> None:
        self._partitions = partitions
        self._size_partitions = [len(p) for p in partitions]
        self._indices_per_partition = [np.arange(d) for d in self._size_partitions]
        self._num_samples_per_dim = num_samples_per_dim

    def __call__(
        self, device: str | torch.device = "cpu"
    ) -> tuple[list[list[int]], list[Tensor]]:
        """Sample points uniformly from the discrete grid.

        Parameters
        ----------
        device : str or torch.device, optional
            Device on which to place the returned tensors.

        Returns
        -------
        indices_per_dim : list[list[int]]
            Randomly chosen indices for each coordinate direction.
        subgrid_per_dim : list[Tensor]
            Sampled grid points for each coordinate direction.
        """
        indices_per_dim = [
            np.random.choice(ran, size=n, replace=False).tolist()
            for ran, n in zip(self._indices_per_partition, self._num_samples_per_dim)
        ]
        subgrid_per_dim = [
            p[indices].to(device)
            for p, indices in zip(self._partitions, indices_per_dim)
        ]
        return indices_per_dim, subgrid_per_dim


class DomainSampler:
    """Random sampler from a rectangular continuous domain (uniform distribution).

    Parameters
    ----------
    intervals : tuple[Tensor | tuple[float, float], ...]
        Intervals defining the rectangular domain in each coordinate direction.
        Each interval is specified as ``(a, b)``.
    num_samples_per_dim : tuple[int, ...]
        Number of points to sample per dimension.
    """

    def __init__(
        self,
        intvals: tuple[Tensor | tuple[float, float], ...],
        num_samples_per_dim: tuple[int, ...],
    ) -> None:
        self._intvals = intvals
        self._num_samples_per_dim = num_samples_per_dim

    def __call__(self, device: str | torch.device = "cpu") -> list[Tensor]:
        """Sample points uniformly from the rectangular domain.

        Parameters
        ----------
        device : str or torch.device, optional
            Device on which to place the returned tensors.

        Returns
        -------
        subgrid_per_dim : list[Tensor]
            Sampled points for each coordinate direction.
        """
        subgrid_per_dim: list[Tensor] = []

        for npts, (a, b) in zip(self._num_samples_per_dim, self._intvals):
            pts = (b - a) * torch.rand(npts) + a
            subgrid_per_dim.append(pts.to(device))

        return subgrid_per_dim
