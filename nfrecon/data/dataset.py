"""Dataset utilities for handling multicoil MRI k-space data.

This module defines a class for loading, normalizing, and organizing raw
multicoil k-space measurements into a form suitable for training neural
field-based MRI reconstruction models.
"""

import logging
import torch
import numpy as np

from pathlib import Path
from typing import Sequence, Optional
from torch import Tensor
from omegaconf import DictConfig

from nfrecon.utils import io
from nfrecon.samplers.grids import construct_grid

logger = logging.getLogger(__name__)


class CoilData:
    """Container for multicoil k-space data and associated metadata.

    This class loads raw k-space data from disk, applies normalization and
    thresholding, and exposes spatial and temporal grids required for neural
    field training.

    Parameters
    ----------
    data_path : Path or str
        Path to a directory or file containing the raw k-space dataset.
    data_keys : DictConfig
        Hydra config providing attribute-style access to dataset keys (e.g.
        ``kspace``, ``kspace_coords``, ``fov``, ``gridsize``, ``time_step``).
    spatial_dim : int
        Number of spatial dimensions.
    dynamic : bool
        If ``True``, the k-space data is treated as time-dependent.
    freq_threshold : float
        Frequency components with magnitude below this threshold are discarded.
    time_scale : float, optional
        Scaling factor applied to temporal sampling.
    percentile_scaling : float, optional
        Percentile used to compute the k-space normalization factor.
    """

    def __init__(
        self,
        data_path: Path | str,
        data_keys: DictConfig,
        spatial_dim: int,
        dynamic: bool,
        freq_threshold: float,
        time_scale: Optional[float],
        percentile_scaling: float = 99.5,
    ) -> None:
        self._spatial_dim = spatial_dim
        self._dynamic = dynamic
        self._freq_threshold = freq_threshold
        self._percentile_scaling = percentile_scaling
        self._time_scale = time_scale

        self._data_path = Path(data_path)
        self._data_keys = data_keys

        data = io.load_array_data(self._data_path)
        self._init_kspace_data(data)

        # Spatial information
        self._fov = data[data_keys.fov]
        self._spatial_domain = [[-s / 2, s / 2] for s in self._fov]
        self._npts_per_spatial_dim = data[data_keys.gridsize]

        _, self._spatial_partitions = construct_grid(
            Tensor(self._spatial_domain),
            self._npts_per_spatial_dim,
            device="cpu",
        )

        self._fully_sampled = (
            np.prod(self._npts_per_spatial_dim) == self._kspace.shape[-1]
        )

        self.print_data_info()

    def _init_kspace_data(
        self, data: dict[str, np.ndarray | dict[str, np.ndarray]]
    ) -> None:
        """Initialize and normalize raw k-space data.

        Parameters
        ----------
        data : dict
            Dictionary containing raw k-space data and associated metadata.
        """
        if self._dynamic:
            # Infer temporal information
            self._num_times = sum(self._data_keys.kspace in key for key in data.keys())
            self._time_step = self._time_scale * float(data[self._data_keys.time_step])
            self._time_partition = self._time_step * torch.arange(self._num_times)
            self._time_domain = [self.time_partition[0], self.time_partition[-1]]

            # Collect $k$-space data per time frame (vectorized storage)
            kspace_raw = [
                data[f"{self._data_keys.kspace}_{t}"] for t in range(self._num_times)
            ]
            self._num_coils = kspace_raw[0].shape[0]
            max_num_pts = max([y.shape[-1] for y in kspace_raw])

            kspace = np.zeros(
                (self._num_coils, self._num_times, max_num_pts), dtype=np.complex64
            )
            kspace_mask = np.ones((self._num_times, max_num_pts), dtype=bool)
            kspace_coords = np.full(
                (self._num_times, max_num_pts, self._spatial_dim), -1, dtype=int
            )

            for t, y in enumerate(kspace_raw):
                num_pts = y.shape[-1]
                kspace[:, t, :num_pts] = y
                kspace_mask[t, num_pts:] = False
                kspace_coords[t, :num_pts] = data[
                    f"{self._data_keys.kspace_coords}_{t}"
                ]

            self._kspace_coords = torch.from_numpy(kspace_coords)
            self._kspace_mask = torch.from_numpy(kspace_mask)

        else:
            kspace = data[self._data_keys.kspace]
            self._kspace_coords = torch.from_numpy(data[self._data_keys.kspace_coords])
            self._num_coils = kspace.shape[0]
            self._time_partition = None
            self._time_domain = None

        if "scale_factor" in self._data_keys:
            self._scale_factor = data[self._data_keys.scale_factor]
        else:
            self._scale_factor = np.percentile(np.abs(kspace), self._percentile_scaling)

        self._kspace = torch.from_numpy(kspace / self._scale_factor)
        self._kspace[torch.abs(self._kspace) < self._freq_threshold] = 0.0

    def _init_sampling_density(self) -> None:
        """Compute sampling density weights for dynamic k-space data."""
        logger.info("Compute sampling density ...")
        invalid_coord = (-1,) * self._spatial_dim

        num_times, max_num_pts = self._kspace_coords.shape[:2]
        coords_freq: dict[tuple[int, ...], float] = {}
        for coords in self._kspace_coords:
            for c in coords:
                c_tuple = tuple(int(x) for x in c)
                if c_tuple in coords_freq:
                    coords_freq[c_tuple] += 1.0
                elif c_tuple != invalid_coord:
                    coords_freq[c_tuple] = 1.0

        self._sampling_density = np.ones((num_times, max_num_pts), dtype=np.float32)
        for t_idx, coords in enumerate(self._kspace_coords):
            for k_idx, c in enumerate(coords):
                c_tuple = tuple(int(x) for x in c)
                if c_tuple in coords_freq:
                    self._sampling_density[t_idx, k_idx] = 1.0 / coords_freq[c_tuple]

    def print_data_info(self) -> None:
        """Log a summary of the loaded dataset."""
        logger.info(f"Successfully initialized MRI data.")
        logger.info(f"Fully sampled: {self._fully_sampled}")
        logger.info(f"Dynamic: {self._dynamic}")
        logger.info(f"Number of coils: {self._num_coils}.")
        logger.info(f"Spatial grid size: {self._npts_per_spatial_dim}.")
        logger.info(f"Field of view: {self._fov}.")
        logger.info(f"k-space scale factor: {self._scale_factor}")

        if self._dynamic:
            logger.info(f"Temporal step size: {self._time_step}.")
            logger.info(f"Number of temporal observations: {self._num_times}.")

        if not self._fully_sampled:
            accel = np.prod(self._npts_per_spatial_dim) / self._kspace.shape[-1]
            logger.info(f"Estimated acceleration factor: {accel}")

    @property
    def spatial_dim(self) -> int:
        """Return number of spatial dimensions.

        Returns
        -------
        int
            Number of spatial dimensions.
        """
        return self._spatial_dim

    @property
    def dynamic(self) -> bool:
        """Indicates whether the kspace data is time-dependent.

        Returns
        -------
        bool
            ``True`` if the k-space data is time-dependent.
        """
        return self._dynamic

    @property
    def num_coils(self) -> int:
        """Return the number of receive coils.

        Returns
        -------
        int
            Number of receive coils.
        """
        return self._num_coils

    @property
    def fully_sampled(self) -> bool:
        """Return whether the k-space data is fully sampled.

        Returns
        -------
        bool:
            ``True`` if the k-space data is fully sampled.
        """
        return self._fully_sampled

    @property
    def time_scale(self) -> float | None:
        """Return the temporal scaling factor.

        Returns
        -------
        float or None
            Temporal scaling factor if the dataset is dynamic, otherwise ``None``.
        """
        return self._time_scale

    @property
    def time_domain(self) -> Tensor | None:
        """Return the temporal domain, if applicable.

        Returns
        -------
        Tensor or None
            Time domain (interval) if the dataset is dynamic, otherwise ``None``.
        """
        return self._time_domain

    @property
    def time_partition(self) -> Tensor | None:
        """Return the temporal sampling grid, if applicable.

        Returns
        -------
        Tensor or None
            Temporal sampling grid if the dataset is dynamic, otherwise ``None``.
        """
        return self._time_partition

    @property
    def spatial_domain(self) -> Sequence[Tensor]:
        """Return spatial domain each coordinate direction.

        Returns
        -------
        Sequence[Tensor]
            Spatial domain (intervals) per coordinate direction.
        """
        return self._spatial_domain

    @property
    def spatial_partitions(self) -> Sequence[Tensor]:
        """Return spatial sampling grids for each coordinate direction.

        Returns
        -------
        Sequence[Tensor]
            Spatial sampling grids per coordinate direction.
        """
        return self._spatial_partitions

    @property
    def kspace(self) -> Tensor:
        """Return the normalized k-space measurements.

        Returns
        -------
        Tensor
            Normalized k-space measurements.
            Shape is (num_coils, num_times, num_kspace_obs).
        """
        return self._kspace

    @property
    def kspace_mask(self) -> Tensor | None:
        """Return k-space mask to indicating artificially appended coordinates.

        Returns
        -------
        Tensor or None
            Boolean mask indicating valid k-space samples
            Shape is (num_times, num_kspace_obs).
        """
        return self._kspace_mask

    @property
    def kspace_coords(self) -> Tensor:
        """Return k-space sampling coordinates.

        Returns
        -------
        Tensor
            Sampling coordinates of k-space observations.
            Shape is (num_times, num_kspace_obs, spatial_dim).
        """
        return self._kspace_coords

    def __len__(self) -> int:
        """Return the number of acquired k-space samples.

        Returns
        -------
        int
            Total number of acquired k-space points.
        """
        return int(self._kspace_mask.sum())
