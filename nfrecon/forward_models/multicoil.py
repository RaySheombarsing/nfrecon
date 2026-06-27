"""Forward multi-coil MRI model.

This module defines neural-field-based forward models for multi-coil MRI,
including complex-valued magnetization and coil sensitivity maps. Both
components are represented using Neural Field Expansions (NFEs) built on
SIREN backbones.
"""

import torch

from omegaconf import DictConfig
from torch import Tensor, nn

from nfrecon.neural_fields.nfe_factories import siren_nfe


class CoilSensitivity(nn.Module):
    """Normalized coil sensitivity model based on neural field expansions.

    Parameters
    ----------
    coil_cfg : DictConfig
        Configuration for the coil sensitivity neural field expansion.
    num_coils : int
        Number of receive coils.
    """

    def __init__(self, coil_cfg: DictConfig, num_coils: int) -> None:
        super().__init__()
        self._coil_cfg = coil_cfg
        self._num_coils = num_coils
        self._backbone = siren_nfe(**coil_cfg, dim_out=num_coils, complex_valued=True)

    @staticmethod
    def _compute_normalization_factor(coil_sense_raw: Tensor) -> Tensor:
        """Compute the pointwise L2 norm across coil channels.

        Parameters
        ----------
        coil_sense_raw : Tensor
            Complex-valued tensor of shape ``(..., num_coils)`` representing
            unnormalized coil sensitivity maps.

        Returns
        -------
        Tensor
            Real-valued tensor of shape ``(...)`` containing the normalization
            factor at each spatial location.
        """
        return torch.sqrt(
            torch.sum(torch.real(coil_sense_raw * torch.conj(coil_sense_raw)), dim=-1)
        )

    @property
    def num_coils(self) -> int:
        """Return the number of receive coils.

        Returns
        -------
        int
            Number of receive coils.
        """
        return self._num_coils

    def partial_jac(
        self,
        argnums: tuple[int, ...],
        *x: Tensor,
        coil_sense_raw: Tensor | None = None,
        norm_factor: Tensor | None = None,
    ) -> Tensor:
        """Evaluate partial Jacobians of normalized coil sensitivities.

        Parameters
        ----------
        argnums : tuple of int
            Indices of coordinate directions to differentiate with respect to.
        x : Tensor
            One tensor per coordinate direction defining the evaluation grid.
        coil_sense_raw : Tensor, optional
            Precomputed unnormalized coil sensitivities.
        norm_factor : Tensor, optional
            Precomputed normalization factors.

        Returns
        -------
        Tensor
            Complex-valued partial Jacobians with shape
            ``(*grid_shape, num_coils, len(argnums))``.
        """
        if coil_sense_raw is None:
            coil_sense_raw = self._backbone(*x)

        if norm_factor is None:
            norm_factor = self._compute_normalization_factor(coil_sense_raw)

        norm_factor_reciproc = 1.0 / norm_factor
        coil_sense_raw = torch.view_as_real(coil_sense_raw)
        jac_coil_sense_raw = torch.view_as_real(self._backbone.partial_jac(argnums, *x))

        # Leibniz rule
        partial_jac_left = torch.einsum(
            "...cp,...clp->...l", coil_sense_raw, jac_coil_sense_raw
        )
        partial_jac_left = torch.einsum(
            "...l,...->...l", partial_jac_left, norm_factor_reciproc**3
        )
        partial_jac_left = torch.einsum(
            "...l,...cp->...clp", partial_jac_left, coil_sense_raw
        )

        partial_jac = (
            torch.einsum("...,...clp->...clp", norm_factor_reciproc, jac_coil_sense_raw)
            - partial_jac_left
        )

        return torch.view_as_complex(partial_jac)

    def forward(self, *x: Tensor) -> Tensor:
        """Evaluate normalized coil sensitivity maps on a grid.

        Parameters
        ----------
        x : Tensor
            One tensor per spatial dimension defining the evaluation grid.

        Returns
        -------
        Tensor
            Complex-valued coil sensitivity maps of shape
            ``(*grid_shape, num_coils)``.
        """
        coil_sense_raw = self._backbone(*x)
        norm_factor = self._compute_normalization_factor(coil_sense_raw)
        return torch.einsum("...c,...->...c", coil_sense_raw, 1.0 / norm_factor + 0j)


class MulticoilModel(nn.Module):
    """Forward multi-coil MRI model.

    This model combines neural field expansions for both the
    magnetization and coil sensitivities.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration for magnetization and coil sensitivity models.
    num_coils : int
        Number of receive coils.
    dynamic : bool
        If ``True``, magnetization is treated as time-dependent.
    """

    def __init__(self, model_cfg: DictConfig, num_coils: int, dynamic: bool) -> None:
        super().__init__()

        self._cfg = model_cfg
        self._num_coils = num_coils
        self._dynamic = dynamic

        self.magnetization = siren_nfe(
            **model_cfg.magnetization, dim_out=1, complex_valued=True
        )
        self.coil_sensitivity = CoilSensitivity(model_cfg.coil_sensitivity, num_coils)

        if self._dynamic:
            self._spatial_dim = len(model_cfg.magnetization.num_modes) - 1
            self._magnetization_spatial_indices = tuple(
                range(self._spatial_dim, 0, -1)
            )  # Order is (x,y,z)
            self.forward = self._forward_dynamic
        else:
            self._spatial_dim = len(model_cfg.magnetization.num_modes)
            self._magnetization_spatial_indices = tuple(
                range(self._spatial_dim - 1, -1, -1)
            )  # Order is (x,y,z)
            self.forward = self._forward_static

        self._coil_spatial_indices = tuple(range(self._spatial_dim))  # Order is (z,y,x)

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
        """Indicates whether model is time-dependent.

        Returns
        -------
        bool
            ``True`` if the model is time-dependent.
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
    def coil_spatial_indices(self) -> tuple[int, ...]:
        """Return spatial indices associated with coil sensitivity maps.

        Returns
        -------
        tuple[int, ...]
            Indices (spatial) components coil sensitivies.
        """
        return self._coil_spatial_indices

    @property
    def magnetization_spatial_indices(self) -> tuple[int, ...]:
        """Return spatial indices associated with magnetization field.

        Returns
        -------
        tuple[int, ...]
            Indices spatial components magnetization.
        """
        return self._magnetization_spatial_indices

    def _forward_static(self, *x: Tensor) -> dict[str, Tensor]:
        """Evaluate magnetization and coil sensitivity maps (static case).

        Parameters
        ----------
        x : Tensor
            One tensor per spatial coordinate direction defining the grid.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary with entries:
            - ``"magnetization"``: complex-valued magnetization evaluated on
              the spatial grid.
            - ``"coil_sensitivity"``: complex-valued coil sensitivity maps
              evaluated on the spatial grid.
        """
        return {
            "magnetization": self.magnetization(*x),
            "coil_sensitivity": self.coil_sensitivity(*x),
        }

    def _forward_dynamic(self, *y: Tensor) -> dict[str, Tensor]:
        """Evaluate magnetization and coil sensitivity maps (dynamic case).

        Parameters
        ----------
        y : Tensor
            One tensor per coordinate direction defining the spacetime grid.
            The first tensor corresponds to time; remaining tensors define
            the spatial grid.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary with entries:
            - ``"magnetization"``: complex-valued magnetization evaluated on
              the spacetime grid.
            - ``"coil_sensitivity"``: complex-valued coil sensitivity maps
              evaluated on the spatial grid.
        """
        return {
            "magnetization": self.magnetization(*y),
            "coil_sensitivity": self.coil_sensitivity(*y[1:]),
        }
