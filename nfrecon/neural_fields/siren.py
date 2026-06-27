"""Implementation of SIREN neural fields.

Based on:
Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions".
"""

import numpy as np
import torch

from torch import Tensor, nn
from typing import Optional

from nfrecon.neural_fields import number_field_utils as field_utils


class SineLayer(nn.Module):
    """Linear layer followed by a sinusoidal activation.

    Parameters
    ----------
    dim_in : int
        Dimension input.
    dim_out : int
        Dimension output.
    bias : bool, optional
        Whether to include a bias term.
    multiply_before_activation : bool, optional
        If ``True``, apply frequency embedding before the linear transformation.
        This is typically used for the first layer in SIREN networks.
    omega : float, optional
        Frequency embedding factor.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        multiply_before_activation: bool = False,
        omega: float = 30.0,
    ) -> None:
        super().__init__()

        self._dim_in = dim_in
        self._dim_out = dim_out
        self._bias = bias
        self._multiply_before_activation = multiply_before_activation
        self._omega = omega

        self._layer = nn.Linear(self._dim_in, self._dim_out, bias=self._bias)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters linear layer."""
        with torch.no_grad():
            if self._multiply_before_activation:
                self._layer.weight.uniform_(-1 / self._dim_in, 1 / self._dim_in)
            else:
                self._layer.weight.uniform_(
                    -np.sqrt(6 / self._dim_in) / self._omega,
                    np.sqrt(6 / self._dim_in) / self._omega,
                )

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate SIREN layer at prescribed set of points.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(*batch_dims, dim_in)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(*batch_dims, *out_shape)``.
        """
        return torch.sin(self._omega * self._layer(x))


class Siren(nn.Module):
    """Implementation SIREN network.

    Parameters
    ----------
    dim_in : int
        Dimension input.
    dim_out : int
        Dimension output.
    dim_latent : int, optional
        Width of the hidden representation.
    num_hidden_layers : int, optional
        Number of hidden layers.
    out_shape : tuple of int, optional
        Final output shape (excluding batch dimensions). If ``None``, defaults
        to ``(dim_out,)``.
    layer_norm : bool, optional
        If ``True``, apply layer normalization after each linear layer.
    init_omega: float
        frequency embedding parameter in first layer.
    hidden_omega: float
        frequency embedding parameter second layer.
    complex_valued : bool, optional
        If ``True``, interpet the output as complex-valued.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_latent: int = 32,
        num_hidden_layers: int = 1,
        out_shape: Optional[tuple[int, ...]] = None,
        init_omega: float = 30.0,
        hidden_omega: float = 30.0,
        complex_valued: bool = False,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("Siren must have at least one hidden layer.")

        self._dim_in = dim_in
        self._dim_out = dim_out
        self._dim_latent = dim_latent
        self._num_hidden_layers = num_hidden_layers
        self._init_omega = init_omega
        self._hidden_omega = hidden_omega
        self._complex_valued = complex_valued

        self._out_shape = out_shape if out_shape is not None else (self._dim_out,)
        self._dim_out_real, self._to_number_field = field_utils.format_field(
            complex_valued, dim_out
        )

        # Initial layer: mapping to latent space
        layers = []

        layers.append(
            SineLayer(
                self._dim_in,
                self._dim_latent,
                multiply_before_activation=True,
                omega=self._init_omega,
            )
        )
        curr_dim = self._dim_latent

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(
                SineLayer(
                    curr_dim,
                    curr_dim,
                    multiply_before_activation=False,
                    omega=self._hidden_omega,
                ),
            )

        out_layer = nn.Linear(curr_dim, self._dim_out_real)
        with torch.no_grad():
            out_layer.weight.uniform_(
                -np.sqrt(6 / self._dim_latent) / self._hidden_omega,
                np.sqrt(6 / self._dim_latent) / self._hidden_omega,
            )
        layers.append(out_layer)

        self._network = nn.Sequential(*layers)

    @property
    def complex_valued(self) -> bool:
        """Return whether the network is complex-valued.

        Returns
        -------
        bool
            True if network is complex-valued.
        """
        return self._complex_valued

    @property
    def dim_out(self) -> int:
        """Return dimension of network output.

        Returns
        -------
        int
            Dimension network output.
        """
        return self._dim_out

    @property
    def out_shape(self) -> tuple[int, ...]:
        """Return shape network output.

        Returns
        -------
        tuple[int,...]
            shape network output.
        """
        return self._out_shape

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate SIREN network at prescribed set of points.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(*batch_dims, dim_in)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(*batch_dims, *out_shape)``.
        """
        batch_shape = x.shape[:-1]
        y = self._to_number_field(self._network(x))
        return y.view(*batch_shape, *self._out_shape)
