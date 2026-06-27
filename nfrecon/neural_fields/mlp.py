"""Fully connected multilayer perceptron with residual connections.

This module provides a compact MLP implementation intended for use as a
building block in neural field models. It supports:

- Residual connections between hidden layers
- Optional layer normalization
- Optional complex-valued outputs (represented via real/imag pairs and
  converted using ``torch.view_as_complex``)

The implementation is deliberately lightweight.
"""

import torch

from torch import Tensor, nn
from typing import Callable, Optional

from nfrecon.neural_fields import number_field_utils as field_utils

Activation = Callable[[Tensor], Tensor] | nn.Module


class MLP(nn.Module):
    """Fully connected (residual) multilayer perceptron.

    Parameters
    ----------
    dim_in : int
        Dimension input.
    dim_out : int
        Dimension output.
    dim_latent : int, optional
        Width of the hidden representation.
    num_hidden_layers : int, optional
        Number of residual hidden layers.
    activation : nn.Module, optional
        Activation function applied after each linear layer.
        Must be an instantiated module (e.g. ``nn.Tanh()``).
    out_shape : tuple of int, optional
        Final output shape (excluding batch dimensions). If ``None``, defaults
        to ``(dim_out,)``.
    layer_norm : bool, optional
        If ``True``, apply layer normalization after each linear layer.
    complex_valued : bool, optional
        If ``True``, interpet the output as complex-valued.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_latent: int = 32,
        num_hidden_layers: int = 1,
        activation: Activation = nn.Tanh(),
        out_shape: Optional[tuple[int, ...]] = None,
        layer_norm: bool = False,
        complex_valued: bool = False,
    ) -> None:
        super().__init__()

        self._dim_in = dim_in
        self._dim_latent = dim_latent
        self._dim_out = dim_out
        self._num_hidden_layers = num_hidden_layers
        self._activation = activation
        self._layer_norm = layer_norm
        self._complex_valued = complex_valued

        self._out_shape = out_shape if out_shape is not None else (self._dim_out,)
        self._dim_out_real, self._to_number_field = field_utils.format_field(
            complex_valued, dim_out
        )

        # Initial layer: mapping to latent space
        if self._num_hidden_layers > 0:
            self._to_latent = nn.Linear(self._dim_in, self._dim_latent)
            curr_dim = self._dim_latent
        else:
            self._to_latent = nn.Identity()
            curr_dim = self._dim_in

        # Residual hidden layers
        self._hidden_layers = nn.ModuleList([])
        for _ in range(num_hidden_layers):
            layers = [
                nn.Linear(curr_dim, curr_dim),
            ]
            if self._layer_norm:
                layers.append(nn.LayerNorm(curr_dim))
            self._hidden_layers.append(nn.Sequential(*layers))

        self._out_layer = nn.Linear(curr_dim, self._dim_out_real)

    @property
    def complex_valued(self) -> bool:
        """Return whether the network is complex-valued.

        Returns
        -------
        complex_valued : bool
            True if the network is complex-valued.
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
        """Evaluate MLP at prescribed set of points.

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

        y = self._activation(self._to_latent(x))
        for layer in self._hidden_layers:
            y = self._activation(layer(y) + y)

        y = self._to_number_field(self._out_layer(y))

        return y.view(*batch_shape, *self._out_shape)
