"""Implementation of Neural Field Expansions (NFE).

This module provides a tensor-product neural field expansion constructed from
univariate neural fields along each coordinate direction. The implementation
supports real- and complex-valued fields and efficient evaluation of partial
Jacobians and Hessians via automatic differentiation.
"""

import torch

from typing import Callable, Sequence
from torch import Tensor, nn
from torch.func import jacrev, vmap

NeuralField = Callable[Tensor, Tensor]


class NeuralFieldExpansion(nn.Module):
    """Neural Field Expansion (NFE).

    A neural field expansion represents a multivariate function as a tensor
    product of univariate neural fields, combined with learned coefficients.

    Parameters
    ----------
    modes_per_dim : sequence of NeuralField
        Univariate neural fields for each coordinate direction. Each field must
        expose ``dim_out``, ``out_shape``, and ``complex_valued`` attributes.
        The output dimension of each field corresponds to the number of modes
        along that coordinate direction.
    dim_out : int
        Output dimension of the expansion.
    """

    def __init__(
        self, modes_per_dim: Sequence[NeuralField] | nn.ModuleList, dim_out: int
    ) -> None:
        super().__init__()

        self._dim_in = len(modes_per_dim)
        self._dim_out = dim_out
        self._modes_per_dim = nn.ModuleList(modes_per_dim)

        self._infer_number_field()
        self._infer_num_modes()
        self._init_shape_formatters()
        self._init_coeffs()

        self.first_deriv_modes = [
            self._init_first_deriv(modes) for modes in self._modes_per_dim
        ]
        self.second_deriv_modes = [
            self._init_second_deriv(modes) for modes in self._modes_per_dim
        ]

    def _infer_number_field(self) -> None:
        """Infer whether the expansion is real- or complex-valued."""
        num_complex_valued = 0
        for modes in self._modes_per_dim:
            num_complex_valued += modes.complex_valued

        if num_complex_valued == 0:
            self._complex_valued = False
        elif num_complex_valued == self._dim_in:
            self._complex_valued = True
        else:
            raise ValueError(
                "All mode networks must be either real- or complex-valued."
            )

    def _infer_num_modes(self) -> None:
        """Infer the number of modes per dimension from the mode networks."""
        self._num_modes: list[int] = []
        for modes in self._modes_per_dim:
            if len(modes.out_shape) != 1:
                raise ValueError(
                    "Mode networks must output one-dimensional arrays of modes."
                )
            self._num_modes.append(modes.dim_out)

    def _init_shape_formatters(self) -> None:
        """Initialize formatters for real/complex conversion and final reshaping."""
        if self._complex_valued:
            self._dtype = torch.complex64
            self._to_number_field = torch.view_as_complex
            self._to_number_field_deriv = torch.view_as_real
        else:
            self._dtype = torch.float32
            self._to_number_field = lambda x: x
            self._to_number_field_deriv = lambda x: x

        if self._dim_out == 1:
            self._final_reshape = lambda x: torch.squeeze(x, dim=-1)
        else:
            self._final_reshape = lambda x: x

    def _init_coeffs(self) -> None:
        """Initialize expansion coefficients."""
        if self._complex_valued:
            coeffs_real = torch.nn.init.orthogonal_(
                torch.empty(*self._num_modes, self._dim_out)
            )
            coeffs_imag = torch.nn.init.orthogonal_(
                torch.empty(*self._num_modes, self._dim_out)
            )
            self._coeffs = torch.nn.Parameter(coeffs_real + 1j * coeffs_imag)
        else:
            self._coeffs = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty(*self._num_modes, self._dim_out))
            )

    @property
    def dim_in(self) -> int:
        """Return the input dimensionality of the expansion.

        Returns
        -------
        int
            Number of coordinate directions (input dimensions).
        """
        return self._dim_in

    @property
    def complex_valued(self) -> bool:
        """Return whether the expansion is complex-valued.

        Returns
        -------
        bool
            ``True`` if the expansion produces complex-valued outputs.
        """
        return self._complex_valued

    def _eval_nfe(
        self,
        coeffs: Tensor,
        modes_per_dim: Sequence[NeuralField] | nn.ModuleList,
        *x: Tensor,
    ) -> Tensor:
        """Evaluate the neural field expansion at the given grid.

        Parameters
        ----------
        coeffs : Tensor
            Expansion coefficients of shape ``(*num_modes, dim_out)``.
        modes_per_dim : sequence of NeuralField
            Univariate neural fields defining modes for each coordinate direction.
        x : Tensor
            One tensor per coordinate direction defining the evaluation grid.

        Returns
        -------
        Tensor
            Expansion evaluated on the full tensor-product grid with shape
            ``(num_points_dim_1, ..., num_points_dim_d, dim_out)``.
        """
        y = coeffs
        for pts, modes in zip(x, modes_per_dim):
            nn_eval = modes(pts.unsqueeze(-1))
            y = torch.einsum("k...d,bk->...bd", y, nn_eval)

        return self._final_reshape(y)

    def _init_first_deriv(
        self, modes: NeuralField | torch.nn.Module
    ) -> Callable[Tensor, Tensor]:
        """Compute first derivative modes in specific coordinate direction.

        Parameters
        ----------
        modes : NeuralField | torch.nn.Module
            modes in specific coordinate direction.

        Returns
        -------
        Callable[Tensor, Tensor]
            first derivative modes in specific coordinate direction.
        """
        deriv = vmap(
            jacrev(
                lambda p: self._to_number_field_deriv(
                    modes(p.unsqueeze(0).unsqueeze(-1)).squeeze(0)
                ),
                argnums=0,
            )
        )
        return lambda p: self._to_number_field(deriv(p.squeeze(-1)))

    def _init_second_deriv(
        self, modes: NeuralField | torch.nn.Module
    ) -> Callable[Tensor, Tensor]:
        """Compute second derivative modes in specific coordinate direction.

        Parameters
        ----------
        modes : NeuralField | torch.nn.Module
            modes in specific coordinate direction.

        Returns
        -------
        Callable[Tensor, Tensor]
            second derivative modes in specific coordinate direction.
        """
        first_deriv = jacrev(
            lambda p: self._to_number_field_deriv(
                modes(p.unsqueeze(0).unsqueeze(-1)).squeeze(0)
            ),
            argnums=0,
        )
        second_deriv = vmap(jacrev(first_deriv, argnums=0))

        return lambda p: self._to_number_field(second_deriv(p.squeeze(-1)))

    def partial_jac(self, argnums: tuple[int, ...], *x: Tensor) -> Tensor:
        """Evaluate selected components of the Jacobian on a grid.

        Parameters
        ----------
        argnums : tuple of int
            Indices of coordinate directions to differentiate with respect to.
        x : Tensor
            One tensor per coordinate direction defining the evaluation grid.

        Returns
        -------
        Tensor
            Partial Jacobian evaluated on the grid with shape
            ``(*grid_shape, dim_out, len(argnums))``.
        """
        jac = []
        for j in argnums:
            deriv_modes_per_dim = list(self._modes_per_dim)
            deriv_modes_per_dim[j] = self.first_deriv_modes[j]
            jac.append(self._eval_nfe(self._coeffs, deriv_modes_per_dim, *x))

        return torch.stack(jac, dim=-1)

    def partial_hess(self, argnums: tuple[tuple[int, int], ...], *x: Tensor) -> Tensor:
        """Evaluate selected components of the Hessian on a grid.

        Parameters
        ----------
        argnums : tuple of (int, int)
            Pairs of coordinate directions defining second derivatives.
        x : Tensor
            One tensor per coordinate direction defining the evaluation grid.

        Returns
        -------
        Tensor
            Partial Hessian evaluated on the grid with shape
            ``(*grid_shape, dim_out, len(argnums))``.
        """
        hess = []
        for i, j in argnums:
            deriv_modes_per_dim = list(self._modes_per_dim)
            if i != j:
                deriv_modes_per_dim[i] = self.first_deriv_modes[i]
                deriv_modes_per_dim[j] = self.first_deriv_modes[j]
            else:
                deriv_modes_per_dim[i] = self.second_deriv_modes[i]

            hess.append(self._eval_nfe(self._coeffs, deriv_modes_per_dim, *x))

        return torch.stack(hess, dim=-1)

    def forward(self, *x: Tensor) -> Tensor:
        """Evaluate the neural field expansion on a grid.

        Parameters
        ----------
        x : Tensor
            One tensor per coordinate direction defining the evaluation grid.

        Returns
        -------
        Tensor
            Expansion evaluated on the tensor-product grid.
        """
        return self._eval_nfe(self._coeffs, self._modes_per_dim, *x)
