"""Factory functions for constructing Neural Field Expansions (NFEs).

This module collects canonical construction patterns for
:class:`~nfrecon.neural_fields.nfe.NeuralFieldExpansion` instances. Each factory
function wires together a specific backbone neural field (e.g. SIREN networks)
with the expansion machinery, without embedding architectural decisions into
core NFE components.
"""

from typing import Sequence
from torch import nn

from nfrecon.neural_fields.siren import Siren
from nfrecon.neural_fields.nfe import NeuralFieldExpansion


def siren_nfe(
    dim_out: int,
    num_modes: tuple[int, ...],
    dim_latent: int = 32,
    num_hidden_layers: int = 1,
    init_omega: float | Sequence[float] = 15.0,
    hidden_omega: float | Sequence[float] = 15.0,
    complex_valued: bool = False,
) -> NeuralFieldExpansion:
    """Construct a Neural Field Expansion with SIREN mode networks.

    Each coordinate direction is represented by an independent SIREN network
    whose outputs define the modes along that dimension. The resulting
    :class:`~nfrecon.neural_fields.nfe.NeuralFieldExpansion` combines these modes
    via a tensor-product expansion with learnable coefficients.

    Parameters
    ----------
    dim_out : int
        Output dimension of the expansion.
    num_modes : tuple of int
        Number of modes per coordinate direction. The length of this tuple
        defines the input dimensionality of the expansion.
    dim_latent : int, optional
        Width of the hidden representation in each SIREN network.
    num_hidden_layers : int, optional
        Number of hidden layers in each SIREN network.
    init_omega : float or sequence of float, optional
        Frequency embedding parameter for the first SIREN layer. If a single
        float is provided, the same value is used for all coordinate
        directions. If a sequence is provided, its length must match
        ``len(num_modes)``.
    hidden_omega : float or sequence of float, optional
        Frequency embedding parameter for hidden SIREN layers. Follows the
        same broadcasting rules as ``init_omega``.
    complex_valued : bool, optional
        If ``True``, construct complex-valued SIREN mode networks.

    Returns
    -------
    NeuralFieldExpansion
        A neural field expansion using SIREN networks as univariate modes.
    """
    dim_in = len(num_modes)

    if isinstance(init_omega, float) and isinstance(hidden_omega, float):
        init_omegas = [init_omega] * dim_in
        hidden_omegas = [hidden_omega] * dim_in
    elif (
        isinstance(init_omega, Sequence)
        and len(init_omega) == dim_in
        and isinstance(hidden_omega, Sequence)
        and len(hidden_omega) == dim_in
    ):
        init_omegas = list(init_omega)
        hidden_omegas = list(hidden_omega)
    else:
        raise ValueError(
            "init_omega and hidden_omega must be floats or sequences of length len(num_modes)."
        )

    modes_per_dim = nn.ModuleList(
        [
            Siren(
                dim_in=1,
                dim_out=n,
                dim_latent=dim_latent,
                num_hidden_layers=num_hidden_layers,
                init_omega=init_w,
                hidden_omega=hidden_w,
                complex_valued=complex_valued,
            )
            for init_w, hidden_w, n in zip(init_omegas, hidden_omegas, num_modes)
        ]
    )

    return NeuralFieldExpansion(modes_per_dim, dim_out)
