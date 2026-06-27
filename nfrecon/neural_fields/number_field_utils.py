"""Common utilities for neural field constructions."""

import torch

from typing import Callable
from torch import Tensor


def format_field(
    complex_valued: bool, dim_out: int
) -> tuple[int, Callable[[Tensor], Tensor]]:
    """Create an output formatter for real or complex-valued network outputs.

    Parameters
    ----------
    complex_valued : bool
        If ``True``, interpret the output as complex-valued with real and
        imaginary parts stored in the last dimension.
    dim_out : int
        Output dimension of network w.r.t. chosen number field

    Returns
    -------
    dim_out_real : int
        real dimension output
    to_number_field: Callable[[Tensor], Tensor]
        Callable converting a real-valued output to the desired number field.
    """
    if complex_valued:
        dim_out_real = 2 * dim_out
        to_number_field = lambda x: torch.view_as_complex(
            x.reshape(*x.shape[:-1], -1, 2)
        )
    else:
        dim_out_real = dim_out
        to_number_field = lambda x: x

    return dim_out_real, to_number_field
