"""Continuous Fourier transform utilities.

This module provides thin wrappers around PyTorch's FFT routines to compute
continuous Fourier transforms and their inverses along spatial dimensions.

Note
----
Strictly speaking, a continuous Fourier transform should include integration
weights (e.g. mesh-size factors). These are omitted here, and the terminology
is used in a pragmatic sense.
"""

import torch.fft as fft

from torch import Tensor


def compute_spatial_indices(signal: Tensor, spatial_dim: int) -> list[int]:
    """Return indices of spatial dimensions for Fourier transforms.

    Parameters
    ----------
    signal : Tensor
        Complex-valued tensor of shape ``(*batch_dims, *spatial_dims)``.
    spatial_dim : int
        Number of spatial dimensions (taken from the trailing dimensions).

    Returns
    -------
    list[int]
        Indices corresponding to the spatial dimensions.
    """
    num_dims = len(signal.shape)
    if spatial_dim > num_dims:
        raise ValueError("spatial_dim cannot exceed the number of tensor dimensions")

    num_batch_dims = num_dims - spatial_dim
    return list(range(num_batch_dims, num_dims))


def cft(signal: Tensor, spatial_dim: int) -> Tensor:
    """Compute a (continuous) Fourier transform along spatial dimensions.

    The input is assumed to be centered in the spatial domain (origin at the
    midpoint of each spatial axis).

    Parameters
    ----------
    signal : Tensor
        Complex-valued tensor of shape ``(*batch_dims, *spatial_dims)``.
    spatial_dim : int
        Number of spatial dimensions.

    Returns
    -------
    Tensor
        Complex-valued Fourier transform of ``signal`` with the same shape.
    """
    fft_dims = compute_spatial_indices(signal, spatial_dim)
    signal_shift = fft.ifftshift(signal, dim=fft_dims)
    return fft.fftshift(fft.fftn(signal_shift, dim=fft_dims), dim=fft_dims)


def icft(signal: Tensor, spatial_dim: int) -> Tensor:
    """Compute the inverse (continuous) Fourier transform along spatial dimensions.

    The input is assumed to be centered in the frequency domain (origin at the
    midpoint of each spatial axis).

    Parameters
    ----------
    signal : Tensor
        Complex-valued tensor of shape ``(*batch_dims, *spatial_dims)``.
    spatial_dim : int
        Number of spatial dimensions.

    Returns
    -------
    Tensor
        Complex-valued inverse Fourier transform of ``signal`` with the same shape.
    """
    fft_dims = compute_spatial_indices(signal, spatial_dim)
    signal_shift = fft.ifftshift(signal, dim=fft_dims)
    return fft.fftshift(fft.ifftn(signal_shift, dim=fft_dims), dim=fft_dims)
