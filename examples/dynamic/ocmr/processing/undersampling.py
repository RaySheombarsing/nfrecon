"""Generate GRO undersampling masks for OCMR-style cardiac MRI.

This module implements the deterministic GRO (Golden Ratio Offset) undersampling
pattern used in the OCMR dataset. The implementation is a faithful translation
of the original MATLAB code provided by the dataset authors:

    https://github.com/MihirJoe/cmr-sampling/blob/main/functions/GRO/gro_fun.m

The code here is intentionally *example-specific* and tailored to this
particular sampling scheme. It is not intended to provide a general framework
for designing or experimenting with arbitrary k-space undersampling patterns.

The primary entry point, ``gro_fn``, returns a boolean k-space sampling mask
with shape ``(num_frames, num_readout_lines, num_freq_encoding)``.
"""

import numpy as np


def gro_fn(
    accel_factor: int,
    num_frames: int,
    num_readout_lines: int,
    num_freq_encoding: int,
    tau: int = 1,
    accel_rate_center: float = 2.2,
    alpha: float = 3.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """Construct a deterministic GRO undersampling mask.

    This function implements the GRO (Golden Ratio Offset) sampling pattern
    introduced by the OCMR dataset authors. The implementation is a direct
    translation of the reference MATLAB code and preserves its numerical
    behavior and assumptions.

    Parameters
    ----------
    accel_factor : int
        Desired acceleration factor relative to fully sampled phase encoding.
    num_frames : int
        Number of dynamic frames (time points).
    num_readout_lines : int
        Number of phase-encoding (ky) lines in a fully sampled acquisition.
    num_freq_encoding : int
        Number of frequency-encoding (kx) samples per phase-encoding line.
    tau : int, optional
        Integer offset applied in the definition of the golden-ratio factor.
        This parameter is inherited from the reference implementation and
        should typically be left at its default value.
    accel_rate_center : float, optional
        Controls the size of the central fully sampled region.
    alpha : float, optional
        Exponent controlling the non-linear expansion from the central region
        to the full k-space extent, producing a variable-density pattern.
    eps : float, optional
        Small numerical constant used to avoid boundary effects when generating
        initial sampling positions.

    Returns
    -------
    np.ndarray
        Boolean undersampling mask of shape
        ``(num_frames, num_readout_lines, num_freq_encoding)``, where ``True``
        entries indicate sampled k-space locations.

    Notes
    -----
    - The returned mask is *deterministic*: repeated calls with identical
      parameters will produce the same sampling pattern.
    - Only the phase-encoding dimension (ky) is undersampled; the
      frequency-encoding direction (kx) is fully sampled for selected ky lines.
    - The implementation closely mirrors the MATLAB code structure for
      traceability, which results in a somewhat non-idiomatic NumPy style.
    """

    if accel_factor <= 0:
        raise ValueError("accel_factor must be a positive integer")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_readout_lines <= 0 or num_freq_encoding <= 0:
        raise ValueError("k-space dimensions must be positive")

    # Golden-ratio-based offset factor (as defined in the reference code)
    golden_ratio_factor = 2.0 / (np.sqrt(5.0) + 2 * tau - 1)

    # Number of phase-encoding positions to sample per frame
    num_pe_pos = int(np.ceil(num_readout_lines / accel_factor))

    # Size of the central (shrunk) phase-encoding region
    size_shrunk_pe_region = int(num_readout_lines // accel_rate_center)

    # Scaling factor for non-linear expansion to full k-space
    kappa = (num_readout_lines - size_shrunk_pe_region) / (
        2 * (size_shrunk_pe_region / 2) ** alpha
    )

    # Allocate sampling mask
    mask = np.zeros((num_frames, num_readout_lines, num_freq_encoding), dtype=bool)

    # Sample uniformly from grid {0, ..., shrunk_num_lines - 1}
    step_size = size_shrunk_pe_region / num_pe_pos
    init_phase_enc_pos = (
        np.arange(0.5 + eps, size_shrunk_pe_region + 0.5 - eps, step_size) + step_size
    )

    # Expansion from central region to full k-space (even / odd handling
    # follows the reference implementation exactly)
    if num_readout_lines % 2 == 0:

        def expand_fn(ky: np.ndarray) -> np.ndarray:
            ky_expand = (
                ky
                - kappa
                * np.sign(size_shrunk_pe_region / 2 + 0.5 - ky)
                * np.abs(size_shrunk_pe_region / 2 + 0.5 - ky) ** alpha
                + (num_readout_lines - size_shrunk_pe_region) / 2
                + 0.5
            )
            return ky_expand - num_readout_lines * (
                ky_expand >= num_readout_lines + 0.5
            )

    else:

        def expand_fn(ky: np.ndarray) -> np.ndarray:
            ky_expand = (
                ky
                - kappa
                * np.sign(size_shrunk_pe_region / 2 + 0.5 - ky)
                * np.abs(size_shrunk_pe_region / 2 + 0.5 - ky) ** alpha
                + (num_readout_lines - size_shrunk_pe_region) / 2
            )

    # Generate per-frame sampling pattern
    for frame_idx in range(num_frames):
        # Golden angle circular shift across frames
        phase_enc_pos = (
            init_phase_enc_pos + frame_idx * step_size * golden_ratio_factor - 1
        ) % size_shrunk_pe_region + 1

        phase_enc_pos = phase_enc_pos - size_shrunk_pe_region * (
            phase_enc_pos >= size_shrunk_pe_region + 0.5
        )

        # Expand to full k-space and round to nearest integer ky indices
        phase_enc_pos_expand = expand_fn(phase_enc_pos)
        ky = np.sort(np.round(phase_enc_pos_expand).astype(int))

        # Alternate direction between frames (zig-zag ordering)
        if frame_idx % 2 == 0:
            ky = np.flip(ky)

        # MATLAB-style 1-based to Python 0-based indexing
        mask[frame_idx, ky - 1] = True

    return mask
