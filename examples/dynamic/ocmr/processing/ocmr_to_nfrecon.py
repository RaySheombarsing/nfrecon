"""Convert OCMR raw fully sampled k-space data to NFRECON-compatible NumPy format.

This script is *example-specific* and demonstrates how fully sampled cardiac
OCMR raw k-space data can be prepared for use with the ``nfrecon`` reconstruction
framework. It assumes a fixed OCMR raw data layout and makes explicit,
documented assumptions about acquisition dimensionality and axis ordering.

The script performs the following steps:

1. Load a single OCMR ``.h5`` raw data file
2. Validate that the acquisition is 2D with singleton auxiliary dimensions
3. Reorder raw k-space axes to match NFRECON conventions
4. Generate fully-sampled and undersampled per-slice k-space data
5. Save each slice in a nfrecon-compatible ``.npz`` format

This code is **not** intended to be a general MRI conversion tool.
"""

import argparse
import logging
import re
import numpy as np
import read_ocmr as read

from enum import Enum
from pathlib import Path

from nfrecon.utils.kspace import flatten_kspace
from undersampling import gro_fn

logger = logging.getLogger(__name__)


def init_cli() -> argparse.Namespace:
    """Initialize and parse command-line arguments for OCMR conversion.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
        - filepath : Path
            Path to the input OCMR ``.h5`` file.
        - out_dir : Path
            Directory in which converted data will be stored.
        - acceleration: list[int]
            Acceleration factors.
        - percentile : float
            Percentile used to determine k-space scale factors.
        - overwrite : bool
            Whether to overwrite existing output directories.
        - log_level : str
            Logging verbosity level.
    """
    parser = argparse.ArgumentParser(
        description="Convert OCMR raw k-space data to NFRECON-compatible format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filepath", type=Path, help="Path to OCMR .h5 file")
    parser.add_argument(
        "out_dir", type=Path, help="Directory in which converted data will be stored."
    )
    parser.add_argument(
        "--acceleration",
        nargs="+",
        type=int,
        help="Acceleration factors.",
        default=[8, 12, 16],
    )
    parser.add_argument(
        "--percentile",
        type=float,
        help="Percentile used to determine k-space scale factor",
        default=99.5,
    )
    parser.add_argument(
        "--overwrite",
        help="Whether to overwrite existing output directories",
        action="store_true",
    )
    parser.add_argument(
        "--log-level", type=str, help="Logging verbosity level", default="INFO"
    )

    return parser.parse_args()


class RawAxis(Enum):
    """Enumeration of OCMR raw k-space axes.

    The ordering corresponds to the layout returned by ``read_ocmr``.
    """

    KX = 0
    KY = 1
    KZ = 2
    COIL = 3
    PHASE = 4
    SET = 5
    SLICE = 6
    REP = 7
    AVG = 8


def validate_and_squeeze_dims(kspace: np.ndarray) -> np.ndarray:
    """Validate expected singleton dimensions and remove them.

    Parameters
    ----------
    kspace : np.ndarray
        Raw k-space array loaded from an OCMR file.

    Returns
    -------
    np.ndarray
        k-space array with singleton dimensions removed.

    Raises
    ------
    RuntimeError
        If any expected singleton dimension has size greater than one.
    """
    for axis in (RawAxis.KZ, RawAxis.SET, RawAxis.REP, RawAxis.AVG):
        if kspace.shape[axis.value] != 1:
            raise RuntimeError(
                f"Expected dimension '{axis.name}' to be 1, got {kspace.shape[axis.value]}"
            )

    return np.squeeze(
        kspace,
        axis=(
            RawAxis.KZ.value,
            RawAxis.SET.value,
            RawAxis.REP.value,
            RawAxis.AVG.value,
        ),
    )


def reorder_axes_to_nfrecon(kspace: np.ndarray) -> np.ndarray:
    """Reorder OCMR raw axes to NFRECON k-space convention.

    The following transformation is applied:

    - OCMR: ``[kx, ky, coil, phase, slice]``
    - NFRECON: ``[coil, phase, slice, ky, kx]``

    Parameters
    ----------
    kspace : np.ndarray
        k-space array after singleton dimensions have been removed.

    Returns
    -------
    np.ndarray
        Reordered k-space array compatible with nfrecon library.
    """
    return np.transpose(kspace, [2, 3, 4, 1, 0])


def extract_temporal_resolution(metadata: dict) -> np.ndarray:
    """Extract temporal resolution from OCMR metadata.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary returned by ``read_ocmr``.

    Returns
    -------
    np.ndarray
        Array containing the temporal resolution in seconds.
    """
    dt_str = re.sub(r"[\[\]]", "", metadata["TRes"])
    return np.array([float(dt_str)]) / 1000.0


def main() -> None:
    """Run the OCMR-to-nfrecon conversion pipeline.

    This function orchestrates loading of raw data, preprocessing,
    undersampling, and storage of nfrecon-compatible slice-wise datasets.
    """
    args = init_cli()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    out_dir = args.out_dir / args.filepath.stem
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory '{out_dir}' exists (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading OCMR raw data")
    kspace, metadata = read.read_ocmr(args.filepath)

    logger.info("Validating and reshaping k-space")
    kspace = validate_and_squeeze_dims(kspace)
    kspace = reorder_axes_to_nfrecon(kspace)
    logger.info(f"Converted k-space shape: {kspace.shape}")

    # Extract metadata
    fov = np.array(metadata["FOV"])[::-1] / 1000.0
    dt = extract_temporal_resolution(metadata)
    logger.info(f"Field of view: {fov}")
    logger.info(f"Time between frames: {dt}")
    np.savez(out_dir / "metadata.npz", **metadata)

    num_times, num_slices = kspace.shape[1:3]
    img_shape = kspace.shape[-2:]
    gridsize = np.array(img_shape, dtype=int)

    for slice_idx in range(num_slices):
        logger.info(f"Processing slice {slice_idx}...")
        kspace_slice = kspace[:, :, slice_idx]

        scale_factor = np.percentile(np.abs(kspace_slice), args.percentile)
        slice_data: dict[str, np.ndarray] = {
            "fov": fov[1:],
            "dt": dt,
            "gridsize": gridsize,
            "scale_factor": scale_factor,
        }

        # Fully sampled kspace
        kspace_coords_fs, kspace_nonzero_fs = flatten_kspace(kspace_slice, dynamic=True)

        for t in range(num_times):
            slice_data[f"kspace_full_{t}"] = kspace_nonzero_fs[t]
            slice_data[f"kspace_coords_full_{t}"] = kspace_coords_fs[t]

        # Undersampled kspace
        for acc in args.acceleration:
            logger.info(f"Slice {slice_idx} (acceleration: {acc})...")
            mask = gro_fn(acc, num_times, *img_shape)
            kspace_us = np.einsum("ctyx,tyx->ctyx", kspace_slice, mask)
            kspace_coords_us, kspace_nonzero_us = flatten_kspace(
                kspace_us, dynamic=True
            )

            for t in range(num_times):
                slice_data[f"kspace_gro_acc_{acc}_{t}"] = kspace_nonzero_us[t]
                slice_data[f"kspace_coords_gro_acc_{acc}_{t}"] = kspace_coords_us[t]

        np.savez(out_dir / f"slice_{slice_idx}.npz", **slice_data)


if __name__ == "__main__":
    main()
