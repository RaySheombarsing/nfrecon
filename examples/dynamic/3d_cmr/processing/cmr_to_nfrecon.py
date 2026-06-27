"""Convert Ohio 3D cine MRI data to NFRECON-compatible format.

This script converts institute-specific `.mat` k-space data into the NFRECON
input format. It is intentionally dataset-specific and uses heuristic
transformations tailored to this dataset.

The goal is clarity and reproducibility rather than generality.
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from nfrecon.utils.kspace import flatten_kspace

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert Ohio 3D cine MRI data to NFRECON format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filepath", type=Path, help="Path to `.mat` k-space file.")
    parser.add_argument("out_dir", type=Path, help="Output directory.")
    parser.add_argument(
        "--acc",
        type=int,
        default=16,
        help="Target acceleration factor for retrospective undersampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--kspace_threshold",
        type=float,
        default=0.0,
        help="Threshold used to idenfity nonzero kspace measurements.",
    )
    return parser.parse_args()


def load_ohio_data(filepath: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Load k-space data and metadata from Ohio dataset.

    Parameters
    ----------
    filepath : Path
        Path to `.mat` file containing raw MRI data.

    Returns
    -------
    dt : float
        Temporal resolution in seconds.
    fov : np.ndarray, shape (3,)
        Field of view in meters as (z, y, x).
    kspace : np.ndarray, shape (nc, nt, nz, ny, nx)
        Complex k-space data reordered for NFRECON.
    """
    with h5py.File(filepath, "r") as file:
        scan_params = file["D"]["scanParam"]
        kspace_raw = file["D"]["kb"][()]
        kspace_real = kspace_raw["real"].astype(np.float32)
        kspace_imag = kspace_raw["imag"].astype(np.float32)

        # Construct complex array without temporaries to minimize peak memory
        kspace = np.empty(kspace_real.shape, dtype=np.complex64)
        kspace.real[:] = kspace_real
        kspace.imag[:] = kspace_imag

        # Original ordering: (nt, nc, nz, ny, nx)
        nt, nc, nz, ny, nx = kspace.shape

        # Convert metadata
        fov_inplane = scan_params["FOV"][()].flatten() / 1000
        fov_z = nz * scan_params["ResPar"][()].flatten()[0] / 1000
        fov = np.array([fov_z, fov_inplane[1], fov_inplane[0]])
        dt = scan_params["TRes"][()].flatten()[0] * 1e-6  # µs → s

        # Reorder to (nc, nt, nz, ny, nx)
        kspace = np.moveaxis(kspace, (0, 1), (1, 0))

    logger.info(f"Grid size = {(nz, ny ,nx)}")
    logger.info(f"Number of frames: {nt}")
    logger.info(f"Number of coils: {nc}")
    logger.info(f"FOV (m) = {fov}")
    logger.info(f"dt (s) = {dt}")

    return dt, fov, kspace


def undersample_kspace(
    kspace: np.ndarray,
    acc: int = 16,
    seed: int | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply retrospective undersampling to k-space.

    The method removes readouts uniformly at random among non-zero readouts
    to achieve the desired acceleration factor.

    Parameters
    ----------
    kspace : np.ndarray
        Input k-space of shape (nc, nt, nz, ny, nx).
    acc : int, optional
        Target acceleration factor, by default 16.
    seed : int | None, optional
        Random seed for reproducibility.
    eps : float, optional
        Threshold used to identify non-zero measurements.

    Returns
    -------
    np.ndarray
        Undersampled k-space (same shape as input).

    Raises
    ------
    ValueError
        If the requested acceleration is smaller than current acceleration.
    """
    rng = np.random.default_rng(seed)

    nc, nt, nz, ny, nx = kspace.shape
    npts_kspace = nz * ny * nx
    num_readouts = nz * ny

    kspace_flat = kspace.reshape(nc, nt, num_readouts, nx)

    for t in range(nt):
        magnitude = np.abs(kspace_flat[0, t])  # assume coil consistency

        nonzero_kspace_indices = np.where(magnitude > eps)[0]
        num_nonzero_kspace = len(nonzero_kspace_indices)
        current_acc = npts_kspace / num_nonzero_kspace

        if current_acc > acc:
            raise ValueError(
                f"Requested acceleration ({acc}) is smaller than current ({current_acc:.2f})."
            )

        nonzero_readout_indices = np.where(np.sum(magnitude, axis=-1) > eps)[0]
        num_nonzero_readout = len(nonzero_readout_indices)

        avg_npts_readout = 0.0
        for idx in nonzero_readout_indices:
            avg_npts_readout += (
                np.count_nonzero(magnitude[idx] > eps) / num_nonzero_readout
            )

        # Determine how many readouts to remove
        target_nonzero = npts_kspace / acc
        npts_remove = num_nonzero_kspace - target_nonzero
        num_readouts_remove = int(np.ceil(npts_remove / avg_npts_readout))

        remove_indices = rng.choice(
            nonzero_readout_indices, size=num_readouts_remove, replace=False
        )

        kspace_flat[:, t, remove_indices, :] = 0.0

    return kspace_flat.reshape(nc, nt, nz, ny, nx)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    dt, fov, kspace = load_ohio_data(args.filepath)
    nt = kspace.shape[1]

    logger.info(f"Applying undersampling (acc={args.acc})")
    kspace = undersample_kspace(
        kspace, acc=args.acc, eps=args.kspace_threshold, seed=args.seed
    )

    logger.info(f"Store kspace data in NFREFON-compatiable format.")
    coords, values = flatten_kspace(kspace, dynamic=True)

    data = {
        "fov": fov,
        "dt": dt,
        "gridsize": np.array(kspace.shape[2:], dtype=int),
    }

    for t in range(nt):
        data[f"kspace_vals_{t}"] = values[t]
        data[f"kspace_coords_{t}"] = coords[t]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "kspace.npz"

    logger.info(f"Saving to {out_file}")
    np.savez(out_file, **data)


if __name__ == "__main__":
    main()
