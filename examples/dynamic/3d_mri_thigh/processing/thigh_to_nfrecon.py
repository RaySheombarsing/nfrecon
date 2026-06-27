"""Convert raw dynamic thigh k-space data to NFRECON-compatible format.

This script is *dataset-specific* and is intended to demonstrate how a public
ISMRMRD-based thigh movement dataset can be converted into a NFRECON-compatible
NumPy representation. It is not part of the NFRECON core library and does not
attempt to generalize across datasets or acquisition protocols.
"""

import logging
import argparse
import ismrmrd
import numpy as np

from pathlib import Path
from typing import Callable
from scipy.linalg import solve_triangular
from ismrmrd.xsd import CreateFromDocument as parse_ismrmd_header

logger = logging.getLogger(__name__)


def init_cli() -> argparse.Namespace:
    """Initialize command line interface.

    Returns
    -------
    argparse.Namespace
        arguments parsed from command line
    """
    parser = argparse.ArgumentParser(
        description="Convert dynamic thigh k-space data to NFRECON-compatible format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filepath", help="Path to ISMRMRD (.mrd) file", type=Path)
    parser.add_argument("out_dir", help="Output directory", type=Path)
    return parser.parse_args()


def extract_matrix_size(encoding) -> tuple[int, int, int]:
    """Extract matrix size and validate k-space centering.

    Parameters
    ----------
    encoding
        ISMRMRD encoding element from the XML header.

    Returns
    -------
    tuple[int, int, int]
        Matrix size as ``(kz, ky, kx)``.

    Raises
    ------
    ValueError
        If an encoding offset is detected.
    """
    kx = encoding.encodedSpace.matrixSize.x
    ky = encoding.encodedSpace.matrixSize.y
    kz = encoding.encodedSpace.matrixSize.z

    kx_lim = encoding.encodingLimits.kspace_encoding_step_0
    ky_lim = encoding.encodingLimits.kspace_encoding_step_1
    kz_lim = encoding.encodingLimits.kspace_encoding_step_2

    if kx // 2 != kx_lim.center:
        raise ValueError("Extent kx and center do not match there is an offset")
    if ky // 2 != ky_lim.center:
        raise ValueError("Extent ky and center do not match there is an offset")
    if kz // 2 != kz_lim.center:
        raise ValueError("Extent kz and center do not match there is an offset")

    logger.info(f"Matrix size (kz, ky, kx) = {(kz, ky, kx)}")
    return kz, ky, kx


def extract_fov(encoding) -> np.ndarray:
    """Extract field-of-view from encoded space.

    Parameters
    ----------
    encoding
        ISMRMRD encoding element from the XML header.

    Returns
    -------
    np.ndarray
        Field of view in meters, ordered as ``(z, y, x)``.
    """
    fov_mm = encoding.encodedSpace.fieldOfView_mm
    fov = np.array([fov_mm.z, fov_mm.y, fov_mm.x], dtype=float) / 1000.0

    logger.info(f"Field of view: {fov}")
    return fov


def compute_prewhitening(
    acq: list[ismrmrd.Acquisition],
) -> Callable[[np.ndarray], np.ndarray]:
    """Compute a prewhitening transform from noise acquisitions.

    Parameters
    ----------
    acq : list[ismrmrd.Acquisition]
        List of ISMRMRD acquisitions, potentially including noise measurements.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function that applies prewhitening to k-space data. If no suitable noise
        data is present, this is the identity map.

    Raises
    ------
    NotImplementedError
        If more than one noise block is present.
    """
    noise_data: list[np.ndarray] = []

    for readout in acq:
        if readout.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            noise_data.append(readout.data)

    if len(noise_data) == 1:
        logger.info("Noise data available; compute prewhitening transform")
        noise_mat = noise_data[0]
        num_samples = noise_mat.shape[1]
        noise_mean = np.mean(noise_mat, axis=1)
        noise_centered = noise_mat - noise_mean[:, None]
        noise_cov = noise_centered @ noise_centered.conj().T / (num_samples - 1)
        cholesky_factor = np.linalg.cholesky(noise_cov)
        return lambda x: solve_triangular(cholesky_factor, x, lower=True)

    if len(noise_data) > 1:
        raise NotImplementedError("Multiple noise blocks are not supported")

    logger.info("No noise data available; skipping prewhitening")
    return lambda x: x


def process_readouts(filepath: Path) -> dict[str, np.ndarray]:
    """Process an ISMRMRD file into NFRECON-compatible sparse k-space data.

    Parameters
    ----------
    filepath : Path
        Path to ISMRMRD (.mrd) file.

    Returns
    -------
    data : dict[str, np.ndarray]
        Dictionary containing k-space samples, coordinates, and metadata in a
        NFRECON-compatible layout.
    """
    dataset = ismrmrd.Dataset(filepath)
    xml_header = parse_ismrmd_header(dataset.read_xml_header())
    encoding = xml_header.encoding[0]

    num_acq = dataset.number_of_acquisitions()
    acq = [dataset.read_acquisition(k) for k in range(num_acq)]
    acq_headers = [a.getHead() for a in acq]

    num_coils = xml_header.acquisitionSystemInformation.receiverChannels
    logger.info(f"Number of coils: {num_coils}")
    kz, ky, kx = extract_matrix_size(encoding)
    fov = extract_fov(encoding)

    data: dict[str, np.ndarray] = {
        "fov": fov,
        "dt": np.array([1.0], dtype=float),
        "gridsize": np.array((kz, ky, kx), dtype=int),
    }

    prewhite = compute_prewhitening(acq)

    logger.info(f"Process readouts")
    kx_coords = np.arange(kx, dtype=int)
    spatial_dim = 3
    t = 0

    for readout, header in zip(acq, acq_headers):
        if readout.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue

        data[f"kspace_readouts_{t}"] = prewhite(readout.data)

        coords = np.zeros((kx, spatial_dim), dtype=int)
        coords[:, 0] = header.idx.kspace_encode_step_2  # kz
        coords[:, 1] = header.idx.kspace_encode_step_1  # ky
        coords[:, 2] = kx_coords  # kx
        data[f"kspace_coords_{t}"] = coords

        t += 1

    dataset.close()
    return data


def main() -> None:
    args = init_cli()
    logging.basicConfig(level=logging.INFO)

    data_dict = process_readouts(args.filepath)

    out_dir = args.out_dir / args.filepath.parent.stem / args.filepath.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "kspace.npz", **data_dict)


if __name__ == "__main__":
    main()
