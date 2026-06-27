"""Command-line interface for evaluating trained NFRECON models.

This module provides a lightweight CLI utility for evaluating a trained
neural field expansion of a multicoil MRI reconstruction model on a
user-specified spatial (and optional temporal) grid. It is intended for
*post-training usage* such as visualization, qualitative inspection,
and downstream analysis.
"""

import argparse
import torch
import logging
import numpy as np

from pathlib import Path

from nfrecon.utils import io
from nfrecon.samplers import grids
from nfrecon.loss.multicoil_loss import init_time_batches

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
#                           CLI argument handling
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Construct the argument parser for the NFRECON evaluation CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser describing all supported CLI options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MRI magnetization and optionally coil sensitivities "
            "from a trained NFRECON neural field model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the serialized trained NFRECON model (e.g. .pt file).",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Directory in which all outputs will be stored.",
    )
    parser.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Desired resolution of evaluation grid. For static models, provide spatial "
            "dimensions only (e.g. [Nz] Ny Nx). For dynamic models, the first "
            "entry specifies the temporal resolution followed by the spatial ones."
        ),
    )
    parser.add_argument(
        "--bsize_t",
        type=int,
        default=32,
        help="Batch size along the temporal dimension for dynamic models.",
    )
    parser.add_argument(
        "--m_threshold",
        type=float,
        default=2e-5,
        help=(
            "Threshold on magnetization magnitude. Values whose absolute magnitude "
            "falls below this threshold are set to zero in the output."
        ),
    )
    parser.add_argument(
        "--compute_coil_maps",
        action="store_true",
        help="If set, additionally evaluate and store coil sensitivity maps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for model evaluation (e.g. 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional string appended to output filenames.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser


# -----------------------------------------------------------------------------
#                               Main evaluation routine
# -----------------------------------------------------------------------------


def main() -> None:
    """Evaluate magnetization and optionally coil sensitivity maps.

    The routine loads a trained multicoil neural field model, constructs a
    user-specified evaluation grid, and evaluates the learned magnetization
    field. For dynamic models, evaluation is performed in temporal batches
    to control memory usage.

    All outputs are written as NumPy ``.npy`` files to the specified output
    directory.
    """
    args = build_argparser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args.out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {args.out_dir}")

    # Load trained model
    time_scale, time_domain, spatial_domain, model = io.load_multicoil_model(
        args.model_path, args.device
    )

    # ------------------------------------------------------------------
    #       Parse resolution specification and construct spatial grid
    # ------------------------------------------------------------------
    if model.dynamic:
        if len(args.resolution) < 3:
            raise ValueError(
                "Dynamic models require at least one temporal and two spatial resolution values."
            )
        npts_time = args.resolution[0]
        npts_spatial = args.resolution[1:]
        logger.info(
            f"Dynamic model detected. Using temporal scaling factor {time_scale}."
        )
    else:
        if len(args.resolution) < 2:
            raise ValueError(
                "Static models require at least two spatial resolution values."
            )
        npts_spatial = args.resolution
        npts_time = None

    _, spatial_partitions = grids.construct_grid(
        spatial_domain,
        npts_spatial,
        device=args.device,
    )

    # ------------------------------------------------------------------
    #                           Evaluate magnetization
    # ------------------------------------------------------------------
    logger.info("Evaluating magnetization field ...")
    with torch.no_grad():
        if model.dynamic:
            time_partition = torch.linspace(
                time_domain[0], time_domain[1], npts_time, device=args.device
            )
            time_indices_batches, _ = init_time_batches(
                time_partition, len(time_partition), args.bsize_t
            )

            m_eval = torch.empty(
                npts_time, *npts_spatial, dtype=torch.complex64, device=args.device
            )
            for indices in time_indices_batches:
                m_eval[indices] = model.magnetization(
                    time_partition[indices], *spatial_partitions
                )
        else:
            m_eval = model.magnetization(*spatial_partitions)

        # Threshold small magnitudes
        m_eval[torch.abs(m_eval) < args.m_threshold] = 0.0

        m_path = args.out_dir / f"{args.name}_magnetization.npy"
        np.save(m_path, m_eval.cpu().numpy())
        logger.info(f"Saved magnetization to {m_path}")

    # ------------------------------------------------------------------
    #                   Optional coil sensitivity evaluation
    # ------------------------------------------------------------------
    if args.compute_coil_maps:
        logger.info("Evaluating coil sensitivity maps ...")
        with torch.no_grad():
            coil_maps_eval = model.coil_sensitivity(*spatial_partitions).cpu().numpy()
            coil_maps_path = args.out_dir / f"{args.name}_coil_maps.npy"
            np.save(coil_maps_path, coil_maps_eval)
            logger.info(f"Saved coil sensitivity maps to {coil_maps_path}")


if __name__ == "__main__":
    main()
