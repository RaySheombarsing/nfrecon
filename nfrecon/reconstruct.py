"""CLI entry point for training neural field expansions for MRI reconstruction.

This module provides a thin command-line interface around the NFRECON library
for training neural field expansions (NFEs) to solve the multi-coil parallel
imaging problem. All reusable logic lives inside NFRECON; this file is
responsible only for orchestration, configuration, and optional experiment
tracking.
"""

import hydra
import logging
import torch
import mlflow
import numpy as np

from omegaconf import DictConfig
from pathlib import Path

from nfrecon.forward_models.multicoil import MulticoilModel
from nfrecon.optimizer.trainer import Trainer
from nfrecon.utils import hydra_tools

# Set up global module logger
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train neural field expansions for parallel MRI reconstruction.

    This function instantiates datasets, models, losses, and the training
    loop based on the provided Hydra configuration. Side effects such as
    logging, seeding, and experiment tracking are explicitly controlled by
    configuration options.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.setup.out_dir)
    out_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    dataset = hydra.utils.instantiate(cfg.data)
    model = MulticoilModel(cfg.model, dataset.num_coils, dataset.dynamic).to(
        cfg.setup.device
    )
    loss = hydra.utils.instantiate(cfg.loss)
    trainer = Trainer(cfg, out_dir)

    if cfg.setup.use_mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{out_dir}/mlflow.db")
        with mlflow.start_run():
            mlflow.log_params(hydra_tools.flatten_dict(cfg))
            mean_loss = trainer.train(
                model,
                dataset,
                loss,
                base_model_dir=cfg.setup.base_model_dir,
                device=cfg.setup.device,
            )
    else:
        mean_loss = trainer.train(
            model,
            dataset,
            loss,
            base_model_dir=cfg.setup.base_model_dir,
            device=cfg.setup.device,
        )

    logger.info("Final performance")
    for name, val in mean_loss.items():
        logger.info(f"{name} - loss = {val:.3f}")


if __name__ == "__main__":
    main()
