"""Trainer utilities for parallel imaging with neural field expansions.

This module implements a lightweight training loop for solving the
multi-coil parallel imaging problem. The trainer is responsible for
orchestrating optimization, checkpointing, logging, and evaluation,
while delegating physics and loss computation to dedicated modules.
"""

import torch
import hydra
import logging
import time
import mlflow

from itertools import chain
from datetime import datetime
from omegaconf import DictConfig
from pathlib import Path
from typing import Optional, Iterator, Callable

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nfrecon.forward_models.multicoil import MulticoilModel
from nfrecon.data.dataset import CoilData
from nfrecon.samplers.samplers import GridSampler, DomainSampler
from nfrecon.samplers.init_utils import init_samplers
from nfrecon.samplers.grids import sample_coordinate_patch
from nfrecon.loss.multicoil_loss import (
    MulticoilLoss,
    eval_loss_dynamic,
    eval_mean_loss_dynamic,
)
from nfrecon.utils import io, scalars

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for multi-coil neural field models.

    This class coordinates optimization, evaluation, checkpointing, and
    logging for solving the parallel imaging problem.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.
    out_dir : Path or str
        Output directory for storing results and checkpoints.
    """

    def __init__(self, cfg: DictConfig, out_dir: str | Path) -> None:
        self._cfg = cfg
        self._out_dir = Path(out_dir)

    # ------------------------------------------------------------------
    #                       Initialization helpers
    # ------------------------------------------------------------------

    def _init_dirs(
        self,
        name_run: str,
        date_format: str = "%Y%m%d-%H%M%S",
        append_date: bool = False,
    ) -> None:
        """Initialize output directories for the current training run.

        Parameters
        ----------
        name_run : str
            Name of the training run.
        date_format : str, optional
            Datetime format used when appending a timestamp.
        append_date : bool, optional
            If ``True``, append the current date/time to ``name_run``.
        """
        if append_date:
            name_run += datetime.now().strftime(date_format)

        self._curr_run_dir = self._out_dir / name_run
        self._curr_run_dir.mkdir(exist_ok=True)

        self._checkpoint_dir = self._curr_run_dir / "checkpoints"
        self._tensorboard_dir = self._curr_run_dir / "tensorboard"

        for dir in [self._checkpoint_dir, self._tensorboard_dir]:
            dir.mkdir(exist_ok=True)

    def _load_checkpoint(
        self,
        model: MulticoilModel,
        base_model_dir: Path | None,
        device: torch.device | str,
    ) -> int:
        """Load a checkpoint and initialize model state.

        Parameters
        ----------
        model : MulticoilModel
            Multicoil forward model to be initialized.
        base_model_dir : Path or None
            Path to a checkpoint directory or file. If ``None``, no checkpoint
            is loaded.
        device : str or torch.device
            Device on which the model is initialized.

        Returns
        -------
        int
            Epoch index from which training should resume.
        """
        if base_model_dir is None:
            return 0
        checkpoint = io.load_multicoil_model(base_model_dir, device, model)
        return checkpoint["epoch"] + 1

    @staticmethod
    def _init_optimizer(
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig | None,
        params: Iterator[torch.nn.Parameter],
    ) -> tuple[Optimizer, Optional[LRScheduler]]:
        """Initialize optimizer and optional learning-rate scheduler.

        Parameters
        ----------
        cfg_optimizer : DictConfig
            Configuration for the optimizer.
        cfg_scheduler : DictConfig or None
            Configuration for the scheduler. If ``None``, no scheduler is used.
        params : Iterator[torch.nn.Parameter]
            Iterable of parameters to optimize.

        Returns
        -------
        optimizer : Optimizer
            Instantiated PyTorch optimizer.
        scheduler : LRScheduler or None
            Instantiated learning-rate scheduler, if configured.
        """
        optimizer = hydra.utils.instantiate(cfg_optimizer, params=params)

        if cfg_scheduler is not None:
            scheduler = hydra.utils.instantiate(cfg_scheduler, optimizer=optimizer)
        else:
            scheduler = None

        return optimizer, scheduler

    # ------------------------------------------------------------------
    #                   Logging and checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: MulticoilModel,
        dataset: CoilData,
        epoch: int,
        filename: str = "model_checkpoint",
        ext: str = "pth",
    ) -> None:
        """Save a model checkpoint.

        Parameters
        ----------
        model : MulticoilModel
            (Currently) Trained multicoil model.
        dataset : CoilData
            Dataset providing spatial and (optionally) temporal grids.
        epoch : int
            Current training epoch.
        filename : str, optional
            Base filename for the checkpoint.
        ext : str, optional
            File extension used for the checkpoint.
        """
        logger.info(f"Epoch {epoch} - Store model")
        path_out = self._checkpoint_dir / f"{filename}_{epoch}.{ext}"

        if path_out.exists():
            logger.warning(
                f"{filename}_{epoch}.{ext} already exists and will be overwritten!"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_cfg": self._cfg.model,
                "model_state_dict": model.state_dict(),
                "num_coils": model.num_coils,
                "spatial_domain": dataset.spatial_domain,
                "dynamic": model.dynamic,
                "time_domain": dataset.time_domain,
                "time_scale": dataset.time_scale,
                "time_grid": dataset.time_partition,
            },
            path_out,
        )

    def _log_metrics(
        self,
        writer: SummaryWriter,
        metrics: dict[str, float],
        epoch: int,
        tag: str,
    ) -> None:
        """Log metrics to TensorBoard and MLflow.

        Parameters
        ----------
        writer : SummaryWriter
            TensorBoard writer instance.
        metrics : dict[str, float]
            Dictionary of metrics to be logged.
        epoch : int
            Current training epoch.
        tag : str
            Tag under which metrics are grouped.
        """
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, step=epoch)

        for name, val in metrics.items():
            writer.add_scalar(f"{tag}/{name}", val, epoch)

    @staticmethod
    def _print_loss(epoch: int, loss_eval: float | Tensor, elapsed_time: float) -> None:
        """Print loss at current epoch to screen.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        loss_eval : float | Tensor
            Loss value to print
        elapsed_time : float
            Time it took to take gradient descent step.
        """
        logger.info(f"Epoch {epoch} loss = {loss_eval}")
        logger.info(f"Elapsed time = {elapsed_time:.3f}s")

    # ------------------------------------------------------------------
    #                       Core training logic
    # ------------------------------------------------------------------

    def _train_step(
        self,
        model: MulticoilModel,
        dataset: CoilData,
        grid_sampler: GridSampler,
        domain_sampler: DomainSampler,
        coil_sampler: Callable[[], list[int]],
        optimizer: Optimizer,
        loss: MulticoilLoss,
        loss_evaluator: Callable,
        device: torch.device | str,
    ) -> dict[str, Tensor]:
        """Execute a single training (gradient descent) step.

        Parameters
        ----------
        model : MulticoilModel
            Multicoil forward model.
        dataset : CoilData
            Dataset providing k-space observations.
        grid_sampler : GridSampler
            Sampler for discrete observation grids.
        domain_sampler : DomainSampler
            Sampler for continuous domain grids.
        coil_sampler : Callable[[], list[int]]
            Sampler returning indices of coils to include.
        optimizer : Optimizer
            Optimizer used for parameter updates.
        loss : MulticoilLoss
            Loss function used for optimization.
        loss_evaluator : Callable
            Function evaluating the loss for a given batch.
        device : str or torch.device
            Device on which computations are performed.

        Returns
        -------
        dict[str, Tensor]
            Dictionary of loss component values for the epoch.
        """
        model.train()
        optimizer.zero_grad()

        coil_indices = coil_sampler()
        coord_patch = sample_coordinate_patch(grid_sampler, domain_sampler, device)

        loss_components = loss_evaluator(
            model, loss, dataset, coord_patch, coil_indices, device
        )
        loss_components["complete_loss"].backward()
        optimizer.step()

        return loss_components

    def train(
        self,
        model: MulticoilModel,
        dataset: CoilData,
        loss: MulticoilLoss,
        base_model_dir: Path | None = None,
        writer: SummaryWriter | None = None,
        name_run: str = "",
        device: torch.device | str = "cpu",
    ) -> dict[str, float]:
        """Run training loop.

        Parameters
        ----------
        model : MulticoilModel
            Multicoil forward model.
        dataset : CoilData
            Dataset containing k-space observations.
        loss : MulticoilLoss
            Loss function used for training.
        base_model_dir : Path or None, optional
            Path to a checkpoint used for initialization.
        writer : SummaryWriter or None, optional
            TensorBoard writer instance. If ``None``, a writer is created.
        name_run : str, optional
            Name assigned to the training run.
        device : str or torch.device, optional
            Device on which computations are performed.

        Returns
        -------
        mean_loss: dict[str, float]
            Returns final mean metrics using the final model after training.
        """
        if dataset.dynamic:
            loss_evaluator = eval_loss_dynamic
            mean_loss_evaluator = eval_mean_loss_dynamic
        else:
            raise NotImplementedError(
                "Trainer currently only supports dynamic datasets"
            )

        self._init_dirs(name_run)

        if writer is None:
            writer = SummaryWriter(self._tensorboard_dir)

        init_epoch = self._load_checkpoint(model, base_model_dir, device)

        coil_sampler, grid_sampler, domain_sampler = init_samplers(
            self._cfg.samplers, dataset
        )

        optimizer, scheduler = self._init_optimizer(
            self._cfg.optimizer.method,
            self._cfg.optimizer.scheduler,
            chain(
                model.magnetization.parameters(), model.coil_sensitivity.parameters()
            ),
        )

        logger.info("Train NFE ...")

        for epoch in range(init_epoch, self._cfg.optimizer.epochs):
            start_time = time.time()
            loss_components = self._train_step(
                model,
                dataset,
                grid_sampler,
                domain_sampler,
                coil_sampler,
                optimizer,
                loss,
                loss_evaluator,
                device,
            )
            elapsed_time = time.time() - start_time

            if scheduler is not None:
                scheduler.step(loss_components["complete_loss"].item())

            self._print_loss(
                epoch, loss_components["complete_loss"].item(), elapsed_time
            )

            with torch.no_grad():
                if (
                    epoch % self._cfg.optimizer.log_freq == 0
                    or epoch == self._cfg.optimizer.epochs - 1
                ):
                    logger.info("Evaluate mean metrics over full spacetime")
                    model.eval()

                    start_time = time.time()
                    mean_loss = mean_loss_evaluator(
                        model,
                        loss,
                        dataset,
                        coil_sampler,
                        device=device,
                    )
                    elapsed_time = time.time() - start_time

                    self._print_loss(
                        epoch, mean_loss["complete_loss"].item(), elapsed_time
                    )
                    self._log_metrics(
                        writer, scalars.to_scalar_dict(mean_loss), epoch, "metrics"
                    )

            if (
                self._cfg.optimizer.save_epoch
                and epoch > 0
                and epoch % self._cfg.optimizer.save_freq == 0
            ):
                self._save_checkpoint(model, dataset, epoch)

        self._save_checkpoint(model, dataset, epoch)
        writer.close()

        return scalars.to_scalar_dict(mean_loss)
