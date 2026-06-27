"""Base classes for loss functions.

This module defines an abstract base class that provides common
functionality for implementing composite loss functions with multiple
weighted components, as well as utilities for logging and aggregation.
"""

import torch
import abc

from torch import Tensor
from collections.abc import KeysView
from typing import Mapping


class BaseLoss(abc.ABC):
    """Abstract base class for composite loss functions.

    This class provides a lightweight template for losses composed of
    multiple weighted components. Subclasses are expected to implement
    the actual computation of individual loss terms.

    Parameters
    ----------
    weights : Mapping[str, float]
        Mapping from loss component names to their corresponding weights.
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        self._weights = dict(weights)
        self._names = self._weights.keys()

    @property
    def names(self) -> KeysView[str]:
        """Return the names of the loss components.

        Returns
        -------
        KeysView[str]
            Iterable view containing the names of the loss components.
        """
        return self._names

    @property
    def weights(self) -> Mapping[str, float]:
        """Return weights associated to each loss component.

        Returns
        -------
        Mapping[str, float]
            dictionary containing loss weights
        """
        return self._weights

    def _zero_init_loss(self) -> dict[str, float]:
        """Initialize a dictionary of zero-valued loss components.

        Returns
        -------
        dict[str, Tensor]
            Dictionary mapping each loss component name to a zero-valued
            PyTorch scalar tensor.
        """
        loss: dict[str, float] = {}
        for key in self._weights:
            loss[key] = 0.0
        return loss

    def _compute_full_loss(
        self, loss: Mapping[str, Tensor], device: str | torch.device
    ) -> Tensor:
        """Compute the weighted sum of all loss components.

        Parameters
        ----------
        loss : Mapping[str, Tensor]
            Mapping from loss component names to their scalar values.
        device : str or torch.device
            Device on which to place the resulting tensor.

        Returns
        -------
        Tensor
            Scalar tensor representing the total weighted loss.
        """
        complete_loss = 0.0
        for loss_key, loss_weight in self._weights.items():
            complete_loss = complete_loss + loss_weight * loss[loss_key]
        return complete_loss

    @abc.abstractmethod
    def __call__(self, *args) -> dict[str, Tensor]:
        """Evaluate individual loss components.

        Parameters
        ----------
        *args
            Arguments required to compute the loss components.

        Returns
        -------
        dict[str, Tensor]
            Dictionary mapping loss component names to scalar loss values.
        """
        raise NotImplementedError
