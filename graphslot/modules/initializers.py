"""Initializers module library."""

# FIXME

import functools
from turtle import forward
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
import numpy as np

from graphslot.lib import utils
from graphslot.modules import misc
from graphslot.modules import video

Shape = Tuple[int]

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]

class ConditionEncoderStateInit(nn.Module):
    """State init that encodes bounding box corrdinates as conditional input.

    Attributes:
        embedding_transform: A nn.Module that is applied on inputs (bounding boxes).
        prepend_background: Boolean flag' whether to prepend a special, zero-valued
            background bounding box to the input. Default: False.
        center_of_mass: Boolean flag; whether to convert bounding boxes to center
            of mass coordinates. Default: False.
        background_value: Default value to fill in the background.
    """

    def __init__(self,
                embedding_transform: nn.Module,
                prepend_background: bool = False,
                background_value: float = 0.
                ):
        super().__init__()

        self.embedding_transform = embedding_transform
        self.prepend_background = prepend_background
        self.background_value = background_value
    
    def forward(self, inputs: Array, batch_size: Optional[int]) -> Array:
        # del batch_size # Unused.

        # inputs.shape = (batch_size, seq_len, bboxes, 4)
        inputs = inputs[:, 0] # Only condition on first time step.
        
        # inputs.shape = (batch_size, bboxes, 4)

        if self.prepend_background:
            # Adds a fake background box [0, 0, 0, 0] at the beginning.
            # batch_size = inputs.shape[0]

            # Encode the background as specified by the background_value.
            device = inputs.device
            background = normal.Normal(loc=0, scale=0.1).sample((batch_size, 1, 4)).to(device)

            inputs = torch.cat([background, inputs], dim=1)

        slots = self.embedding_transform(inputs)

        return slots
    
class GaussianStateInit(nn.Module):
    """Random state initialization with zero-mean, unit-variance Gaussian

    Note: This module does not contain any trainable parameters.
        This module also ignores any conditional input (by design).
    """

    def __init__(self,
                 shape: Sequence[int],
                ):
        super().__init__()

        self.shape = shape
    
    def forward(self, inputs: Optional[Array], batch_size: int) -> Array:
        del inputs # Unused.
        return torch.normal(mean=torch.zeros([batch_size] + list(self.shape)))