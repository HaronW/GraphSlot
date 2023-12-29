"""Convolutional module library."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import math

from graphslot.lib.utils import init_fn

Shape = Tuple[int]

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]

class CNN(nn.Module):
    """Flexible CNN model with conv. and normalization layers."""

    def __init__(self,
                 features: Sequence[int], # FIXME: [in_channels, *out_channels]
                 kernel_size: Sequence[Tuple[int, int]],
                 strides: Sequence[Tuple[int, int]],
                 layer_transpose: Sequence[bool],
                 transpose_double: bool = True,
                 padding: Union[Sequence[Tuple[int, int]], str] = None,
                 activation_fn: Callable[[Array], Array] = nn.ReLU,
                 norm_type: Optional[str] = None,
                 axis_name: Optional[str] = None, # Over which axis to aggregate batch stats.
                 output_size: Optional[int] = None,
                 weight_init = None
                ):
        super().__init__()

        self.features = features
        self.kernel_size = kernel_size
        self.strides = strides
        self.layer_transpose = layer_transpose
        self.transpose_double = transpose_double
        self.padding = padding
        self.activation_fn = activation_fn
        self.norm_type = norm_type
        self.axis_name = axis_name
        self.output_size = output_size
        self.weight_init = weight_init

        # submodules
        num_layers = len(features) - 1 # account for input features (channels)

        if padding is None:
            padding = 0
        if isinstance(padding, int) or isinstance(padding, str):
            padding = [padding for _ in range(num_layers)]
        self.padding = padding

        assert num_layers >= 1, "Need to have at least one layer."
        assert len(kernel_size) == num_layers, (
            f"len(kernel_size): {len(kernel_size)} and len(features): {len(features)} must match.")
        assert len(strides) == num_layers, (
            f"len(strides): {len(strides)} and len(features): {len(features)} must match.")
        assert len(layer_transpose) == num_layers, (
            f"len(layer_transpose): {len(layer_transpose)} and len(features): {len(features)} must match.")

        if self.norm_type:
            assert self.norm_type in {"batch", "group", "instance", "layer"}, (
                f"({self.norm_type}) is not a valid normalization type")

        # Whether transpose conv or regular conv
        conv_module = {False: nn.Conv2d, True: nn.ConvTranspose2d}

        if self.norm_type == "batch":
            norm_module = functools.partial(nn.BatchNorm2d, momentum=0.9)
        elif self.norm_type == "group":
            norm_module = lambda x: nn.GroupNorm(num_groups=32, num_channels=x)
        elif self.norm_type == "layer":
            norm_module = functools.partial(nn.LayerNorm, eps=1e-6)
        elif self.norm_type == "instance":
            norm_module = functools.partial(nn.InstanceNorm2d)

        # model
        self.cnn_layers = nn.ModuleList()
        for i in range(num_layers):

            convname = "convtranspose" if layer_transpose[i] else "conv"
            pad = padding[i]
            if "convtranspose" == convname and isinstance(pad, str):
                pad = 0
            name = f"{convname}_{i}"
            module = conv_module[self.layer_transpose[i]](
                    in_channels=features[i], out_channels=features[i+1],
                    kernel_size=kernel_size[i], stride=strides[i], padding=pad,
                    bias=False if norm_type else True)
            self.cnn_layers.add_module(name, module)

            init_fn[weight_init['conv_w']](module.weight)
            if not norm_type:
                init_fn[weight_init['conv_b']](module.bias)

            if self.norm_type:
                self.cnn_layers.add_module(
                    f"{self.norm_type}_norm_{i}",
                    norm_module(features[i+1]))

            self.cnn_layers.add_module(
                f"activ_{i}",
                activation_fn())

        ## Final Dense Layer
        if self.output_size:
            self.project_to_output = nn.Linear(features[-1], self.output_size, bias=True)
            # nn.init.xavier_uniform_(self.project_to_output.weight)
            init_fn[weight_init['linear_w']](self.project_to_output.weight)
            init_fn[weight_init['linear_b']](self.project_to_output.bias)

    def forward(self, inputs: Array, channels_last=False) -> Tuple[Dict[str, Array]]:
        if channels_last:
            inputs = inputs.permute((0, 3, 1, 2))

        x = inputs
        for name, layer in self.cnn_layers.named_children():
            layer_fn = lambda x_in: layer(x_in)
            if "convtranspose" in name and self.transpose_double:
                output_shape = (x.shape[-2]*2, x.shape[-1]*2)
                layer_fn = lambda x_in: layer(x_in, output_size=output_shape)
            x = layer_fn(x)

        if channels_last:
            # x.shape = (batch_size, n_features, h*, w*)
            x = x.permute((0, 2, 3, 1))
            # x.shape = (batch_size, h*, w*, n_features)
        
        if self.output_size:
            x = self.project_to_output(x)

        return x

class CNN2(nn.Module):
    """New CNN module because above wasn't too flexible in torch."""

    def __init__(self,
                 conv_modules: nn.ModuleList,
                 transpose_modules = None,
                 activation_fn: nn.Module = nn.ReLU,
                 norm_type: Optional[str] = None,
                 output_size: Optional[str] = None,
                 weight_init = None
                ):
        super().__init__()

        self.transpose_modules = transpose_modules
        self.activation = activation_fn
        self.norm_type = norm_type
        self.output_size = output_size
        self.weight_init = weight_init
        self.features = [c.out_channels for c in conv_modules.children()]

        # submodules
        num_layers = len(conv_modules)

        # check if there are transposed convolutions (needed for weight init)
        if transpose_modules is not None:
            assert len(transpose_modules) == num_layers, (
                "need to specify which modules are transposed convolutions.")
        else:
            transpose_modules = [False for _ in range(num_layers)]
            self.transpose_modules = transpose_modules
        init_map = {True: 'convtranspose_w', False: 'conv_w'}

        if self.norm_type:
            assert self.norm_type in {"batch", "group", "instance", "layer"}, (
                f"({self.norm_type}) is not a valid normalization type")

        if self.norm_type == "batch":
            norm_module = functools.partial(nn.BatchNorm2d, momentum=0.9)
        elif self.norm_type == "group":
            norm_module = lambda x: nn.GroupNorm(num_groups=32, num_channels=x)
        elif self.norm_type == "layer":
            norm_module = functools.partial(nn.LayerNorm, eps=1e-6)
        elif self.norm_type == "instance":
            norm_module = functools.partial(nn.InstanceNorm2d)

        # model
        self.cnn_layers = nn.ModuleList()
        for i in range(num_layers):
            name = f"conv_{i}"
            conv = conv_modules[i]
            init_fn[weight_init[init_map[transpose_modules[i]]]](conv.weight)
            if conv.bias is not None:
                init_fn[weight_init['conv_b']](conv.bias)
            self.cnn_layers.add_module(name, conv)

            if self.norm_type:
                self.cnn_layers.add_module(
                    f"{self.norm_type}_norm_{i}",
                    norm_module(self.features[i]))

            self.cnn_layers.add_module(
                f"act_{i}",
                activation_fn())

        if self.output_size:
            self.project_to_output = nn.Linear(self.features[-1], self.outptu_size, bias=True)
            init_fn[weight_init['linear_w']](self.project_to_output.weight)
            init_fn[weight_init['linear_b']](self.project_to_output.bias)

    def forward(self, inputs: Array, channels_last=False) -> Tuple[Dict[str, Array]]:
        if channels_last:
            # inputs.shape = (batch_size, height, width, n_channels)
            inputs = inputs.permute((0, 3, 1, 2))
            # inputs.shape = (batch_size, n_channels, height, width)

        x = inputs
        for name, layer in self.cnn_layers.named_children():
            x = layer(x)

        if channels_last:
            # x.shape = (batch_size, n_features, h*, w*)
            x = x.permute((0, 2, 3, 1))
            # x.shape = (batch_size, h*, w*, n_features)

        if self.output_size:
            x = self.project_to_output(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def weight(self):
        return self.conv1.weight

    @out_channels.setter
    def out_channels(self, value):
        self._out_channels = value

    @property
    def bias(self):
        return self.conv1.bias

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        