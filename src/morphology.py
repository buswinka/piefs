from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def _compute_zero_padding(kernel_size: List[int]) -> List[int]:
    r"""Utility function that computes the padding tuple.
    Adapted from Kornia.
    """
    return [(k - 1) // 2 for k in kernel_size]


@torch.jit.script
def _get_binary_kernel3d(window_size: int, device: str) -> Tensor:
    r"""Creates a symmetric binary kernel to extract the patches. If the window size
    is HxWxD will create a (H*W*D)xHxWxD kernel.

    Adapted from a 2D Kornia implementation.
    """
    window_range: int = int(window_size ** 3)
    kernel: Tensor = torch.zeros((window_range, window_range, window_range), device=device)
    for i in range(window_range):
        kernel[i, i, i] += 1.0
    kernel = kernel.view(-1, 1, window_size, window_size, window_size)

    # get rid of all zero kernels
    ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
    return kernel[ind[:, 0], ...]


@torch.jit.script
def _get_binary_kernel2d(window_size: int, device: str) -> Tensor:
    r"""Creates a symmetric binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.

    Adapted from a 2D Kornia implementation.
    """
    window_range: int = int(window_size ** 2)
    kernel: Tensor = torch.zeros((window_range, window_range), device=device)
    for i in range(window_range):
        kernel[i, i] += 1.0
    kernel = kernel.view(-1, 1, window_size, window_size)

    # get rid of all zero kernels
    ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
    return kernel[ind[:, 0], ...]


def _binary_convolution2d(input: Tensor, padding_mode: str = 'constant') -> Tensor:
    """
    Returns the nearest neighbors along a new axis of a 2D image.

    Shapes:
     - input: :math:`(B, C, X, Y)`
     - returns: :math:`(B, C, 9, X, Y)`

    :param input: Input Tensor
    :param padding_mode: the method by which padding is filled around the image. See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    for more information.Accepted values are: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

    :return: The nearest neighbor of each pixel in a new dimension in a new dimension.
    """
    # Pad the image...
    padding: List[int] = _compute_zero_padding([3, 3])
    input = F.pad(input=input, pad=padding + padding, mode=padding_mode)

    kernel: Tensor = _get_binary_kernel2d(3, str(input.device))
    b, c, h, w = input.shape

    # map the local window to single vector
    features: Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel,
                                padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)

    padding_slice = slice(padding[0], -padding[0], 1)

    return features[..., padding_slice, padding_slice]


def _binary_convolution3d(input: Tensor, padding_mode: str = 'constant') -> Tensor:
    """
    returns the nearest neighbors along a new axis of a 3D image.

    Shapes:
     - input: :math:`(B, C, X, Y, Z)`
     - returns: :math:`(B, C, 27, X, Y, Z)`

    :param input: Input Tensor
    :param padding_mode: the method by which padding is filled around the image. See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    for more information.Accepted values are: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

    :return: The nearest neighbor of each pixel in a new dimension in a new dimension.
    """
    # Pad the image...
    padding: List[int] = _compute_zero_padding([3, 3])
    input = F.pad(input=input, pad=padding + padding + padding, mode=padding_mode)

    kernel: Tensor = _get_binary_kernel3d(3, str(input.device))
    b, c, h, w, d = input.shape

    # map the local window to single vector
    features: Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel,
                                padding=padding, stride=1)
    features = features.view(b, c, -1, h, w, d)

    padding_slice = slice(padding[0], -padding[0], 1)

    return features[..., padding_slice, padding_slice, padding_slice]


def binary_convolution(input: Tensor, padding_mode: str = 'constant') -> Tensor:
    """
    Returns the nearest neighbors of each pixel along a nex axis of a 2D or 3D tensor. A 2D image has
    9 (including self) neighbors. A 3D image has 27.

    Shapes:
     - input: :math:`(B, C, X, Y)` or :math:`(B, C, X, Y, Z)`
     - returns: :math:`(B, C, 9, X, Y)` or :math:`(B, C, 27, X, Y, Z)`

    :param input: input tensor
    :param padding_mode: one of the following: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

    :return: affinities of the input tensor
    """

    if input.ndim == 4:
        out = _binary_convolution2d(input, padding_mode)
    elif input.ndim == 5:
        out = _binary_convolution3d(input, padding_mode)
    else:
        raise RuntimeError(f'Number of dimensions not supported: {input.shape}')

    return out

#  Copyright Chris Buswinka, 2023
