from math import sqrt
from typing import List

import torch
from torch import Tensor
from src.morphology import binary_convolution


def _apply_update(minimum_paired: Tensor, f: float) -> Tensor:
    """
    Partial Psi from one direction of connected components.
    Should only be called from eikonal_single_step.

    Written by Kevin Cutler from Omnipose, adapted by Chris Buswinka.

    Shapes:
        - minimum_paired: (N_pairs, B, C, ...)
        - return (B, C, ...)

    :param minimum_paired: minimums from eikonal_single_step.
    :param f: distance of pair from the central pixel

    :return: partial psi for that neighborhood.
    """

    # Four necessary channels, remaining num are spatial dims...
    d: int = len(minimum_paired.shape) - 3

    # sorting was the source of the small artifact bug
    minimum_paired, _ = torch.sort(minimum_paired, dim=0)

    a = minimum_paired * ((minimum_paired - minimum_paired[-1, ...]) < f)

    sum_a = a.sum(dim=0)
    sum_a2 = (a ** 2).sum(dim=0)

    out = (1 / d) * (
            sum_a + torch.sqrt(torch.clamp((sum_a ** 2) - d * (sum_a2 - f ** 2), min=0))
    )

    return out


def eikonal_single_step(connected_components: Tensor) -> Tensor:
    """
    Returns the output of one iteration of an eikonal equation.

    Shapes:
        - connected_components: (B, C, N_components=9, X, Y) or (B, C, N_components=27, X, Y, Z)
        - returns: (B, C, X, Y) or (B, C, X, Y, Z)

    :param connected_components: The connected components of the previous step of the eikonal update function.
    :return: the result of the next step of a solution to the eikonal equation.
    """

    # The solution needs to understand which connected pixel of affinity graph are how far from the central pixel
    # Omnipose generalizes this to ND. I am not smart enough to do this.
    if connected_components.ndim == 5:  # 2D
        factors: List[float] = [0.0, 1.0, sqrt(2)]
        index_list: List[List[int]] = [[4], [1, 3, 5, 7], [0, 2, 6, 8]]

    elif connected_components.ndim == 6:  # 3D
        factors: List[float] = [0.0, 1.0, sqrt(2), sqrt(3)]
        index_list: List[List[int]] = [
            [13],
            [4, 10, 12, 14, 16, 22],
            [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25],
            [0, 2, 6, 8, 18, 20, 24, 26],
        ]
    else:
        raise RuntimeError(
            f"Number of dimensions: {len(connected_components.shape) - 3} is not supported."
        )

    phi = torch.ones_like(connected_components[:, :, 0, ...])

    # find the minimum of each hypercube pair along each axis.
    for ind, f in zip(index_list[1:], factors[1:]):
        n_pair = len(ind) // 2
        minimum_paired = torch.stack(
            [
                torch.minimum(
                    connected_components[:, :, ind[i], ...],
                    connected_components[:, :, ind[-(i + 1)], ...],
                )
                for i in range(n_pair)
            ]
        )
        phi.mul_(_apply_update(minimum_paired, f))

    phi = torch.pow(phi, 1 / 2)
    return phi


def solve_eikonal(
        instance_mask: Tensor, eps: float = 1e-3, min_steps: int = 51
) -> Tensor:
    """
    Solves the eikonal equation on a collection of instance masks. In practice, generates
    a specialized distance mask for each instance of an object in an image. Input may be a 2D or 3D image.

    Shapes:
        - instance_mask (B, C, X, Y) or (B, C, X, Y, Z)
        - solution to eikonal function (B, C, X, Y) or (B, C, X, Y, Z)

    Data Types:
        - instance_mask: int
        - returns: float

    Examples:
        >>> from src.eikonal import solve_eikonal
        >>> import torch
        >>>
        >>> image = torch.load('path/to/my/image.pt')  # An image with shape (B, C, X, Y, Z)
        >>> eikonal = solve_eikonal(image)

    :param instance_mask: Input mask with instances denoted by integers and zero as background
    :param eps: Minimum tolerable error to eikonal function
    :param min_steps: Minimum number of iterations to solve eikonal function
    :return: Solution to eikonal function. Basically a fancy smooth distance map.
    """

    # Get the values of adjacent pixels of the input image.
    # Returns a (B, C, N, X, Y, Z?) image where N=9 for a 2D image, and 27 for a 3D image.
    affinity_mask: Tensor = binary_convolution(instance_mask, padding_mode="replicate")

    # Remove all connections to pixels with different labels from center
    affinity_mask[affinity_mask != instance_mask.unsqueeze(2)] = 0.0

    # Mask for removing updates to background pixels
    affinity_mask = affinity_mask.gt(0)
    semantic_mask = instance_mask.gt(0)

    T, T0 = torch.ones_like(instance_mask), torch.ones_like(instance_mask)

    t = 0
    error = float("inf")

    while error > eps and t < min_steps:

        T: Tensor = eikonal_single_step(
            binary_convolution(T, "replicate") * affinity_mask
        )

        T.mul_(semantic_mask)  # zero out background

        error = (T - T0).square().mean()

        if t < 1:  # Omnipose includes smoothing at t=0.
            T = (
                binary_convolution(T, "replicate").mul(affinity_mask).mean(2)
            )  # Returns B, C, N, ...

        T0.copy_(T)
        t += 1

    return T

#  Copyright Chris Buswinka, 2023

