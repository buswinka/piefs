import torch
from torch import Tensor
import triton
import triton.language as tl

@triton.jit
def _eikonal_3d_step_kernel(
    # pointers to matricies
    phi_ptr,
    mask_ptr,
    out_ptr,
    # shape of matricies
    x_shape,
    y_shape,
    z_shape,
    # strides of matricies
    x_stride,
    y_stride,
    z_stride,
):
    """
    This is about the stupidest way you can do this, but it should be fast...

    :param phi_ptr: pointer of previous step of eikonal iteration
    :param mask_ptr: pointer to instance mask with identical shape, and stride as phi
    :param out_ptr: tensor storing current step results iwth identical shape and stride as phi
    :param x_shape: shape of x dim
    :param y_shape: shape of y dim
    :param z_shape: shape of z dim
    :param x_stride: stride of x dim
    :param y_stride: stride of y dim
    :param zstride: stride of z dim
    :return: None. Does everything in place.
    """
    # padding
    x0 = tl.program_id(axis=0)
    y0 = tl.program_id(axis=1)
    z0 = tl.program_id(axis=2)

    mask_center = tl.load(input + (x0 * x_stride + y0 * y_stride + z0 * z_stride))
    if mask_center == 0:
        return

    # phi affinities X with boundary checks
    # if an affinity goes to a boundary, assign it as self
    if x0 != 0:
        _phi_x0 = tl.load(phi_ptr + ((x0 - 1) * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_x0 = tl.load(mask_ptr + ((x0 - 1) * x_stride + y0 * y_stride + z0 * z_stride))
    else:
        _phi_x0 = tl.load(phi_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_x0 = tl.load(mask_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))

    if x0 + 1 != x_shape:
        _phi_x1 = tl.load(phi_ptr + ((x0 + 1) * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_x1 = tl.load(mask_ptr + ((x0 + 1) * x_stride + y0 * y_stride + z0 * z_stride))
    else:
        _phi_x1 = tl.load(phi_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_x1 = tl.load(mask_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))

    # phi affinities Y
    if y0 != 0:
        _phi_y0 = tl.load(phi_ptr + x0 * x_stride + (y0-1) * y_stride + z0 * z_stride)
        _mask_y0 = tl.load(mask_ptr + x0 * x_stride + (y0-1) * y_stride + z0 * z_stride)
    else:
        _phi_y0 = tl.load(phi_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_y0 = tl.load(mask_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))

    if y0 + 1 != y_shape:
        _phi_y1 = tl.load(phi_ptr + x0 * x_stride + (y0+1) * y_stride + z0 * z_stride)
        _mask_y1 = tl.load(mask_ptr + x0 * x_stride + (y0+1) * y_stride + z0 * z_stride)
    else:
        _phi_y1 = tl.load(phi_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_y1 = tl.load(mask_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))

    # phi affinities Z
    if z0 != 0:
        _phi_z0 = tl.load(phi_ptr + x0 * x_stride + y0 * y_stride + (z0-1) * z_stride)
        _mask_z0 = tl.load(mask_ptr + x0 * x_stride + y0 * y_stride + (z0-1) * z_stride)
    else:
        _phi_z0 = tl.load(phi_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_z0 = tl.load(mask_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))

    if z0 + 1 != z_shape:
        _phi_z1 = tl.load(phi_ptr + x0 * x_stride + y0 * y_stride + (z0+1) * z_stride)
        _mask_z1 = tl.load(mask_ptr + x0 * x_stride + y0 * y_stride + (z0+1) * z_stride)
    else:
        _phi_z1 = tl.load(phi_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))
        _mask_z1 = tl.load(mask_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride))


    # Zero out non adjacent pixels
    if _mask_x0 != mask_center and x0 != 1:
        _phi_x0 = 0.0
    if _mask_x1 != mask_center and x0+1 != x_shape:
        _phi_x1 = 0.0

    if _mask_y0 != mask_center and y0 != 1:
        _phi_y0 = 0.0
    if _mask_y1 != mask_center and y0+1 != x_shape:
        _phi_y1 = 0.0

    if _mask_z0 != mask_center and z0 != 1:
        _phi_z0 = 0.0
    if _mask_z1 != mask_center and y0+1 != x_shape:
        _phi_z1 = 0.0

    minx = tl.minimum(_phi_x0, _phi_x1)
    miny = tl.minimum(_phi_y0, _phi_y1)
    minz = tl.minimum(_phi_z0, _phi_z1)

    # sort into a1, a2, a3 where a1 < a2 < a3
    a1 = tl.minimum(tl.minimum(minx, miny), minz)
    a3 = tl.maximum(tl.maximum(minx, miny), minz)

    a2 = minx + miny + minz - (a1 + a3)

    # UPDATES
    if tl.abs(a1 - a3) < 1:
        suma = a1 + a2 + a3
        suma2 = (a1 * a1) + (a2 * a2) + (a3 * a3)
        next_step = ((2 * suma) + tl.sqrt((4 * suma * suma) - (12 * (suma2 - 1)))) / 6
    elif tl.abs(a1 - a2) < 1:
        next_step = (a1 + a2 + tl.sqrt(2 - ((a1 - a2) * (a1 - a2)))) / 2
    else:
        next_step = a1 + 1.0

    tl.store(out_ptr + (x0 * x_stride + y0 * y_stride + z0 * z_stride), next_step)


def update3d(phi: Tensor, instance_mask: Tensor) -> Tensor:
    """
    Applies a single step update of the eikonal eq with a fused kernel.

    Assumes f=1, and uses FIM.

    :param phi: last step.
    :return:
    """

    assert phi.ndim == 3, "may only be applied to tensor with shape of 3"
    assert phi.is_cuda, "only works on cuda tensors"
    assert phi.device == instance_mask.device

    with torch.cuda.device(phi.device):  # somehow fixes cuda:1 issue
        x, y, z = phi.shape

        instance_mask = instance_mask.contiguous()
        phi = phi.contiguous()

        output = torch.zeros_like(phi)

        for i in range(phi.ndim):
            assert phi.stride(i) == instance_mask.stride(i), "strides must be the same"

        # grid = lambda META: (x, y, z)  # launch kernel for each px with 3D grid
        # print(grid)
        _eikonal_3d_step_kernel[(x, y, z)](
            phi, instance_mask, output, x, y, z, phi.stride(0), phi.stride(1), phi.stride(2)
        )

    return output

