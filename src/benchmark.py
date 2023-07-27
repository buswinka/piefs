import timeit
from omnipose.core import masks_to_flows_batch
from src.eikonal import solve_eikonal
import skimage.io as io


image = io.imread(
    "/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/plant-root/test/Movie1_t00045_crop_gt.labels.tif"
)
image = image[[100], ...]

omni_setup = """
from omnipose.core import masks_to_flows_batch
from src.eikonal import solve_eikonal
import torch
import skimage.io as io
import numpy as np


image = io.imread(
    "/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/plant-root/test/Movie1_t00045_crop_gt.labels.tif"
)[[100], 0:100, 0:100]
image = image.astype(int)
"""

chris_setup = """
from omnipose.core import masks_to_flows_batch
from src.eikonal import solve_eikonal
import torch
import skimage.io as io


image = io.imread(
    "/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/plant-root/test/Movie1_t00045_crop_gt.labels.tif"
)
image = torch.from_numpy(image.astype(int)).unsqueeze(0).float().cuda()[[100], 0:100, 0:100]

compiled_solve_eikonal = torch.compile(solve_eikonal)
compiled_solve_eikonal(image)  # call function once because compilation occurs in background...

"""

omni_time = timeit.timeit("masks_to_flows_batch(image)", setup=omni_setup, number=10)
chris_time = timeit.timeit("compiled_solve_eikonal(image)", setup=chris_setup, number=10)


print(f'{omni_time=}, {chris_time=}')  # roughly 5x speedup with CUDA on my machine...




