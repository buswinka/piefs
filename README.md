# omnipose_pure_torch
Solves the Eikonal Function on instance masks in an 2D/3D Image -- The Omnipose Target Function

This implementation uses convolutions to calculate affinity masks, which may be faster and
can occur on cuda, however uses dense representations of the affinity masks and therefore is 
memory intensive. It may be possible for much of the mask operations to occur with sparse tensors
for memory efficiency. 

A big thanks to Kevin Cutler (Original Omnipose Author) for helping me create this. 

Example:
```python
from src.eikonal import solve_eikonal
import torch

image = torch.load('path/to/my/image.pt')  # An image with shape (B, C, X, Y, Z)
eikonal = solve_eikonal(image) # A Distance map with shape (B, C, X, Y, Z)
```
