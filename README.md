# Pykonal 

Pykonal is a library which solves the Eikonal Function for 2D and 3D instance masks, using the Fast Iterative Method. 
It achieves memory efficiency through a fused kernel written in triton. 
It also provides functionality to calculate gradients of the eikonal field. 

This implementation uses convolutions to calculate affinity masks, which may be faster and
can occur on cuda, however uses dense representations of the affinity masks and therefore is 
memory intensive. It may be possible for much of the mask operations to occur with sparse tensors
for memory efficiency. 

A big thanks to Kevin Cutler (Original Omnipose Author) for helping me create this. 

Example:

```python
from src.eikonal import solve_eikonal, gradient_from_eikonal
import torch

image = torch.load('path/to/my/image.pt')  # An image with shape (B, C=1, X, Y, Z)
eikonal = solve_eikonal(image)  # A Distance map with shape (B, C=1, X, Y, Z)
gradients = gradient_from_eikonal(eikonal)  # A Gradient tensor with shape (B, C=3, X, Y, Z)
```
