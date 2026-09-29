"""AS2.5LM: adaptive semantic 2.5D LiDAR mapping.

Raw LiDAR scan -> SalsaNext semantic segmentation -> adaptive 2.5D grid (5-40 cm cells)
with elevation, roughness and traversability, on the GPU (C++/CUDA) or CPU (NumPy).
"""

__version__ = "1.0.0"
