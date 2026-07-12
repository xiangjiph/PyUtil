"""Point-cloud geometry public API.

The package is split by responsibility while preserving imports from
``pyutil.geometry.point_cloud``.
"""

from functools import cached_property
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from ... import stat, util
from .downsample import (
    GridDownsamplerND,
    downsample_points_by_averaging,
    grid_centroids,
)
from .group import PointCloudGroup
from .legacy_surface import PointCloud3DSurfaceFit
from .outlier import select_points_near_pc1, select_points_near_pc1_iterative
from .surface import PCSurface3D, PolySurface3D

PointCloud3DPolynomialSurface = PolySurface3D
PointCloud3DPolynomialSurfaceFit = PCSurface3D

__all__ = [
    "GridDownsamplerND",
    "PCSurface3D",
    "PointCloud3DPolynomialSurface",
    "PointCloud3DPolynomialSurfaceFit",
    "PointCloud3DSurfaceFit",
    "PointCloudGroup",
    "PolySurface3D",
    "downsample_points_by_averaging",
    "grid_centroids",
    "select_points_near_pc1",
    "select_points_near_pc1_iterative",
]
