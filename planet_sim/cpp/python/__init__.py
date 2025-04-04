"""
AeonTerra C++ Implementation - Python Bindings

This package provides Python access to the high-performance C++ implementation
of the AeonTerra planet generation system.
"""

from ._aeonterracpp import (
    # Vector types
    Vec3f, Vec3d, Vec3i, BoundingBox,
    
    # Core classes
    SphericalGrid,
    
    # Utility functions
    to_radians, to_degrees
)

__version__ = "0.1.0"