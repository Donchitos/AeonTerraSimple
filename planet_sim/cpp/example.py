#!/usr/bin/env python3
"""
Example script showing how to use the C++ implementation from Python.
This script creates a spherical grid and visualizes it using matplotlib.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the build directory to the path to find the Python module
# Note: You'll need to adjust this to the actual build directory after compilation
sys.path.append('build/python')

try:
    from _aeonterracpp import SphericalGrid, BoundingBox, Vec3f
except ImportError:
    print("Error: Could not import the _aeonterracpp module.")
    print("Make sure you've compiled the C++ code and adjusted the path above.")
    sys.exit(1)

def main():
    # Create a spherical grid with Earth's radius and resolution 3
    print("Creating SphericalGrid...")
    grid = SphericalGrid(radius=6371.0, resolution=3)
    print(f"Created grid with {grid.vertex_count} vertices and {grid.face_count} faces")
    
    # Get vertices and faces as numpy arrays
    vertices = grid.get_vertices_array()
    faces = grid.get_faces_array()
    
    # Create a colormap based on elevation (distance from center)
    elevations = np.linalg.norm(vertices, axis=1)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot a subset of the faces for better visibility
    max_faces_to_plot = 1000
    face_subset = faces[:max_faces_to_plot]
    
    # Plot triangles
    for i, face in enumerate(face_subset):
        triangle = vertices[face]
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                      color=plt.cm.viridis(i / max_faces_to_plot),
                      alpha=0.7)
    
    # Setup plot
    ax.set_title(f"SphericalGrid (Resolution {grid.resolution})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Demonstrate coordinate conversion
    print("\nCoordinate conversion examples:")
    locations = [
        ("North Pole", 90.0, 0.0),
        ("South Pole", -90.0, 0.0),
        ("Greenwich", 51.4772, 0.0),
        ("Tokyo", 35.6895, 139.6917),
        ("Sydney", -33.8688, 151.2093),
    ]
    
    for name, lat, lon in locations:
        pos = grid.lat_lon_to_cartesian(lat, lon)
        nearest_idx = grid.find_nearest_vertex(lat, lon)
        nearest_pos = grid.get_vertex(nearest_idx)
        nearest_lat, nearest_lon = grid.cartesian_to_lat_lon(nearest_pos)
        
        print(f"{name}: ({lat}, {lon}) -> 3D position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        print(f"  Nearest vertex: {nearest_idx}, at ({nearest_lat:.2f}, {nearest_lon:.2f})")
    
    # Demonstrate region query
    print("\nRegion query example:")
    # Europe bounding box (approximate)
    europe = BoundingBox(lat_min=35.0, lon_min=-10.0, lat_max=70.0, lon_max=40.0)
    europe_vertices = grid.get_vertices_in_region(europe)
    print(f"Found {len(europe_vertices)} vertices in Europe region")

if __name__ == "__main__":
    main()