#!/usr/bin/env python3
"""
Example script for using the AeonTerra C++ Planet implementation from Python.
This demonstrates creating a planet, modifying properties, advancing time,
and serialization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import our C++ module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'test_build'))
from _aeonterracpp import (
    Planet, BoundingBox, Vec3f, to_radians, to_degrees
)

def visualize_planet(planet, title="Planet Visualization"):
    """Visualize the planet in 3D with elevation coloring."""
    # Get the grid from the planet
    grid = planet.get_grid()
    vertex_count = grid.get_vertex_count()
    
    # Extract vertex positions and property data
    positions = []
    elevations = []
    crust_types = []
    
    for i in range(vertex_count):
        pos = grid.get_vertex(i)
        positions.append((pos.x, pos.y, pos.z))
        elevations.append(planet.get_elevation(i))
        crust_types.append(planet.get_crust_type(i))
    
    # Convert to NumPy arrays
    positions = np.array(positions)
    elevations = np.array(elevations)
    crust_types = np.array(crust_types)
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap: blue for oceans, green/brown for land
    colors = np.zeros((len(elevations), 3))
    
    # Ocean (blue, darker for deeper)
    ocean_mask = elevations < 0
    ocean_depth_normalized = np.clip(-elevations[ocean_mask] / 10.0, 0, 1)
    colors[ocean_mask, 2] = 0.5 + (0.5 * (1 - ocean_depth_normalized))  # Blue
    
    # Land (green to brown based on elevation)
    land_mask = ~ocean_mask
    land_elevation_normalized = np.clip(elevations[land_mask] / 3.0, 0, 1)
    colors[land_mask, 0] = 0.5 * land_elevation_normalized  # Red
    colors[land_mask, 1] = 0.3 + (0.4 * (1 - land_elevation_normalized))  # Green
    
    # Plot the points
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=colors, s=10, alpha=0.8
    )
    
    # Set the aspect ratio to be equal
    ax.set_box_aspect([1,1,1])
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set title
    ax.set_title(title)
    
    # Add info text
    info_text = (
        f"Planet: {planet.get_name()}\n"
        f"Age: {planet.get_age():.2f} million years\n"
        f"Vertices: {vertex_count}\n"
        f"Elevation range: {min(elevations):.2f} to {max(elevations):.2f} km"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace')
    
    return fig

def main():
    print("AeonTerra Planet Example")
    print("=======================")
    
    # Create a new planet (radius=6371 km, resolution=4)
    print("\nCreating new planet...")
    planet = Planet(6371.0, 4, "Example Planet")
    print(f"Planet created: {planet.get_name()}")
    print(f"  Resolution: {planet.get_grid().get_resolution()}")
    print(f"  Vertex count: {planet.get_vertex_count()}")
    
    # Get some basic stats
    min_elevation = float('inf')
    max_elevation = float('-inf')
    ocean_count = 0
    
    for i in range(planet.get_vertex_count()):
        elev = planet.get_elevation(i)
        min_elevation = min(min_elevation, elev)
        max_elevation = max(max_elevation, elev)
        if elev < 0:
            ocean_count += 1
    
    ocean_percent = (ocean_count / planet.get_vertex_count()) * 100
    print(f"  Elevation range: {min_elevation:.2f} to {max_elevation:.2f} km")
    print(f"  Ocean coverage: {ocean_percent:.1f}%")
    
    # Modify elevations in a region
    print("\nModifying planet terrain...")
    bbox = BoundingBox(-30.0, 30.0, 30.0, 90.0)  # Region in lat/lon
    vertices_in_region = planet.get_grid().get_vertices_in_region(bbox)
    print(f"  Found {len(vertices_in_region)} vertices in region")
    
    # Create a mountain range
    for vertex_idx in vertices_in_region:
        pos = planet.get_grid().get_vertex(vertex_idx)
        lat, lon = planet.get_grid().cartesian_to_lat_lon(pos)
        
        # Distance from center of the region
        center_lat, center_lon = 0.0, 60.0
        dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
        
        # Make mountains that taper by distance
        if dist < 25:
            height = 5.0 * (1.0 - dist/25.0)**2
            planet.set_elevation(vertex_idx, height)
            planet.set_crust_type(vertex_idx, 1)  # Continental
            planet.set_crust_thickness(vertex_idx, 35.0 + height * 2)
    
    # Advance the planet's age
    print("\nAdvancing planet age...")
    planet.advance_age(25.0)
    planet.record_snapshot()
    print(f"  Current age: {planet.get_age()} million years")
    print(f"  Snapshots recorded: {len(planet.get_history())}")
    
    # Save the planet to a file
    filename = "example_planet.bin"
    print(f"\nSaving planet to {filename}...")
    planet.save_to_file(filename)
    
    # Load the planet from the file
    print(f"\nLoading planet from {filename}...")
    loaded_planet = Planet.load_from_file(filename)
    print(f"  Loaded: {loaded_planet.get_name()}")
    print(f"  Age: {loaded_planet.get_age()} million years")
    print(f"  Vertex count: {loaded_planet.get_vertex_count()}")
    
    # Visualize
    print("\nVisualizing planet...")
    fig = visualize_planet(loaded_planet)
    
    # Save the visualization
    output_image = "example_planet.png"
    fig.savefig(output_image)
    print(f"  Visualization saved to {output_image}")
    
    # Show the visualization (comment this out in non-interactive environments)
    # plt.show()
    
    print("\nExample completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())