#!/usr/bin/env python3
"""
Example demonstrating the heightmap generation capabilities of AeonTerra.

This example:
1. Creates a planet with tectonic plates
2. Simulates tectonic processes for some time
3. Generates a heightmap from the resulting planet
4. Visualizes the heightmap and other derived maps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AeonTerra C++ bindings
try:
    from planet_sim.cpp.python import _aeonterracpp
except ImportError:
    print("Failed to import AeonTerra C++ bindings. Make sure they're built properly.")
    print("Try running: cd planet_sim/cpp && ./build.sh")
    sys.exit(1)

def main():
    print("AeonTerra Heightmap Generation Example")
    print("--------------------------------------")
    
    # Create a planet
    print("Creating planet...")
    planet = _aeonterracpp.Planet(radius=6371.0, resolution=7, name="ExamplePlanet")
    
    # Create a tectonic simulation
    print("Initializing tectonic simulation...")
    tectonics = _aeonterracpp.TectonicSimulation(planet, num_plates=7, seed=42)
    
    # Simulate tectonic activity (200 million years)
    print("Running tectonic simulation for 200 million years...")
    for i in range(10):
        print(f"  Simulating {i*20}-{(i+1)*20} million years...")
        tectonics.simulate_step(20.0)
    
    # Create a heightmap generator
    print("Creating heightmap generator...")
    heightmap_gen = _aeonterracpp.HeightmapGenerator(planet, tectonics)
    
    # Configure heightmap parameters
    params = _aeonterracpp.HeightmapParams()
    params.width = 1024
    params.height = 512
    params.max_elevation = 10.0
    params.min_elevation = -10.0
    params.noise_strength = 0.7
    params.mountain_roughness = 0.8
    params.erosion_iterations = 8
    params.tectonic_influence = 0.9
    params.climate_influence = 0.7
    heightmap_gen.set_parameters(params)
    
    # Generate the heightmap
    print("Generating heightmap...")
    heightmap_gen.generate(True)  # True for equirectangular projection
    
    # Get the heightmap data as a numpy array
    heightmap = heightmap_gen.data
    
    # Generate normal map and slope map
    print("Generating additional maps...")
    normal_map = heightmap_gen.generate_normal_map()
    slope_map = heightmap_gen.generate_slope_map()
    
    # Visualize the results
    print("Visualizing results...")
    visualize_results(heightmap, normal_map, slope_map, planet, tectonics)
    
    print("Done!")

def visualize_results(heightmap, normal_map, slope_map, planet, tectonics):
    """Visualize the generated heightmap and related data."""
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Heightmap visualization
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(heightmap, cmap='terrain', vmin=-10, vmax=10)
    ax1.set_title("Heightmap")
    plt.colorbar(im1, ax=ax1, label="Elevation (km)")
    
    # 2. Slope map visualization
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(slope_map.reshape(heightmap.shape), cmap='Greys_r')
    ax2.set_title("Slope Map")
    plt.colorbar(im2, ax=ax2, label="Slope")
    
    # 3. Plate map visualization
    ax3 = fig.add_subplot(223)
    
    # Convert plate IDs to a 2D array matching the heightmap dimensions
    plate_ids = np.array(planet.plate_id)
    
    # Sample plate IDs at points corresponding to the heightmap grid
    plate_map = np.zeros(heightmap.shape, dtype=int)
    h, w = heightmap.shape
    
    # Create a simple visualization - color each plate differently
    plate_colors = np.zeros((h, w, 3))
    unique_plates = np.unique(plate_ids)
    num_plates = len(unique_plates)
    
    # Generate a color for each plate ID
    color_map = {}
    for i, plate_id in enumerate(unique_plates):
        if plate_id >= 0:  # Skip unassigned vertices
            hue = i / num_plates
            color_map[plate_id] = plt.cm.hsv(hue)[:3]  # Get RGB from HSV colormap
    
    # Sample plate IDs and assign colors
    grid = planet.get_grid()
    for y in range(h):
        for x in range(w):
            # Convert pixel coordinates to lat/lon
            lon = (x / w) * 360.0 - 180.0
            lat = 90.0 - (y / h) * 180.0
            
            # Convert lat/lon to 3D position
            lat_rad = lat * (np.pi / 180.0)
            lon_rad = lon * (np.pi / 180.0)
            
            # Convert to Cartesian
            pos_x = np.cos(lat_rad) * np.cos(lon_rad)
            pos_y = np.cos(lat_rad) * np.sin(lon_rad)
            pos_z = np.sin(lat_rad)
            
            # Create position vector
            pos = _aeonterracpp.Vec3f(pos_x, pos_y, pos_z)
            pos = pos * grid.get_radius()
            
            # Find closest vertex
            vertex_idx = grid.find_closest_vertex(pos)
            if vertex_idx >= 0:
                plate_id = planet.get_plate_id(vertex_idx)
                plate_map[y, x] = plate_id
                
                if plate_id >= 0 and plate_id in color_map:
                    plate_colors[y, x] = color_map[plate_id]
    
    # Display the plate map with custom colors
    ax3.imshow(plate_colors)
    ax3.set_title("Tectonic Plates")
    
    # 4. Climate map visualization (precipitation)
    ax4 = fig.add_subplot(224)
    
    # Get precipitation data
    precipitation = np.array(tectonics.precipitation)
    
    # Sample precipitation data to match heightmap dimensions
    precip_map = np.zeros(heightmap.shape)
    for y in range(h):
        for x in range(w):
            # Use same mapping approach as above
            lon = (x / w) * 360.0 - 180.0
            lat = 90.0 - (y / h) * 180.0
            
            lat_rad = lat * (np.pi / 180.0)
            lon_rad = lon * (np.pi / 180.0)
            
            pos_x = np.cos(lat_rad) * np.cos(lon_rad)
            pos_y = np.cos(lat_rad) * np.sin(lon_rad)
            pos_z = np.sin(lat_rad)
            
            pos = _aeonterracpp.Vec3f(pos_x, pos_y, pos_z)
            pos = pos * grid.get_radius()
            
            vertex_idx = grid.find_closest_vertex(pos)
            if vertex_idx >= 0 and vertex_idx < len(precipitation):
                precip_map[y, x] = precipitation[vertex_idx]
    
    # Visualize precipitation
    im4 = ax4.imshow(precip_map, cmap='Blues')
    ax4.set_title("Precipitation (mm/year)")
    plt.colorbar(im4, ax=ax4)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()