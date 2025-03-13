#!/usr/bin/env python3
"""
Test script for enhanced AeonTerra visualizations
"""

import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from planet_sim.core.planet import Planet
from planet_sim.core.tectonics import TectonicSimulation

def ensure_output_dir():
    """Make sure output directory exists"""
    os.makedirs("output", exist_ok=True)

def visualize_land_water(planet, save_path=None, show=False, projection='mercator'):
    """
    Visualize land/water distribution with green/blue coloring.
    
    Parameters:
    - planet: The Planet object
    - save_path: If provided, save the visualization to this file path
    - show: Whether to display the plot
    - projection: Map projection to use
    """
    # Create figure
    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Setup the projection
    width, height = 1200, 600  # Pixels for projection calculation
    
    # Draw each vertex with land or water color
    for i, vertex in enumerate(planet.grid.vertices):
        # Convert vertex to lat/lon
        x, y, z = vertex
        lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
        lon = np.arctan2(y, x) * 180 / np.pi
        
        # Project to 2D based on projection type
        if projection == 'mercator':
            px, py = lat_lon_to_mercator(lat, lon, width, height)
        else:  # Default to equirectangular
            px = (lon + 180) * width / 360
            py = (90 - lat) * height / 180
        
        # Skip if outside bounds
        if px < 0 or px >= width or py < 0 or py >= height:
            continue
        
        # Plot point with land/water color
        elev = planet.elevation[i]
        color = 'green' if elev >= 0 else 'blue'
        ax.scatter(px, py, c=color, s=5, alpha=0.8)
    
    # Add rivers if available
    if hasattr(planet, 'rivers') and planet.rivers:
        for river in planet.rivers:
            if len(river) > 1:
                river_x = []
                river_y = []
                
                for idx in river:
                    vertex = planet.grid.vertices[idx]
                    x, y, z = vertex
                    lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
                    lon = np.arctan2(y, x) * 180 / np.pi
                    
                    if projection == 'mercator':
                        px, py = lat_lon_to_mercator(lat, lon, width, height)
                    else:
                        px = (lon + 180) * width / 360
                        py = (90 - lat) * height / 180
                    
                    if 0 <= px < width and 0 <= py < height:
                        river_x.append(px)
                        river_y.append(py)
                
                if river_x and river_y:
                    ax.plot(river_x, river_y, 'blue', linewidth=1.5, alpha=0.7)
    
    # Set axis limits
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis for standard map orientation
    
    # Add title and grid
    plt.title(f'Land and Water Distribution (Age: {planet.age:.1f} million years)')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label='Land'),
        Patch(facecolor='blue', edgecolor='blue', label='Water')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        if not show:
            plt.close(fig)
        print(f"Land/water visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    return save_path if save_path else None

def visualize_heightmap(planet, save_path=None, show=False, projection='mercator'):
    """
    Visualize planet as a grayscale heightmap.
    
    Parameters:
    - planet: The Planet object
    - save_path: If provided, save the visualization to this file path
    - show: Whether to display the plot
    - projection: Map projection to use
    """
    # Create figure
    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Setup the projection
    width, height = 1200, 600  # Pixels for projection calculation
    
    # Find elevation extremes for normalization
    min_elev = min(planet.elevation)
    max_elev = max(planet.elevation)
    
    # Create a grayscale norm
    norm = colors.Normalize(vmin=min_elev, vmax=max_elev)
    
    # Prepare pixel data
    pixel_data = np.zeros((height, width))
    pixel_count = np.zeros((height, width))
    
    # Draw each vertex with height color
    for i, vertex in enumerate(planet.grid.vertices):
        # Convert vertex to lat/lon
        x, y, z = vertex
        lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
        lon = np.arctan2(y, x) * 180 / np.pi
        
        # Project to 2D based on projection type
        if projection == 'mercator':
            px, py = lat_lon_to_mercator(lat, lon, width, height)
        else:  # Default to equirectangular
            px = (lon + 180) * width / 360
            py = (90 - lat) * height / 180
        
        # Ensure coordinates are within bounds
        px_int = int(px)
        py_int = int(py)
        
        if 0 <= px_int < width and 0 <= py_int < height:
            # Add elevation to pixel grid
            pixel_data[py_int, px_int] += planet.elevation[i]
            pixel_count[py_int, px_int] += 1
    
    # Calculate average elevation where multiple vertices map to the same pixel
    mask = pixel_count > 0
    pixel_data[mask] = pixel_data[mask] / pixel_count[mask]
    
    # Display the heightmap
    img = ax.imshow(pixel_data, cmap='gray', norm=norm, extent=[0, width, height, 0])
    
    # Add title and color bar
    plt.title(f'Elevation Heightmap (Age: {planet.age:.1f} million years)')
    plt.colorbar(img, ax=ax, label='Elevation (km)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        if not show:
            plt.close(fig)
        print(f"Heightmap visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    return save_path if save_path else None

def lat_lon_to_mercator(lat, lon, width, height):
    """
    Convert latitude/longitude to Mercator projection x,y coordinates.
    
    Parameters:
    - lat: Latitude in degrees
    - lon: Longitude in degrees
    - width: Width of the output image in pixels
    - height: Height of the output image in pixels
    
    Returns:
    - x, y coordinates in the Mercator projection
    """
    # Constrain latitude to prevent infinite scaling near poles
    lat = max(min(lat, 85), -85)
    
    # Mercator projection formulas
    x = (lon + 180) * (width / 360)
    
    # Convert latitude to radians
    lat_rad = lat * math.pi / 180
    
    # Mercator y-coordinate formula
    merc_n = math.log(math.tan((math.pi / 4) + (lat_rad / 2)))
    y = (height / 2) - (width * merc_n / (2 * math.pi))
    
    return x, y

def main():
    """Run a simple test to generate and visualize a planet"""
    print("Creating planet...")
    
    # Create a planet with custom parameters
    planet = Planet(radius=6371.0, resolution=4)
    
    # Initialize tectonic simulation
    tectonics = TectonicSimulation(
        planet,
        num_plates=8,
        initial_age=4500,
        tectonic_activity=1.0,
        continental_fraction=0.3
    )
    
    # Initialize plates and mantle
    tectonics.initialize_mantle(num_cells=12, num_plumes=5)
    tectonics.initialize_plates()
    
    # Run a quick simulation
    for i in range(3):
        tectonics.simulate_step(time_step=100.0)
    
    # Create output directory
    ensure_output_dir()
    
    # Generate the visualizations
    print("Generating visualizations...")
    
    # Standard 3D view
    planet.visualize(save_path="output/planet_3d.png", mode='elevation')
    
    # Enhanced 2D maps
    tectonics.visualize_plates_2d(save_path="output/plates_mercator.png", 
                                 projection='mercator', show_features=True)
    
    tectonics.visualize_plates_2d(save_path="output/plates_equirectangular.png", 
                                 projection='equirectangular', show_features=True)
    
    # Land/water map - use the function directly rather than a method
    visualize_land_water(planet, save_path="output/land_water.png", 
                        projection='mercator')
    
    # Heightmap - use the function directly rather than a method
    visualize_heightmap(planet, save_path="output/heightmap.png", 
                       projection='mercator')
    
    print("Visualizations complete! Check the 'output' directory.")

if __name__ == "__main__":
    main()