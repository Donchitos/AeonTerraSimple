#!/usr/bin/env python3
"""
Example demonstrating the biome system in AeonTerra.

This example:
1. Creates a planet with tectonic plates
2. Simulates tectonic processes to generate a basic landscape
3. Initializes and runs the climate system
4. Initializes and runs the biome system
5. Visualizes biomes, vegetation cover, and soil properties
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.patches import Patch
import sys
import os
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AeonTerra C++ bindings
try:
    from planet_sim.cpp.python import _aeonterracpp
except ImportError:
    print("Failed to import AeonTerra C++ bindings. Make sure they're built properly.")
    print("Try running: cd planet_sim/cpp && ./build_climate.sh")
    sys.exit(1)

def main():
    print("AeonTerra Biome System Example")
    print("-------------------------------")
    
    # Create a planet with resolution level 6 (40,962 vertices) for better detail
    print("Creating planet...")
    planet = _aeonterracpp.EnhancedPlanet(radius=6371.0, resolution=6, name="BiomeWorld")
    
    # Create a tectonic simulation
    print("Initializing tectonic simulation...")
    tectonics = _aeonterracpp.EnhancedTectonicSimulation(planet, num_plates=8, seed=42)
    
    # Enable supercontinent cycle
    tectonics.enable_supercontinent_cycle(True)
    
    # Add a few mantle plumes
    tectonics.generate_random_mantle_plumes(4)
    
    # Simulate tectonic activity (200 million years)
    print("Running tectonic simulation for 200 million years...")
    for i in range(10):
        print(f"  Simulating {i*20}-{(i+1)*20} million years...")
        tectonics.simulate_step(20.0)
    
    # Initialize the climate system
    print("Initializing climate system...")
    climate = _aeonterracpp.ClimateSystem(planet)
    
    # Configure climate parameters
    params = _aeonterracpp.ClimateParams()
    params.axial_tilt = 23.5               # Earth-like axial tilt
    params.global_precipitation = 1000.0   # Global average precipitation in mm/year
    params.seasonal_variation = 1.0        # Normal seasonal variation
    params.monsoon_strength = 1.2          # Slightly stronger monsoons
    climate.set_parameters(params)
    
    # Initialize the climate system
    climate.initialize()
    
    # Run climate simulation for a full year at each step
    print("Running climate simulation...")
    climate_steps = 4  # Four seasons
    for i in range(climate_steps):
        day_of_year = i * (365.24 / climate_steps)
        print(f"  Simulating day {day_of_year:.1f}...")
        climate.simulate(0.1, day_of_year)  # 0.1 million years per step
    
    # Initialize the biome system
    print("Initializing biome system...")
    biome_system = _aeonterracpp.BiomeSystem(planet, climate)
    biome_system.initialize()
    
    # Run biome updates
    print("Running biome simulation...")
    for i in range(5):
        print(f"  Updating biomes, step {i+1}...")
        biome_system.update(0.2)  # 0.2 million years per step
    
    # Visualize the results
    print("Visualizing results...")
    visualize_results(planet, tectonics, climate, biome_system)
    
    print("Done!")

def visualize_results(planet, tectonics, climate, biome_system):
    """Visualize the biome system results."""
    # Get components
    atmosphere = climate.get_atmosphere()
    hydrology = climate.get_hydrology()
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Sample data for visualization
    grid_width = 360
    grid_height = 180
    
    # Create latitude-longitude grid for sampling
    lons = np.linspace(-180, 180, grid_width)
    lats = np.linspace(-90, 90, grid_height)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Create sample points to pass to C++ functions
    sample_points = []
    for i in range(grid_height):
        for j in range(grid_width):
            lat = lat_grid[i, j]
            lon = lon_grid[i, j]
            
            # Convert lat/lon to 3D position
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)
            
            # Create vector and scale to planet radius
            pos = _aeonterracpp.Vec3f(x, y, z)
            pos = pos * planet.get_radius()
            sample_points.append(pos)
    
    # 1. Elevation
    ax1 = fig.add_subplot(231)
    elevation_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            vertex_idx = planet.get_grid().find_closest_vertex(pos)
            if vertex_idx >= 0:
                elevation_data[i, j] = planet.get_elevation(vertex_idx)
    
    # Custom colormap for elevation
    terrain_cmap = plt.cm.terrain
    ocean_cmap = plt.cm.Blues_r
    
    # Create a custom colormap that uses terrain_cmap for positive values and ocean_cmap for negative
    elevation_colors = np.zeros((256, 4))
    for i in range(256):
        if i < 128:  # Ocean
            elevation_colors[i] = ocean_cmap(1.0 - i/127)
        else:  # Land
            elevation_colors[i] = terrain_cmap((i-128)/127)
    
    custom_cmap = colors.ListedColormap(elevation_colors)
    
    # Plot elevation with custom colormap
    im1 = ax1.imshow(elevation_data, cmap=custom_cmap, extent=[-180, 180, -90, 90], origin='lower')
    ax1.set_title("Elevation")
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, label="Elevation (km)")
    
    # 2. Temperature
    ax2 = fig.add_subplot(232)
    temperature_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            temperature_data[i, j] = atmosphere.get_temperature(pos)
    
    # Plot temperature
    im2 = ax2.imshow(temperature_data, cmap='RdBu_r', extent=[-180, 180, -90, 90], origin='lower')
    ax2.set_title("Temperature")
    plt.colorbar(im2, ax=ax2, label="Temperature (°C)")
    
    # 3. Precipitation
    ax3 = fig.add_subplot(233)
    precip_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            precip_data[i, j] = hydrology.get_precipitation(pos)
    
    # Plot precipitation
    im3 = ax3.imshow(precip_data, cmap='Blues', extent=[-180, 180, -90, 90], origin='lower')
    ax3.set_title("Annual Precipitation")
    plt.colorbar(im3, ax=ax3, label="Precipitation (mm/year)")
    
    # 4. Biome Distribution
    ax4 = fig.add_subplot(234)
    biome_data = np.zeros((grid_height, grid_width), dtype=int)
    biome_colors = {}
    biome_count = defaultdict(int)
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            biome_type = biome_system.get_biome_type(pos)
            
            # Get biome properties
            biome_id = int(biome_type)
            biome_data[i, j] = biome_id
            
            # Get biome color and name
            if biome_id not in biome_colors:
                color = biome_system.get_biome_color(biome_type)
                biome_colors[biome_id] = [float(color[0]), float(color[1]), float(color[2])]
            
            # Count biome occurrences
            biome_name = biome_system.get_biome_name(biome_type)
            biome_count[biome_name] += 1
    
    # Create a custom colormap for biomes
    n_biomes = len(biome_colors)
    biome_cmap_colors = np.zeros((n_biomes, 4))
    for i, biome_id in enumerate(sorted(biome_colors.keys())):
        biome_cmap_colors[i, 0:3] = biome_colors[biome_id]
        biome_cmap_colors[i, 3] = 1.0  # Alpha
    
    biome_cmap = colors.ListedColormap(biome_cmap_colors)
    bounds = np.arange(-0.5, n_biomes + 0.5, 1)
    norm = colors.BoundaryNorm(bounds, biome_cmap.N)
    
    # Plot biomes
    im4 = ax4.imshow(biome_data, cmap=biome_cmap, norm=norm, extent=[-180, 180, -90, 90], origin='lower')
    ax4.set_title("Biome Distribution")
    
    # Create legend for the top 10 most common biomes
    top_biomes = sorted(biome_count.items(), key=lambda x: x[1], reverse=True)[:10]
    legend_elements = []
    for biome_name, count in top_biomes:
        # Find the biome ID by name
        for biome_id in range(int(_aeonterracpp.BiomeType.Count)):
            if biome_system.get_biome_name(_aeonterracpp.BiomeType(biome_id)) == biome_name:
                color = biome_colors.get(biome_id, [0.5, 0.5, 0.5])
                legend_elements.append(Patch(facecolor=color, label=f"{biome_name}"))
                break
    
    ax4.legend(handles=legend_elements, loc='lower right', fontsize='x-small')
    
    # 5. Vegetation Cover
    ax5 = fig.add_subplot(235)
    veg_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            veg_data[i, j] = biome_system.get_vegetation_density(pos)
    
    # Plot vegetation
    im5 = ax5.imshow(veg_data, cmap='Greens', extent=[-180, 180, -90, 90], origin='lower')
    ax5.set_title("Vegetation Cover")
    plt.colorbar(im5, ax=ax5, label="Density (0-1)")
    
    # 6. Soil Depth
    ax6 = fig.add_subplot(236)
    soil_data = np.zeros((grid_height, grid_width))
    ocean_mask = elevation_data < 0
    
    for i in range(grid_height):
        for j in range(grid_width):
            # Skip ocean areas
            if ocean_mask[i, j]:
                soil_data[i, j] = np.nan
                continue
                
            idx = i * grid_width + j
            pos = sample_points[idx]
            
            # Get soil depth through biome properties
            props = biome_system.get_biome_properties(pos)
            soil_data[i, j] = props.soil_depth
    
    # Plot soil depth with masked oceans
    im6 = ax6.imshow(soil_data, cmap='YlOrBr', extent=[-180, 180, -90, 90], origin='lower')
    ax6.set_title("Soil Depth")
    plt.colorbar(im6, ax=ax6, label="Depth (meters)")
    
    # Adjust layout and show
    plt.tight_layout()
    plt.savefig("biome_example_output.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()