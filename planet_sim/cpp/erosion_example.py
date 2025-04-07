#!/usr/bin/env python3
"""
Example demonstrating the integrated erosion system in AeonTerra.

This example:
1. Creates a planet with tectonic plates
2. Simulates tectonic processes to generate a basic landscape
3. Initializes and runs the climate system
4. Initializes and runs the biome system
5. Runs the enhanced erosion simulation with all processes
6. Visualizes the interconnected impacts of these systems
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
    print("AeonTerra Integrated Erosion System Example")
    print("-------------------------------------------")
    
    # Create a planet with resolution level 6 (40,962 vertices) for better detail
    print("Creating planet...")
    planet = _aeonterracpp.EnhancedPlanet(radius=6371.0, resolution=6, name="ErosionWorld")
    
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
    
    # Save initial elevation for comparison
    vertices = planet.get_vertex_count()
    initial_elevation = np.zeros(vertices)
    for i in range(vertices):
        initial_elevation[i] = planet.get_elevation(i)
    
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
    
    # Run climate simulation for a full year
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
    for i in range(2):
        print(f"  Updating biomes, step {i+1}...")
        biome_system.update(0.1)  # 0.1 million years per step
    
    # Initialize the enhanced erosion system
    print("Initializing erosion system...")
    erosion = _aeonterracpp.EnhancedErosion(planet, climate, biome_system)
    
    # Configure erosion parameters
    erosion_params = _aeonterracpp.ErosionParams()
    erosion_params.global_erosion_rate = 1.0       # Earth-like erosion rate
    erosion_params.fluvial_erosion_strength = 1.5  # Stronger river erosion
    erosion_params.coastal_erosion_strength = 1.2  # Slightly stronger coastal erosion
    erosion_params.time_scale_factor = 100.0       # Accelerate for demonstration
    erosion.set_parameters(erosion_params)
    
    # Initialize erosion system
    erosion.initialize(erosion_params)
    
    # Run erosion simulation
    print("Running erosion simulation...")
    total_erosion_time = 5.0  # 5 million years
    erosion_steps = 5
    step_time = total_erosion_time / erosion_steps
    
    # Save erosion data for each step
    erosion_history = []
    
    for i in range(erosion_steps):
        print(f"  Erosion step {i+1}: {step_time} million years...")
        
        # Run the erosion step
        erosion.simulate_step(step_time)
        
        # Save erosion rates for visualization
        erosion_data = np.zeros(vertices)
        for j in range(vertices):
            pos = planet.get_grid().get_vertex(j)
            erosion_data[j] = erosion.get_erosion_rate(pos)
        
        erosion_history.append(erosion_data.copy())
        
        # Update biome effects from erosion (feedback loop)
        erosion.update_biome_effects(biome_system)
        
        # Update hydrology effects from erosion (feedback loop)
        erosion.update_hydrology_effects(climate)
        
        # Update biomes to reflect changes in terrain
        biome_system.update(step_time)
        
        # Update climate to reflect changes in terrain
        climate.simulate(step_time, 182.0)  # Mid-year conditions
    
    # Get final elevation for comparison
    final_elevation = np.zeros(vertices)
    for i in range(vertices):
        final_elevation[i] = planet.get_elevation(i)
    
    # Calculate total elevation change
    elevation_change = final_elevation - initial_elevation
    
    # Visualize the results
    print("Visualizing results...")
    visualize_results(planet, climate, biome_system, erosion, elevation_change, erosion_history)
    
    print("Done!")

def visualize_results(planet, climate, biome_system, erosion, elevation_change, erosion_history):
    """Visualize the erosion system results."""
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
    
    # 1. Elevation Map
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
    ax1.set_title("Current Elevation")
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, label="Elevation (km)")
    
    # 2. Elevation Change
    ax2 = fig.add_subplot(232)
    change_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            vertex_idx = planet.get_grid().find_closest_vertex(pos)
            if vertex_idx >= 0 and vertex_idx < len(elevation_change):
                change_data[i, j] = elevation_change[vertex_idx]
    
    # Plot elevation change
    im2 = ax2.imshow(change_data, cmap='RdBu_r', extent=[-180, 180, -90, 90], origin='lower')
    ax2.set_title("Elevation Change from Erosion")
    plt.colorbar(im2, ax=ax2, label="Change (km)")
    
    # 3. Total Erosion Rate
    ax3 = fig.add_subplot(233)
    erosion_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            erosion_data[i, j] = erosion.get_erosion_rate(pos)
    
    # Plot erosion rate
    im3 = ax3.imshow(erosion_data, cmap='YlOrRd', extent=[-180, 180, -90, 90], origin='lower')
    ax3.set_title("Total Erosion Rate")
    plt.colorbar(im3, ax=ax3, label="Rate (m/My)")
    
    # 4. Fluvial Erosion Rate
    ax4 = fig.add_subplot(234)
    fluvial_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            fluvial_data[i, j] = erosion.get_process_erosion_rate(pos, _aeonterracpp.ErosionProcessType.Fluvial)
    
    # Plot fluvial erosion
    im4 = ax4.imshow(fluvial_data, cmap='Blues', extent=[-180, 180, -90, 90], origin='lower')
    ax4.set_title("Fluvial Erosion Rate")
    plt.colorbar(im4, ax=ax4, label="Rate (m/My)")
    
    # 5. Biomes
    ax5 = fig.add_subplot(235)
    biome_data = np.zeros((grid_height, grid_width), dtype=int)
    biome_colors = {}
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            biome_type = biome_system.get_biome_type(pos)
            
            # Get biome properties
            biome_id = int(biome_type)
            biome_data[i, j] = biome_id
            
            # Get biome color
            if biome_id not in biome_colors:
                color = biome_system.get_biome_color(biome_type)
                biome_colors[biome_id] = [float(color[0]), float(color[1]), float(color[2])]
    
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
    im5 = ax5.imshow(biome_data, cmap=biome_cmap, norm=norm, extent=[-180, 180, -90, 90], origin='lower')
    ax5.set_title("Biomes (After Erosion)")
    
    # 6. Erosion Over Time
    ax6 = fig.add_subplot(236)
    
    # Calculate mean erosion rates over land
    land_mask = elevation_data > 0
    land_indices = np.where(land_mask)
    
    # Process types to track
    process_labels = [
        "Fluvial", "Coastal", "Glacial", 
        "Aeolian", "Chemical", "Thermal", 
        "Mass Movement", "Biotic"
    ]
    
    # Create time points
    time_points = np.arange(1, len(erosion_history) + 1)
    
    # Sample the erosion data from a single location for simplicity
    sample_lat, sample_lon = 30, 45  # Choose a land location
    sample_idx = None
    
    # Find the closest point to our sample location
    for i in range(grid_height):
        for j in range(grid_width):
            lat = lat_grid[i, j]
            lon = lon_grid[i, j]
            if abs(lat - sample_lat) < 5 and abs(lon - sample_lon) < 5 and elevation_data[i, j] > 0:
                sample_idx = i * grid_width + j
                break
        if sample_idx is not None:
            break
    
    if sample_idx is not None:
        # Get vertex index
        pos = sample_points[sample_idx]
        vertex_idx = planet.get_grid().find_closest_vertex(pos)
        
        # Get process erosion rates for each step
        erosion_by_process = np.zeros((len(process_labels), len(erosion_history)))
        
        for step in range(len(erosion_history)):
            for proc_idx, proc_name in enumerate(process_labels):
                proc_type = getattr(_aeonterracpp.ErosionProcessType, proc_name)
                rate = erosion.get_process_erosion_rate(pos, proc_type)
                erosion_by_process[proc_idx, step] = rate
        
        # Use last step values for the pie chart
        sizes = erosion_by_process[:, -1]
        if np.sum(sizes) > 0:
            # Create pie chart of erosion processes
            ax6.pie(sizes, labels=process_labels, autopct='%1.1f%%', 
                   shadow=True, startangle=90)
            ax6.set_title(f"Erosion Processes at ({sample_lat}°, {sample_lon}°)")
        else:
            ax6.text(0.5, 0.5, "No erosion data available", 
                    horizontalalignment='center', verticalalignment='center')
    else:
        ax6.text(0.5, 0.5, "Could not find suitable land location", 
                horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.savefig("erosion_example_output.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()