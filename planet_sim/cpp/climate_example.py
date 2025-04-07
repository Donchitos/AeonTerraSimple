#!/usr/bin/env python3
"""
Example demonstrating the integrated climate system in AeonTerra.

This example:
1. Creates a planet with tectonic plates
2. Simulates tectonic processes to generate a basic landscape
3. Initializes and runs the climate system
4. Visualizes climate patterns including:
   - Temperature distribution
   - Wind patterns
   - Ocean currents
   - Precipitation
   - Climate zones
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.patches import Patch
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
    print("AeonTerra Climate System Example")
    print("--------------------------------")
    
    # Create a planet with resolution level 6 (40,962 vertices) for better detail
    print("Creating planet...")
    planet = _aeonterracpp.EnhancedPlanet(radius=6371.0, resolution=6, name="EarthClimate")
    
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
    # This is much faster than tectonic simulation
    print("Running climate simulation...")
    climate_steps = 12  # One step per month
    for i in range(climate_steps):
        day_of_year = i * (365.24 / climate_steps)
        print(f"  Simulating day {day_of_year:.1f}...")
        climate.simulate(0.1, day_of_year)  # 0.1 million years per step
    
    # Visualize the results
    print("Visualizing results...")
    visualize_results(planet, tectonics, climate)
    
    print("Done!")

def visualize_results(planet, tectonics, climate):
    """Visualize the climate system results."""
    # Get components
    atmosphere = climate.get_atmosphere()
    ocean = climate.get_ocean()
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
    
    # 1. Elevation and Plates visualization
    ax1 = fig.add_subplot(231)
    elevation_data = np.zeros((grid_height, grid_width))
    plate_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            vertex_idx = planet.get_grid().find_closest_vertex(pos)
            if vertex_idx >= 0:
                elevation_data[i, j] = planet.get_elevation(vertex_idx)
                plate_data[i, j] = planet.get_plate_id(vertex_idx)
    
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
    ax1.set_title("Elevation and Plate Boundaries")
    
    # Overlay plate boundaries
    plate_edges = np.zeros_like(plate_data, dtype=bool)
    for i in range(1, grid_height-1):
        for j in range(1, grid_width-1):
            current = plate_data[i, j]
            if (current != plate_data[i-1, j] or current != plate_data[i+1, j] or
                current != plate_data[i, j-1] or current != plate_data[i, j+1]):
                plate_edges[i, j] = True
    
    plate_edges_masked = np.ma.masked_where(~plate_edges, plate_edges)
    ax1.imshow(plate_edges_masked, cmap='binary', alpha=0.5, extent=[-180, 180, -90, 90], origin='lower')
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, label="Elevation (km)")
    
    # 2. Temperature and Wind patterns
    ax2 = fig.add_subplot(232)
    temperature_data = np.zeros((grid_height, grid_width))
    wind_u = np.zeros((grid_height, grid_width))
    wind_v = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            temperature_data[i, j] = atmosphere.get_temperature(pos)
            
            # Get wind vector
            wind_vec = atmosphere.get_wind_vector(pos)
            # Project wind vector to lat/lon space for plotting
            # This is a simplified projection that works for visualization
            lat_rad = np.radians(lat_grid[i, j])
            lon_rad = np.radians(lon_grid[i, j])
            
            # Create local coordinate system
            east = _aeonterracpp.Vec3f(-np.sin(lon_rad), np.cos(lon_rad), 0)
            north = _aeonterracpp.Vec3f(-np.sin(lat_rad)*np.cos(lon_rad), 
                                       -np.sin(lat_rad)*np.sin(lon_rad), 
                                       np.cos(lat_rad))
            
            # Project wind vector into east-north components
            wind_u[i, j] = wind_vec.dot(east)
            wind_v[i, j] = wind_vec.dot(north)
    
    # Plot temperature
    im2 = ax2.imshow(temperature_data, cmap='RdBu_r', extent=[-180, 180, -90, 90], origin='lower')
    ax2.set_title("Temperature and Wind")
    plt.colorbar(im2, ax=ax2, label="Temperature (°C)")
    
    # Plot wind vectors (subsample for clarity)
    skip = 8
    ax2.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], 
               wind_u[::skip, ::skip], wind_v[::skip, ::skip],
               scale=50, color='black', alpha=0.7)
    
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
    
    # 4. Ocean currents and SST
    ax4 = fig.add_subplot(234)
    sst_data = np.zeros((grid_height, grid_width))
    current_u = np.zeros((grid_height, grid_width))
    current_v = np.zeros((grid_height, grid_width))
    ocean_mask = np.zeros((grid_height, grid_width), dtype=bool)
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            vertex_idx = planet.get_grid().find_closest_vertex(pos)
            
            if vertex_idx >= 0 and planet.get_elevation(vertex_idx) < 0:
                ocean_mask[i, j] = True
                sst_data[i, j] = ocean.get_sea_surface_temperature(pos)
                
                # Get current vector
                current_vec = ocean.get_current_vector(pos)
                
                # Project current vector to lat/lon space
                lat_rad = np.radians(lat_grid[i, j])
                lon_rad = np.radians(lon_grid[i, j])
                
                # Create local coordinate system
                east = _aeonterracpp.Vec3f(-np.sin(lon_rad), np.cos(lon_rad), 0)
                north = _aeonterracpp.Vec3f(-np.sin(lat_rad)*np.cos(lon_rad), 
                                         -np.sin(lat_rad)*np.sin(lon_rad), 
                                         np.cos(lat_rad))
                
                # Project current vector into east-north components
                current_u[i, j] = current_vec.dot(east)
                current_v[i, j] = current_vec.dot(north)
            else:
                sst_data[i, j] = np.nan  # Mask land areas
    
    # Plot sea surface temperature
    im4 = ax4.imshow(sst_data, cmap='turbo', extent=[-180, 180, -90, 90], origin='lower')
    ax4.set_title("Sea Surface Temperature and Currents")
    plt.colorbar(im4, ax=ax4, label="Temperature (°C)")
    
    # Plot ocean currents (subsample for clarity)
    skip = 8
    current_u_masked = np.ma.masked_where(~ocean_mask, current_u)
    current_v_masked = np.ma.masked_where(~ocean_mask, current_v)
    ax4.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip], 
               current_u_masked[::skip, ::skip], current_v_masked[::skip, ::skip],
               scale=30, color='white', alpha=0.7)
    
    # 5. River systems
    ax5 = fig.add_subplot(235)
    river_data = np.zeros((grid_height, grid_width))
    land_mask = np.zeros((grid_height, grid_width), dtype=bool)
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            vertex_idx = planet.get_grid().find_closest_vertex(pos)
            
            if vertex_idx >= 0:
                if planet.get_elevation(vertex_idx) >= 0:
                    land_mask[i, j] = True
                    river_data[i, j] = hydrology.get_river_flow(pos)
                else:
                    river_data[i, j] = np.nan
    
    # Plot elevation for land areas
    im5 = ax5.imshow(elevation_data, cmap='terrain', extent=[-180, 180, -90, 90], origin='lower')
    
    # Plot rivers with log scale for better visualization
    river_data_log = np.log1p(river_data)  # Log scale for river flow (log(1+x))
    river_data_masked = np.ma.masked_where(river_data < 10, river_data_log)  # Only show significant rivers
    ax5.imshow(river_data_masked, cmap='Blues', alpha=0.7, extent=[-180, 180, -90, 90], origin='lower')
    
    ax5.set_title("River Systems")
    plt.colorbar(im5, ax=ax5, label="Elevation (km)")
    
    # 6. Climate zones (Köppen classification)
    ax6 = fig.add_subplot(236)
    
    # Get climate zones data
    climate_zones = np.zeros((grid_height, grid_width), dtype=int)
    
    # Sample from climate model
    # In a real implementation, climate zones would be stored in the climate system
    # Here we're using a simplified approach based on temperature and precipitation
    
    # Basic Köppen climate classification
    # A: Tropical (red)
    # B: Arid (yellow)
    # C: Temperate (green)
    # D: Continental (blue)
    # E: Polar (purple)
    
    for i in range(grid_height):
        for j in range(grid_width):
            lat = lat_grid[i, j]
            temp = temperature_data[i, j]
            precip = precip_data[i, j]
            elev = elevation_data[i, j]
            
            if elev < 0:
                climate_zones[i, j] = 0  # Ocean
            elif temp > 18:
                climate_zones[i, j] = 1  # A: Tropical
            elif precip < 500:
                climate_zones[i, j] = 2  # B: Arid
            elif temp > 0 and temp < 18:
                climate_zones[i, j] = 3  # C: Temperate
            elif temp < 0 and temp > -10:
                climate_zones[i, j] = 4  # D: Continental
            else:
                climate_zones[i, j] = 5  # E: Polar
    
    # Define colors for climate zones
    cmap_climate = colors.ListedColormap(['#77AADD', '#DD4444', '#DDDD44', '#44DD44', '#4444DD', '#AA44DD'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = colors.BoundaryNorm(bounds, cmap_climate.N)
    
    im6 = ax6.imshow(climate_zones, cmap=cmap_climate, norm=norm, extent=[-180, 180, -90, 90], origin='lower')
    ax6.set_title("Climate Zones (Köppen Classification)")
    
    # Add legend for climate zones
    legend_elements = [
        Patch(facecolor='#77AADD', label='Ocean'),
        Patch(facecolor='#DD4444', label='A: Tropical'),
        Patch(facecolor='#DDDD44', label='B: Arid'),
        Patch(facecolor='#44DD44', label='C: Temperate'),
        Patch(facecolor='#4444DD', label='D: Continental'),
        Patch(facecolor='#AA44DD', label='E: Polar')
    ]
    ax6.legend(handles=legend_elements, loc='lower left')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()