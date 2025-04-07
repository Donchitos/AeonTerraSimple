#!/usr/bin/env python3
"""
Integrated Earth System Example

This script demonstrates the full integration of all earth system components:
- Climate (atmosphere, oceans, hydrology)
- Biomes (vegetation, soil properties)
- Erosion (multiple processes)
- Biome-Climate Feedback
- River-Ocean Chemistry

It shows how these systems interact and affect each other through various
feedback mechanisms, creating a more realistic planet simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import sys
import os
from collections import defaultdict
import time

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
    print("AeonTerra Integrated Earth System Example")
    print("------------------------------------------")
    
    # Create a planet with resolution level 6 (40,962 vertices) for better detail
    print("Creating planet...")
    planet = _aeonterracpp.EnhancedPlanet(radius=6371.0, resolution=6, name="IntegratedEarth")
    
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
    
    # Initialize the biome-climate feedback system
    print("Initializing biome-climate feedback system...")
    feedback = _aeonterracpp.BiomeClimateFeedback(planet, climate, biome_system)
    
    # Configure feedback parameters
    feedback_params = _aeonterracpp.BiomeClimateFeedbackParams()
    feedback_params.albedo_feedback_strength = 1.2      # Stronger albedo effects
    feedback_params.evapotranspiration_strength = 1.5   # Stronger water cycle effects
    feedback_params.carbon_sequestration_rate = 1.3     # Stronger carbon effects
    feedback.set_parameters(feedback_params)
    
    # Initialize feedback system
    feedback.initialize()
    
    # Initialize the river-ocean chemistry system
    print("Initializing river-ocean chemistry system...")
    chemistry = _aeonterracpp.RiverOceanChemistry(planet, climate, erosion)
    
    # Configure chemistry parameters
    chemistry_params = _aeonterracpp.RiverOceanChemistryParams()
    chemistry_params.river_erosion_sediment_factor = 1.2    # More sediment from erosion
    chemistry_params.ocean_primary_production_rate = 1.5    # Higher biological activity
    chemistry_params.time_scale_factor = 100.0              # Accelerate for demonstration
    chemistry.set_parameters(chemistry_params)
    
    # Initialize chemistry system
    chemistry.initialize(chemistry_params)
    
    # Track system history for visualization
    history = {
        'avg_temp': [],
        'avg_precip': [],
        'avg_vegetation': [],
        'avg_erosion': [],
        'avg_albedo': [],
        'carbon_sequestered': [],
        'sediment_to_ocean': [],
        'oxygen_level': []
    }
    
    # Run the full integrated simulation
    print("Running integrated simulation...")
    total_sim_time = 5.0  # 5 million years
    sim_steps = 10
    step_time = total_sim_time / sim_steps
    
    # Track timing
    start_time = time.time()
    
    for step in range(sim_steps):
        print(f"  Integrated step {step+1}/{sim_steps}: {step_time} million years...")
        
        # Update all systems in sequence with bidirectional feedbacks
        
        # 1. Erosion processes affect topography
        erosion.simulate_step(step_time)
        
        # 2. Track chemistry response to erosion (river sediment transport)
        chemistry.update(step_time)
        
        # 3. Climate responds to changed topography and chemistry
        climate.simulate(step_time, 182.0)  # Mid-year conditions
        
        # 4. Biomes respond to changed climate conditions
        biome_system.update(step_time)
        
        # 5. Biome-climate feedback (vegetation affects climate)
        feedback.update(step_time)
        
        # Modify systems based on feedback effects
        
        # 6. Erosion system incorporates biome effects
        erosion.update_biome_effects(biome_system)
        
        # 7. Climate system incorporates chemical effects
        chemistry.apply_chemical_effects_to_climate()
        
        # Record history for analysis
        # Get average temperature
        atmosphere = climate.get_atmosphere()
        temp_sum = 0.0
        land_count = 0
        
        # Sample some vertices for data collection
        sample_vertices = np.random.choice(vertices, 1000)
        
        for idx in sample_vertices:
            if planet.get_elevation(idx) >= 0:  # Only land areas
                pos = planet.get_grid().get_vertex(idx)
                temp_sum += atmosphere.get_temperature(pos)
                land_count += 1
        
        avg_temp = temp_sum / max(1, land_count)
        history['avg_temp'].append(avg_temp)
        
        # Get average precipitation
        hydrology = climate.get_hydrology()
        precip_sum = 0.0
        
        for idx in sample_vertices:
            if planet.get_elevation(idx) >= 0:  # Only land areas
                pos = planet.get_grid().get_vertex(idx)
                precip_sum += hydrology.get_precipitation(pos)
        
        avg_precip = precip_sum / max(1, land_count)
        history['avg_precip'].append(avg_precip)
        
        # Get average vegetation cover
        veg_sum = 0.0
        
        for idx in sample_vertices:
            if planet.get_elevation(idx) >= 0:  # Only land areas
                pos = planet.get_grid().get_vertex(idx)
                veg_sum += biome_system.get_vegetation_density(pos)
        
        avg_vegetation = veg_sum / max(1, land_count)
        history['avg_vegetation'].append(avg_vegetation)
        
        # Get average erosion rate
        erosion_sum = 0.0
        
        for idx in sample_vertices:
            if planet.get_elevation(idx) >= 0:  # Only land areas
                pos = planet.get_grid().get_vertex(idx)
                erosion_sum += erosion.get_erosion_rate(pos)
        
        avg_erosion = erosion_sum / max(1, land_count)
        history['avg_erosion'].append(avg_erosion)
        
        # Get chemical data
        sediment_input = chemistry.get_total_river_input(_aeonterracpp.ChemicalElement.Sediment)
        history['sediment_to_ocean'].append(sediment_input)
        
        oxygen_level = chemistry.get_average_ocean_concentration(_aeonterracpp.ChemicalElement.Oxygen)
        history['oxygen_level'].append(oxygen_level)
        
        # Get feedback data
        if hasattr(feedback, 'get_albedo_history'):
            if len(feedback.get_albedo_history()) > 0:
                history['avg_albedo'].append(feedback.get_albedo_history()[-1])
            else:
                history['avg_albedo'].append(0.3)  # Default albedo
                
            # Track carbon sequestration (fictional data for demo)
            carbon_seq = sum([biome_system.get_vegetation_density(planet.get_grid().get_vertex(idx)) for idx in sample_vertices if planet.get_elevation(idx) >= 0]) * 0.01
            history['carbon_sequestered'].append(carbon_seq)
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Get final elevation for comparison
    final_elevation = np.zeros(vertices)
    for i in range(vertices):
        final_elevation[i] = planet.get_elevation(i)
    
    # Calculate elevation change
    elevation_change = final_elevation - initial_elevation
    
    # Visualize the results
    print("Visualizing integrated results...")
    visualize_integrated_results(planet, climate, biome_system, erosion, 
                                chemistry, feedback, elevation_change, history)
    
    print("Done!")

def visualize_integrated_results(planet, climate, biome_system, erosion, 
                                chemistry, feedback, elevation_change, history):
    """Visualize the integrated earth system results."""
    # Create a large figure
    plt.figure(figsize=(20, 16))
    
    # Use GridSpec for more control over layout
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 0.5])
    
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
    ax1 = plt.subplot(gs[0, 0])
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
    ax1.set_title("Final Topography")
    plt.colorbar(im1, ax=ax1, label="Elevation (km)")
    
    # 2. Elevation Change
    ax2 = plt.subplot(gs[0, 1])
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
    ax2.set_title("Elevation Change")
    plt.colorbar(im2, ax=ax2, label="Change (km)")
    
    # 3. Biomes
    ax3 = plt.subplot(gs[0, 2])
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
    im3 = ax3.imshow(biome_data, cmap=biome_cmap, norm=norm, extent=[-180, 180, -90, 90], origin='lower')
    ax3.set_title("Biome Distribution")
    
    # 4. Erosion Rate
    ax4 = plt.subplot(gs[1, 0])
    erosion_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            erosion_data[i, j] = erosion.get_erosion_rate(pos)
    
    # Plot erosion rate
    im4 = ax4.imshow(erosion_data, cmap='YlOrRd', extent=[-180, 180, -90, 90], origin='lower')
    ax4.set_title("Erosion Rate")
    plt.colorbar(im4, ax=ax4, label="Rate (m/My)")
    
    # 5. Vegetation Cover
    ax5 = plt.subplot(gs[1, 1])
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
    
    # 6. Ocean Chemistry - Sediment
    ax6 = plt.subplot(gs[1, 2])
    sediment_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            # Only plot for ocean areas
            if planet.get_elevation(planet.get_grid().find_closest_vertex(pos)) < 0:
                sediment_data[i, j] = chemistry.get_chemical_concentration(
                    pos, _aeonterracpp.ChemicalElement.Sediment)
            else:
                sediment_data[i, j] = np.nan
    
    # Plot sediment
    im6 = ax6.imshow(sediment_data, cmap='YlOrBr', extent=[-180, 180, -90, 90], origin='lower')
    ax6.set_title("Ocean Sediment Concentration")
    plt.colorbar(im6, ax=ax6, label="mg/L")
    
    # 7. River Flow
    ax7 = plt.subplot(gs[2, 0])
    flow_data = np.zeros((grid_height, grid_width))
    hydrology = climate.get_hydrology()
    
    # Get river flow data
    river_flow = hydrology.get_river_flow_data()
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            vertex_idx = planet.get_grid().find_closest_vertex(pos)
            
            if vertex_idx >= 0 and vertex_idx < len(river_flow):
                flow_data[i, j] = min(100, river_flow[vertex_idx])  # Cap for visualization
    
    # Plot river flow
    im7 = ax7.imshow(flow_data, cmap='Blues', extent=[-180, 180, -90, 90], origin='lower')
    ax7.set_title("River Flow")
    plt.colorbar(im7, ax=ax7, label="Flow (m³/s)")
    
    # 8. Precipitation
    ax8 = plt.subplot(gs[2, 1])
    precip_data = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            precip_data[i, j] = hydrology.get_precipitation(pos)
    
    # Plot precipitation
    im8 = ax8.imshow(precip_data, cmap='Blues', extent=[-180, 180, -90, 90], origin='lower')
    ax8.set_title("Precipitation")
    plt.colorbar(im8, ax=ax8, label="mm/year")
    
    # 9. Temperature
    ax9 = plt.subplot(gs[2, 2])
    temp_data = np.zeros((grid_height, grid_width))
    atmosphere = climate.get_atmosphere()
    
    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j
            pos = sample_points[idx]
            temp_data[i, j] = atmosphere.get_temperature(pos)
    
    # Plot temperature
    im9 = ax9.imshow(temp_data, cmap='RdYlBu_r', extent=[-180, 180, -90, 90], origin='lower')
    ax9.set_title("Temperature")
    plt.colorbar(im9, ax=ax9, label="°C")
    
    # 10. Time Series Data
    ax10 = plt.subplot(gs[3, :])
    steps = np.arange(len(history['avg_temp']))
    
    # Plot multiple time series on same axis
    ax10.plot(steps, history['avg_temp'], 'r-', label='Temperature (°C)')
    ax10.plot(steps, history['avg_vegetation'], 'g-', label='Vegetation (0-1)')
    ax10.plot(steps, history['avg_erosion'] / 10, 'y-', label='Erosion ÷ 10')
    ax10.plot(steps, history['sediment_to_ocean'] / 1000, 'b-', label='Sediment ÷ 1000')
    
    # Add labels
    ax10.set_title("System Evolution Over Time")
    ax10.set_xlabel("Simulation Steps")
    ax10.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("integrated_earth_output.png", dpi=150)
    print("Visualization saved to integrated_earth_output.png")
    plt.show()

if __name__ == "__main__":
    main()