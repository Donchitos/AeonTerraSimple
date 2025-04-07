#!/usr/bin/env python3
"""
Optimized Heightmap Generation Example

This script demonstrates the use of the performance optimization components 
to generate planet heightmaps with improved speed and memory efficiency.
It showcases spatial acceleration, parallel processing, memory optimization,
and the configuration system for parameter-driven terrain customization.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AeonTerra C++ bindings
try:
    from planet_sim.cpp.python import _aeonterracpp
except ImportError:
    print("Failed to import AeonTerra C++ bindings. Make sure they're built properly.")
    print("Try running: cd planet_sim/cpp && ./optimized_config_build.sh")
    sys.exit(1)

def main():
    print("AeonTerra Optimized Heightmap Generation")
    print("----------------------------------------")
    
    # Setup the configuration system
    config = _aeonterracpp.optimization.ConfigManager.get_instance()
    
    # Create heightmap section
    heightmap_section = config.create_section("heightmap", "Heightmap Generation Parameters")
    
    # Add parameters
    heightmap_section.add_param("resolution", _aeonterracpp.optimization.ParamValue(1024), 
                               create_metadata("resolution", "Resolution of the heightmap", _aeonterracpp.optimization.ParamType.Int))
    
    heightmap_section.add_param("seed", _aeonterracpp.optimization.ParamValue(42),
                               create_metadata("seed", "Random seed", _aeonterracpp.optimization.ParamType.Int))
    
    heightmap_section.add_param("noise_octaves", _aeonterracpp.optimization.ParamValue(6),
                               create_metadata("noise_octaves", "Number of noise octaves", _aeonterracpp.optimization.ParamType.Int))
    
    heightmap_section.add_param("noise_persistence", _aeonterracpp.optimization.ParamValue(0.5),
                               create_metadata("noise_persistence", "Noise persistence", _aeonterracpp.optimization.ParamType.Float))
    
    heightmap_section.add_param("mountain_height", _aeonterracpp.optimization.ParamValue(0.8),
                               create_metadata("mountain_height", "Mountain height factor", _aeonterracpp.optimization.ParamType.Float))
    
    # Create erosion section
    erosion_section = config.create_section("erosion", "Erosion Parameters")
    
    # Add erosion parameters
    erosion_section.add_param("erosion_cycles", _aeonterracpp.optimization.ParamValue(250),
                             create_metadata("erosion_cycles", "Number of erosion cycles", _aeonterracpp.optimization.ParamType.Int))
    
    erosion_section.add_param("erosion_strength", _aeonterracpp.optimization.ParamValue(0.15),
                             create_metadata("erosion_strength", "Erosion strength factor", _aeonterracpp.optimization.ParamType.Float))
    
    # Create performance section
    perf_section = config.create_section("performance", "Performance Parameters")
    
    # Add performance parameters
    perf_section.add_param("use_spatial_acceleration", _aeonterracpp.optimization.ParamValue(True),
                          create_metadata("use_spatial_acceleration", "Use spatial acceleration", _aeonterracpp.optimization.ParamType.Bool))
    
    perf_section.add_param("multithreading", _aeonterracpp.optimization.ParamValue(True),
                          create_metadata("multithreading", "Enable multithreading", _aeonterracpp.optimization.ParamType.Bool))
    
    perf_section.add_param("thread_count", _aeonterracpp.optimization.ParamValue(0),
                          create_metadata("thread_count", "Thread count (0=auto)", _aeonterracpp.optimization.ParamType.Int))
    
    # Set memory management parameters
    memory_manager = _aeonterracpp.optimization.MemoryManager.get_instance()
    memory_config = _aeonterracpp.optimization.MemoryConfig()
    memory_config.strategy = _aeonterracpp.optimization.MemoryStrategy.PoolAllocator
    memory_config.pool_size = 1024 * 1024 * 10  # 10MB pool size
    memory_manager.configure(memory_config)
    
    # Create a planet with configured parameters
    print("Creating planet...")
    start_time = time.time()
    
    # Get heightmap parameters
    resolution = heightmap_section.get_param("resolution").getValue(0)
    seed = heightmap_section.get_param("seed").getValue(0)
    noise_octaves = heightmap_section.get_param("noise_octaves").getValue(0)
    noise_persistence = heightmap_section.get_param("noise_persistence").getValue(0.0)
    mountain_height = heightmap_section.get_param("mountain_height").getValue(0.0)
    
    # Create a planet with specific resolution
    resolution_level = 6  # ~40,962 vertices
    planet = _aeonterracpp.Planet(6371.0, resolution_level, "OptimizedPlanet")
    
    # Create and configure a heightmap generator
    print("Configuring heightmap generator...")
    heightmap_gen = _aeonterracpp.HeightmapGenerator(planet)
    heightmap_gen.set_random_seed(seed)
    heightmap_gen.set_noise_octaves(noise_octaves)
    heightmap_gen.set_noise_persistence(noise_persistence)
    heightmap_gen.set_mountain_height(mountain_height)
    
    # Use spatial acceleration if enabled
    if perf_section.get_param("use_spatial_acceleration").getValue(False):
        print("Setting up spatial acceleration...")
        accel_params = _aeonterracpp.optimization.SpatialAccelerationParams()
        accel_params.type = _aeonterracpp.optimization.SpatialAccelerationType.Octree
        accel_params.max_depth = 8
        accel_params.grid_resolution = 1.0
        accel_params.parallel_construction = perf_section.get_param("multithreading").getValue(False)
        
        # Create spatial acceleration
        acceleration = _aeonterracpp.optimization.SpatialAcceleration.create(
            _aeonterracpp.optimization.SpatialAccelerationType.Octree,
            planet, accel_params)
        
        # Set it in the heightmap generator
        heightmap_gen.use_spatial_acceleration(acceleration)
    
    # Set up parallel processing
    if perf_section.get_param("multithreading").getValue(False):
        parallel_config = _aeonterracpp.optimization.ParallelConfig()
        parallel_config.num_threads = perf_section.get_param("thread_count").getValue(0)
        parallel_config.enable_dynamic_load = True
        
        # Enable parallel processing in the heightmap generator
        heightmap_gen.enable_parallel_processing(parallel_config)
    
    # Generate heightmap
    print("Generating heightmap...")
    heightmap_gen_time_start = time.time()
    heightmap_gen.generate_heightmap()
    heightmap_gen_time = time.time() - heightmap_gen_time_start
    
    # Get erosion parameters
    erosion_cycles = erosion_section.get_param("erosion_cycles").getValue(0)
    erosion_strength = erosion_section.get_param("erosion_strength").getValue(0.0)
    
    # Apply erosion
    if erosion_cycles > 0:
        print(f"Applying erosion ({erosion_cycles} cycles)...")
        erosion_time_start = time.time()
        
        # Set up erosion
        erosion = _aeonterracpp.EnhancedErosion(planet)
        erosion.set_erosion_strength(erosion_strength)
        
        # Use parallel processing if enabled
        if perf_section.get_param("multithreading").getValue(False):
            erosion.enable_parallel_processing(parallel_config)
        
        # Apply erosion
        erosion.apply_erosion(erosion_cycles)
        
        erosion_time = time.time() - erosion_time_start
    else:
        erosion_time = 0
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Display performance metrics
    print("\nPerformance Metrics:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"  - Heightmap generation: {heightmap_gen_time:.2f} seconds")
    if erosion_cycles > 0:
        print(f"  - Erosion: {erosion_time:.2f} seconds")
    
    # Display memory metrics
    memory_stats = memory_manager.get_stats()
    print("\nMemory Usage:")
    print(f"Peak memory: {_aeonterracpp.optimization.MemoryStats.format_size(memory_stats.peak_usage)}")
    print(f"Current memory: {_aeonterracpp.optimization.MemoryStats.format_size(memory_stats.current_usage)}")
    print(f"Allocation count: {memory_stats.allocation_count}")
    
    # Display terrain properties
    print("\nTerrain Properties:")
    print(f"Number of vertices: {planet.get_vertex_count()}")
    print(f"Planet radius: {planet.get_radius()} km")
    
    # Visualize the results
    print("\nGenerating visualization...")
    visualize_planet(planet)
    
    # Save configuration to file
    config.save_to_file("optimized_heightmap_config.json")
    print("\nConfiguration saved to optimized_heightmap_config.json")
    
    print("\nDone!")

def create_metadata(name, description, param_type):
    """Create parameter metadata for the configuration system"""
    metadata = _aeonterracpp.optimization.ParamMetadata()
    metadata.name = name
    metadata.description = description
    metadata.type = param_type
    return metadata

def visualize_planet(planet):
    """Create a visualization of the planet's heightmap"""
    # Create a grid for sampling
    resolution = 500
    lat = np.linspace(-90, 90, resolution)
    lon = np.linspace(-180, 180, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Sample heightmap
    heightmap = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            # Convert lat/lon to 3D position
            lat_rad = np.radians(lat_grid[i, j])
            lon_rad = np.radians(lon_grid[i, j])
            
            # Create position vector
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)
            
            # Create normalized position vector
            pos = _aeonterracpp.Vec3f(x, y, z)
            
            # Scale to planet radius
            pos = pos * planet.get_radius()
            
            # Get elevation at position
            elevation = planet.get_elevation_at_position(pos)
            heightmap[i, j] = elevation
    
    # Create custom colormap that transitions from blue to green to brown to white
    cmap = plt.cm.terrain
    
    # Plot heightmap
    plt.figure(figsize=(12, 8))
    plt.imshow(heightmap, cmap=cmap, extent=[-180, 180, -90, 90])
    plt.colorbar(label="Elevation (km)")
    plt.title("Optimized Planet Heightmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("optimized_heightmap.png", dpi=150)
    print("Visualization saved to optimized_heightmap.png")
    plt.show()

if __name__ == "__main__":
    main()