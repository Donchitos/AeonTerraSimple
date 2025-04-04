#!/usr/bin/env python3
"""
Example script for using the AeonTerra C++ HeightmapGenerator from Python.
This demonstrates generating heightmaps from a planet and visualizing them.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Import our C++ module
sys.path.append(os.path.join(os.path.dirname(__file__), 'test_build'))
from _aeonterracpp import (
    Planet, TectonicSimulation, HeightmapGenerator, 
    HeightmapConfigFactory, BoundingBox
)

def visualize_heightmap(heightmap, title="Heightmap Visualization", cmap=cm.terrain):
    """Visualize a heightmap using matplotlib."""
    # Get heightmap data as numpy array
    data = heightmap.data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the heightmap
    img = ax.imshow(data, cmap=cmap, extent=[
        heightmap.region.lon_min, heightmap.region.lon_max,
        heightmap.region.lat_max, heightmap.region.lat_min  # Note: inverted y-axis
    ])
    
    # Add a colorbar
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Elevation (km)')
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add info text
    info_text = (
        f"Size: {heightmap.width}x{heightmap.height}\n"
        f"Region: {heightmap.region.lat_min:.1f}° to {heightmap.region.lat_max:.1f}° lat, "
        f"{heightmap.region.lon_min:.1f}° to {heightmap.region.lon_max:.1f}° lon\n"
        f"Elevation range: {heightmap.min_value:.2f} to {heightmap.max_value:.2f} km"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    return fig

def main():
    print("AeonTerra Heightmap Generator Example")
    print("=====================================")
    
    # Create a new planet
    print("\nCreating new planet...")
    planet = Planet(6371.0, 5, "Heightmap Test Planet")
    print(f"Planet created: {planet.get_name()}")
    print(f"  Resolution: {planet.get_grid().get_resolution()}")
    print(f"  Vertex count: {planet.get_vertex_count()}")
    
    # Create tectonic simulation to generate realistic terrain
    print("\nInitializing tectonic simulation...")
    sim = TectonicSimulation(planet, 12, 42)  # 12 plates, seed=42
    
    # Simulate tectonic activity to create interesting terrain
    print("\nSimulating tectonic activity...")
    sim.set_tectonic_activity(0.8)  # Higher activity level
    
    # Run simulation for 50 million years in 5 steps
    for i in range(5):
        print(f"  Step {i+1}: +10 million years")
        sim.simulate_step(10.0)
    
    print(f"Simulation complete. Planet age: {planet.get_age()} million years")
    
    # Create heightmap generator
    print("\nCreating heightmap generator...")
    generator = HeightmapGenerator()
    
    # Generate a global heightmap
    print("\nGenerating global heightmap...")
    start_time = time.time()
    global_config = HeightmapConfigFactory.create_global_config(1024, 512)
    global_heightmap = generator.generate_heightmap(planet, global_config)
    elapsed = time.time() - start_time
    print(f"  Generated {global_heightmap.width}x{global_heightmap.height} global heightmap in {elapsed:.2f} seconds")
    
    # Visualize and save global heightmap
    fig1 = visualize_heightmap(global_heightmap, "Global Heightmap")
    fig1.savefig("global_heightmap.png", dpi=150)
    print("  Saved global heightmap to global_heightmap.png")
    
    # Generate a regional heightmap with enhanced detail
    print("\nGenerating regional heightmap...")
    start_time = time.time()
    
    # Find an interesting region (mountain range)
    region = BoundingBox(20.0, -30.0, 40.0, 10.0)  # Example region
    
    # Create a regional config with high detail
    regional_config = HeightmapConfigFactory.create_detailed_terrain_config(region, 16)
    regional_config.enhance_detail = True
    regional_config.roughness = 0.7
    regional_config.octaves = 6
    regional_config.detail_scale = 0.3
    
    regional_heightmap = generator.generate_heightmap(planet, regional_config)
    elapsed = time.time() - start_time
    print(f"  Generated {regional_heightmap.width}x{regional_heightmap.height} regional heightmap in {elapsed:.2f} seconds")
    
    # Visualize and save regional heightmap
    fig2 = visualize_heightmap(regional_heightmap, "Regional Heightmap (Enhanced Detail)")
    fig2.savefig("regional_heightmap.png", dpi=150)
    print("  Saved regional heightmap to regional_heightmap.png")
    
    # Export heightmap data to files
    print("\nExporting heightmap data...")
    generator.export_raw(global_heightmap, "global_heightmap.raw")
    generator.export_raw(regional_heightmap, "regional_heightmap.raw")
    print("  Exported heightmap data to .raw files")
    
    # Generate heightmaps with different detail levels
    print("\nGenerating heightmaps with varying detail levels...")
    detail_levels = [0.0, 0.3, 0.7]
    fig3, axes = plt.subplots(1, len(detail_levels), figsize=(15, 5))
    
    # Same region, different detail levels
    for i, detail in enumerate(detail_levels):
        config = HeightmapConfigFactory.create_regional_config(30.0, -10.0, 10.0, 10.0, 10)
        config.enhance_detail = True
        config.roughness = 0.6
        config.octaves = 6
        config.detail_scale = detail
        
        heightmap = generator.generate_heightmap(planet, config)
        
        # Plot in the appropriate subplot
        ax = axes[i]
        img = ax.imshow(heightmap.data, cmap=cm.terrain)
        ax.set_title(f"Detail Scale: {detail}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    fig3.savefig("detail_comparison.png", dpi=150)
    print("  Saved detail comparison to detail_comparison.png")
    
    print("\nExample completed successfully!")
    print("Generated visualization files:")
    print("  global_heightmap.png")
    print("  regional_heightmap.png")
    print("  detail_comparison.png")
    
    # Show the visualizations (comment this out in non-interactive environments)
    # plt.show()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())