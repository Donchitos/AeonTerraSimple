#!/usr/bin/env python3
"""
Example script for using the AeonTerra C++ TectonicSimulation implementation from Python.
This demonstrates creating a planet, simulating tectonic activity, and visualizing the results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys

# Import our C++ module
sys.path.append(os.path.join(os.path.dirname(__file__), 'test_build'))
from _aeonterracpp import (
    Planet, TectonicSimulation, PlateInteractionType, 
    BoundingBox, Vec3f, to_radians, to_degrees
)

def visualize_planet_properties(planet, property_func, title="Planet Visualization", cmap=cm.viridis):
    """Visualize a planet property in 3D."""
    # Get the grid from the planet
    grid = planet.get_grid()
    vertex_count = grid.get_vertex_count()
    
    # Extract vertex positions and property data
    positions = []
    values = []
    
    for i in range(vertex_count):
        pos = grid.get_vertex(i)
        positions.append((pos.x, pos.y, pos.z))
        values.append(property_func(i))
    
    # Convert to NumPy arrays
    positions = np.array(positions)
    values = np.array(values)
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap based on the values
    colors = cmap(plt.Normalize()(values))
    
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
        f"Value range: {min(values):.2f} to {max(values):.2f}"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace')
    
    # Add a colorbar
    norm = plt.Normalize(min(values), max(values))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1)
    
    return fig

def visualize_plate_boundaries(planet, sim, title="Plate Boundaries"):
    """Visualize plate boundaries on a 3D sphere."""
    # Get the grid from the planet
    grid = planet.get_grid()
    vertex_count = grid.get_vertex_count()
    
    # Get boundary vertices
    boundary_vertices = sim.get_boundary_vertices()
    boundary_set = set(boundary_vertices)
    
    # Extract vertex positions
    positions = []
    colors = []
    sizes = []
    
    for i in range(vertex_count):
        pos = grid.get_vertex(i)
        positions.append((pos.x, pos.y, pos.z))
        
        # Color and size based on whether it's a boundary vertex
        if i in boundary_set:
            colors.append((1.0, 0.0, 0.0))  # Red for boundaries
            sizes.append(20)
        else:
            plate_id = planet.get_plate_id(i)
            if plate_id >= 0:
                # Use different colors for different plates
                plate_color = plt.cm.tab10(plate_id % 10)
                colors.append(plate_color)
            else:
                colors.append((0.7, 0.7, 0.7))  # Gray for unassigned
            sizes.append(5)
    
    # Convert to NumPy arrays
    positions = np.array(positions)
    colors = np.array(colors)
    sizes = np.array(sizes)
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=colors, s=sizes, alpha=0.8
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
        f"Plates: {len(sim.get_plates())}\n"
        f"Boundary vertices: {len(boundary_vertices)}"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace')
    
    return fig

def main():
    print("AeonTerra Tectonics Example")
    print("===========================")
    
    # Create a new planet (radius=6371 km, resolution=4)
    print("\nCreating new planet...")
    planet = Planet(6371.0, 4, "Tectonic Test Planet")
    print(f"Planet created: {planet.get_name()}")
    print(f"  Resolution: {planet.get_grid().get_resolution()}")
    print(f"  Vertex count: {planet.get_vertex_count()}")
    
    # Create tectonic simulation with 7 plates
    print("\nInitializing tectonic simulation...")
    sim = TectonicSimulation(planet, 7, 42)  # 7 plates, seed=42 for reproducibility
    
    # Visualize initial plate configuration
    print("\nVisualizing initial plate configuration...")
    fig1 = visualize_plate_boundaries(planet, sim, "Initial Plate Configuration")
    fig1.savefig("tectonics_initial_plates.png")
    
    # Visualize initial elevation
    print("\nVisualizing initial elevation...")
    fig2 = visualize_planet_properties(
        planet, 
        lambda i: planet.get_elevation(i), 
        "Initial Elevation (km)",
        cm.terrain
    )
    fig2.savefig("tectonics_initial_elevation.png")
    
    # Simulate tectonic activity
    print("\nSimulating tectonic activity...")
    sim.set_tectonic_activity(0.8)  # Higher activity level
    
    # Run simulation for 50 million years in 5 steps
    for i in range(5):
        print(f"  Step {i+1}: +10 million years")
        sim.simulate_step(10.0)
    
    print(f"Simulation complete. Planet age: {planet.get_age()} million years")
    
    # Visualize final plate configuration
    print("\nVisualizing final plate configuration...")
    fig3 = visualize_plate_boundaries(planet, sim, "Final Plate Configuration")
    fig3.savefig("tectonics_final_plates.png")
    
    # Visualize final elevation
    print("\nVisualizing final elevation...")
    fig4 = visualize_planet_properties(
        planet, 
        lambda i: planet.get_elevation(i), 
        "Final Elevation After 50 Million Years (km)",
        cm.terrain
    )
    fig4.savefig("tectonics_final_elevation.png")
    
    # Visualize crust thickness
    print("\nVisualizing crust thickness...")
    fig5 = visualize_planet_properties(
        planet, 
        lambda i: planet.get_crust_thickness(i), 
        "Crust Thickness (km)",
        cm.magma
    )
    fig5.savefig("tectonics_crust_thickness.png")
    
    # Show plate statistics
    print("\nPlate statistics:")
    plates = sim.get_plates()
    print(f"  Number of plates: {len(plates)}")
    
    for i, plate in enumerate(plates):
        vertices = plate.get_vertices()
        print(f"  Plate {i}:")
        print(f"    ID: {plate.get_id()}")
        print(f"    Type: {'Oceanic' if plate.is_oceanic() else 'Continental'}")
        print(f"    Vertices: {len(vertices)}")
        print(f"    Area: {plate.calculate_area():.2f} units")
        
        # Movement vector
        movement = plate.get_movement()
        print(f"    Movement: ({movement.x:.4f}, {movement.y:.4f}, {movement.z:.4f})")
    
    print("\nExample completed successfully!")
    print("Generated visualization files:")
    print("  tectonics_initial_plates.png")
    print("  tectonics_initial_elevation.png")
    print("  tectonics_final_plates.png")
    print("  tectonics_final_elevation.png")
    print("  tectonics_crust_thickness.png")
    
    # Show the visualization (comment this out in non-interactive environments)
    # plt.show()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())