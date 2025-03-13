#!/usr/bin/env python3
"""
Enhanced AeonTerra Procedural Planet Generator
Main test script with improved scientific complexity and visualization capabilities
"""

from planet_sim.core.planet import Planet
from planet_sim.core.tectonics import TectonicSimulation
from planet_sim.core.climate import ClimateSimulation
from planet_sim.core.erosion import ErosionSimulation
from planet_sim.core.biome import BiomeClassifier
from planet_sim.utils.export import ExportManager
import numpy as np
import os
import argparse
import json
import time
from datetime import datetime

def parse_arguments():
    """Parse command line arguments for simulation control"""
    parser = argparse.ArgumentParser(description='AeonTerra Planet Generator')
    
    # Basic simulation parameters
    parser.add_argument('--resolution', type=int, default=4, 
                        help='Planet resolution level (3-6, higher values require more memory)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducible planets')
    parser.add_argument('--steps', type=int, default=5, 
                        help='Number of tectonic simulation steps')
    parser.add_argument('--step-size', type=float, default=100.0, 
                        help='Size of each step in million years')
    
    # Planet customization
    parser.add_argument('--radius', type=float, default=6371.0, 
                        help='Planet radius in kilometers')
    parser.add_argument('--num-plates', type=int, default=8, 
                        help='Number of tectonic plates')
    parser.add_argument('--initial-age', type=float, default=4500, 
                        help='Initial age of the planet in million years')
    parser.add_argument('--tectonic-activity', type=float, default=1.0, 
                        help='Level of tectonic activity (0.5-2.0)')
    parser.add_argument('--continental-fraction', type=float, default=0.3, 
                        help='Target fraction of continental crust (0.1-0.9)')
    
    # Visualization options
    parser.add_argument('--vis-mode', type=str, default='3d', choices=['2d', '3d', 'both'],
                        help='Visualization mode: 2d, 3d, or both')
    parser.add_argument('--projection', type=str, default='equirectangular', 
                        choices=['equirectangular', 'mollweide', 'orthographic'],
                        help='Map projection for 2D visualization')
    parser.add_argument('--show-features', action='store_true', 
                        help='Show geological features in visualizations')
    parser.add_argument('--output-dir', type=str, default='output', 
                        help='Directory for output files')
    
    # Export options
    parser.add_argument('--export-heightmap', action='store_true', 
                        help='Export heightmap files')
    parser.add_argument('--export-climate', action='store_true', 
                        help='Export climate data maps')
    parser.add_argument('--export-all', action='store_true', 
                        help='Export all data formats')
    parser.add_argument('--region', type=str, default=None, 
                        help='Export specific region (format: lat1,lon1,lat2,lon2)')
    
    # Simulation components
    parser.add_argument('--run-climate', action='store_true', 
                        help='Run climate simulation')
    parser.add_argument('--run-erosion', action='store_true', 
                        help='Run erosion simulation')
    parser.add_argument('--run-biomes', action='store_true', 
                        help='Generate biomes')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the simulation environment"""
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up session subfolder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"planet_{timestamp}")
    os.makedirs(session_dir)
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    return session_dir

def create_planet(args):
    """Create and initialize the planet"""
    print(f"Creating planet with resolution {args.resolution}...")
    planet = Planet(radius=args.radius, resolution=args.resolution)
    
    # Initialize tectonic simulation
    print("Initializing tectonic simulation...")
    tectonics = TectonicSimulation(
        planet,
        num_plates=args.num_plates,
        initial_age=args.initial_age,
        tectonic_activity=args.tectonic_activity,
        continental_fraction=args.continental_fraction
    )
    
    # Initialize early planetary evolution
    print("Simulating early planetary evolution...")
    tectonics.initialize_planetary_evolution()
    
    # Initialize mantle convection system
    tectonics.initialize_mantle(num_cells=12, num_plumes=5)
    
    # Initialize plates
    tectonics.initialize_plates()
    
    return planet, tectonics

def run_simulation(planet, tectonics, args, session_dir):
    """Run the simulation with the specified parameters"""
    # Create metadata and history tracking
    history = {
        "parameters": vars(args),
        "stages": [],
        "events": []
    }
    
    # Save initial state
    save_state(planet, tectonics, session_dir, 0, "initial")
    
    print(f"\nRunning tectonic simulation for {args.steps} steps of {args.step_size} million years each...")
    
    # Main simulation loop
    for step in range(args.steps):
        print(f"\nStep {step+1}/{args.steps}: Simulating from {planet.age:.1f} to {planet.age + args.step_size:.1f} million years...")
        
        # Tectonic simulation step
        start_time = time.time()
        tectonics.simulate_step(time_step=args.step_size)
        tectonic_time = time.time() - start_time
        
        # Save periodic state
        step_name = f"step_{step+1}"
        save_state(planet, tectonics, session_dir, step+1, step_name)
        
        # Record history for this stage
        stage_data = {
            "age": planet.age,
            "step": step+1,
            "elevation_range": [float(planet.elevation.min()), float(planet.elevation.max())],
            "num_plates": len(tectonics.plates),
            "processing_time": tectonic_time,
            "continental_area": calculate_continental_area(planet),
            "ocean_area": calculate_ocean_area(planet),
            "plate_statistics": get_plate_statistics(tectonics)
        }
        history["stages"].append(stage_data)
        
        # Capture any significant events from this step
        for event in tectonics.get_recent_events():
            history["events"].append(event)
    
    # Save final simulation history
    save_history(history, session_dir)
    
    # Additional simulations if requested
    if args.run_climate or args.run_erosion or args.run_biomes:
        run_additional_simulations(planet, args, session_dir)
    
    return history

def run_additional_simulations(planet, args, session_dir):
    """Run additional simulation components"""
    # Erosion simulation 
    if args.run_erosion:
        print("\nRunning erosion simulation...")
        erosion_sim = ErosionSimulation(planet)
        erosion_sim.simulate(iterations=3)
        
        # Generate rivers
        print("Generating river networks...")
        rivers = erosion_sim.generate_rivers(min_flow=5)
        planet.rivers = rivers
        
        # Save visualization with erosion and rivers
        planet.visualize(save_path=os.path.join(session_dir, "planet_with_erosion.png"),
                        mode='elevation')
        
        # Save river data
        with open(os.path.join(session_dir, "river_data.json"), 'w') as f:
            river_data = [{"length": len(river), "vertices": river} for river in rivers]
            json.dump(river_data, f)
    
    # Climate simulation
    if args.run_climate:
        print("\nRunning climate simulation...")
        climate_sim = ClimateSimulation(planet)
        climate_sim.simulate(seasons=4)
        
        # Save temperature and precipitation visualizations
        planet.visualize(save_path=os.path.join(session_dir, "planet_temperature.png"),
                         mode='temperature')
        planet.visualize(save_path=os.path.join(session_dir, "planet_precipitation.png"),
                         mode='precipitation')
        
        # Save climate data
        save_climate_data(planet, session_dir)
    
    # Biome classification
    if args.run_biomes:
        if planet.temperature is None or planet.precipitation is None:
            print("Running climate simulation first (required for biomes)...")
            climate_sim = ClimateSimulation(planet)
            climate_sim.simulate(seasons=4)
        
        print("\nClassifying biomes...")
        biome_classifier = BiomeClassifier(planet)
        biome_classifier.classify()
        
        # Save biome visualization
        planet.visualize(save_path=os.path.join(session_dir, "planet_biomes.png"),
                         mode='biome')
        
        # Save biome data
        save_biome_data(planet, biome_classifier, session_dir)

def save_state(planet, tectonics, session_dir, step, name):
    """Save visualization and data for the current planet state"""
    # 3D visualizations
    if args.vis_mode in ['3d', 'both']:
        # Save 3D elevation visualization
        planet.visualize(
            save_path=os.path.join(session_dir, f"planet_{name}_3d.png"),
            mode='elevation'
        )
        
        # Save 3D plate visualization
        tectonics.visualize_plates(
            save_path=os.path.join(session_dir, f"plates_{name}_3d.png"),
            show_features=args.show_features
        )
    
    # 2D visualizations
    if args.vis_mode in ['2d', 'both']:
        # Save 2D elevation map
        planet.visualize_2d(
            save_path=os.path.join(session_dir, f"planet_{name}_2d.png"),
            mode='elevation',
            projection=args.projection
        )
        
        # Save 2D plate map
        tectonics.visualize_plates_2d(
            save_path=os.path.join(session_dir, f"plates_{name}_2d.png"),
            show_features=args.show_features,
            projection=args.projection
        )
    
    # Export heightmap if requested
    if args.export_heightmap or args.export_all:
        export_manager = ExportManager(planet)
        
        # Full planet heightmap
        export_manager.export_heightmap(
            os.path.join(session_dir, f"heightmap_{name}.png"),
            width=2048,
            height=1024
        )
        
        # Region heightmap if specified
        if args.region:
            try:
                lat1, lon1, lat2, lon2 = map(float, args.region.split(','))
                export_manager.export_heightmap(
                    os.path.join(session_dir, f"region_{name}.png"),
                    region=(lat1, lon1, lat2, lon2),
                    width=1024,
                    height=1024
                )
            except Exception as e:
                print(f"Error exporting region: {e}")

def save_history(history, session_dir):
    """Save the simulation history as JSON"""
    history_path = os.path.join(session_dir, "simulation_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Simulation history saved to {history_path}")

def save_climate_data(planet, session_dir):
    """Save climate data to files"""
    climate_data = {
        "temperature": {
            "min": float(planet.temperature.min()),
            "max": float(planet.temperature.max()),
            "mean": float(planet.temperature.mean())
        },
        "precipitation": {
            "min": float(planet.precipitation.min()),
            "max": float(planet.precipitation.max()),
            "mean": float(planet.precipitation.mean())
        }
    }
    
    climate_path = os.path.join(session_dir, "climate_data.json")
    with open(climate_path, 'w') as f:
        json.dump(climate_data, f, indent=2)

def save_biome_data(planet, biome_classifier, session_dir):
    """Save biome data to files"""
    # Count biomes
    unique_biomes, counts = np.unique(planet.biome_ids, return_counts=True)
    
    biome_data = {
        "biome_counts": {
            biome_classifier.get_biome_name(int(b)): int(c) 
            for b, c in zip(unique_biomes, counts)
        },
        "biome_types": [biome_classifier.get_biome_name(i) for i in range(len(biome_classifier.biome_types))]
    }
    
    biome_path = os.path.join(session_dir, "biome_data.json")
    with open(biome_path, 'w') as f:
        json.dump(biome_data, f, indent=2)

def calculate_continental_area(planet):
    """Calculate percentage of surface covered by continents"""
    land_count = np.sum(planet.elevation >= 0)
    return (land_count / len(planet.elevation)) * 100

def calculate_ocean_area(planet):
    """Calculate percentage of surface covered by oceans"""
    ocean_count = np.sum(planet.elevation < 0)
    return (ocean_count / len(planet.elevation)) * 100

def get_plate_statistics(tectonics):
    """Get detailed statistics for all plates"""
    stats = []
    for plate in tectonics.plates:
        plate_type = "Oceanic" if plate['is_oceanic'] else "Continental"
        vel_mag = np.linalg.norm(plate['velocity']) * 100  # Convert to cm/year
        
        stats.append({
            "id": plate['id'],
            "type": plate_type,
            "area_percentage": float(plate['area'] * 100),
            "velocity_cm_year": float(vel_mag),
            "continental_percentage": float(plate['continental_percentage']),
            "age": float(plate['age']),
            "feature_counts": {
                "orogenic_belts": len(plate['features']['orogenic_belts']),
                "rifts": len(plate['features']['rifts']),
                "volcanic_arcs": len(plate['features']['volcanic_arcs']),
                "hotspots": len(plate['features']['hotspots']),
                "transform_zones": len(plate['features']['transform_zones'])
            }
        })
    return stats

def print_final_stats(history, tectonics):
    """Print final statistics about the planet"""
    final_stage = history["stages"][-1]
    
    print("\n========================================")
    print("FINAL PLANET STATISTICS")
    print("========================================")
    print(f"Age: {tectonics.planet.age:.1f} million years")
    print(f"Number of plates: {len(tectonics.plates)}")
    print(f"Elevation range: {final_stage['elevation_range'][0]:.2f} to {final_stage['elevation_range'][1]:.2f} km")
    print(f"Land coverage: {final_stage['continental_area']:.1f}%")
    print(f"Ocean coverage: {final_stage['ocean_area']:.1f}%")
    
    print("\nMajor tectonic events:")
    for event in history["events"][-5:]:  # Show last 5 events
        print(f"- {event['age']} Ma: {event['description']}")
    
    print("\nPlate information:")
    for plate in sorted(tectonics.plates, key=lambda p: p['area'], reverse=True)[:5]:  # Top 5 plates
        plate_type = "Oceanic" if plate['is_oceanic'] else "Continental"
        vel_mag = np.linalg.norm(plate['velocity']) * 100
        print(f"- Plate {plate['id']}: {plate_type}, Size: {plate['area']*100:.1f}%, Velocity: {vel_mag:.1f} cm/year")
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup environment
    session_dir = setup_environment(args)
    
    # Create and initialize the planet
    planet, tectonics = create_planet(args)
    
    # Save initial state visualization
    save_state(planet, tectonics, session_dir, 0, "initial")
    
    # Run the simulation
    history = run_simulation(planet, tectonics, args, session_dir)
    
    # Save tectonic history visualization
    tectonics.visualize_history(save_path=os.path.join(session_dir, "tectonic_history.png"))
    
    # Print final statistics
    print_final_stats(history, tectonics)
    
    print(f"\nAll output saved to: {session_dir}")