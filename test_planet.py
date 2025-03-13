from planet_sim.core.planet import Planet
from planet_sim.core.tectonics import TectonicSimulation
import numpy as np

def main():
    # Create a planet
    print("Creating planet...")
    planet = Planet(resolution=4)  # Lower resolution for faster testing
    
    # Initialize tectonic simulation with new customization parameters
    print("Initializing tectonic simulation...")
    tectonics = TectonicSimulation(
        planet, 
        num_plates=8, 
        initial_age=4500,
        tectonic_activity=1.2,     # More active than Earth
        mantle_viscosity=0.9,      # Less viscous mantle (faster plates)
        continental_fraction=0.35,  # More continental crust
        plate_thickness_variation=1.3  # More varied crust thickness
    )
    
    # Initialize early planetary evolution (new feature)
    tectonics.initialize_planetary_evolution()
    
    # Initialize mantle convection system
    tectonics.initialize_mantle(num_cells=12, num_plumes=5)
    
    # Initialize plates
    tectonics.initialize_plates()
    
    # Save initial state visualization
    planet.visualize(save_path="planet_initial.png", mode='elevation')
    tectonics.visualize_plates(save_path="plates_initial.png", show_features=True)
    
    # Run simulation for 500 million years
    print("Running tectonic simulation for 500 million years...")
    for i in range(5):
        tectonics.simulate_step(time_step=100.0)
        # Save an intermediate visualization
        planet.visualize(save_path=f"planet_{planet.age:.1f}.png", mode='elevation')
        tectonics.visualize_plates(save_path=f"plates_{planet.age:.1f}.png", show_features=True)
    
    # Save history visualization with enhanced statistics
    tectonics.visualize_history(save_path="tectonic_history.png")
    
    # Print final statistics
    print("\nFinal planet statistics:")
    print(f"Age: {planet.age} million years")
    print(f"Number of plates: {len(tectonics.plates)}")
    print(f"Elevation range: {planet.elevation.min():.2f} to {planet.elevation.max():.2f} km")
    
    # Calculate land vs ocean percentage
    land_area = 100 * np.sum(planet.elevation >= 0) / len(planet.elevation)
    ocean_area = 100 - land_area
    print(f"Land coverage: {land_area:.1f}%")
    print(f"Ocean coverage: {ocean_area:.1f}%")
    
    # Print plate information
    print("\nCurrent plates:")
    for plate in tectonics.plates:
        plate_type = "Oceanic" if plate['is_oceanic'] else "Continental"
        vel_mag = np.linalg.norm(plate['velocity']) * 100  # Convert to cm/year
        print(f"Plate {plate['id']}: {plate_type}, Size: {plate['area']*100:.1f}%, Velocity: {vel_mag:.1f} cm/year")

if __name__ == "__main__":
    main()