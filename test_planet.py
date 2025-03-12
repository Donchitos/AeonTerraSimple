from planet_sim.core.planet import Planet
from planet_sim.core.tectonics import TectonicSimulation
import numpy as np  # Add this import

def main():
    # Create a planet
    print("Creating planet...")
    planet = Planet(resolution=4)  # Lower resolution for faster testing
    
    # Initialize tectonic simulation
    print("Initializing tectonic simulation...")
    tectonics = TectonicSimulation(planet, num_plates=8, initial_age=4500)
    
    # Initialize mantle convection system
    tectonics.initialize_mantle(num_cells=12)
    
    # Initialize plates
    tectonics.initialize_plates()
    
    # Save initial state visualization
    planet.visualize(save_path="planet_initial.png", mode='elevation')
    tectonics.visualize_plates(save_path="plates_initial.png")
    
    # Run simulation for 500 million years
    print("Running tectonic simulation for 500 million years...")
    for i in range(5):
        tectonics.simulate_step(time_step=100.0)
        # Save an intermediate visualization
        planet.visualize(save_path=f"planet_{planet.age}.png", mode='elevation')
        tectonics.visualize_plates(save_path=f"plates_{planet.age}.png")
    
    # Save history visualization
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
        print(f"Plate {plate['id']}: {plate_type}, Size: {plate['area']*100:.1f}% of surface")

if __name__ == "__main__":
    main()