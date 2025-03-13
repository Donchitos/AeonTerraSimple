from planet_sim.core.planet import Planet
from planet_sim.core.tectonics import TectonicSimulation
import os
import numpy as np

# Create output directory
os.makedirs("output", exist_ok=True)

# Create planet
print("Creating planet...")
planet = Planet(radius=6371.0, resolution=4)

# Initialize tectonic simulation
print("Initializing tectonic simulation...")
tectonics = TectonicSimulation(
    planet, 
    num_plates=8,
    initial_age=4500,
    tectonic_activity=1.0,
    continental_fraction=0.3
)

# Initialize early planetary evolution
print("Initializing planetary evolution...")
tectonics.initialize_planetary_evolution()

# Initialize mantle convection and plates
print("Initializing mantle...")
tectonics.initialize_mantle(num_cells=10, num_plumes=4)

print("Initializing plates...")
tectonics.initialize_plates()

# Save initial state visualization
planet.visualize(save_path="output/planet_initial.png")
tectonics.visualize_plates(save_path="output/plates_initial.png")

# Run simulation for 5 steps of 100 million years each
print("\nRunning simulation...")
for step in range(5):
    print(f"Step {step+1}/5: Simulating 100 million years...")
    tectonics.simulate_step(time_step=100.0)
    
    # Save visualization after each step
    planet.visualize(save_path=f"output/planet_step{step+1}.png")
    tectonics.visualize_plates(save_path=f"output/plates_step{step+1}.png")

# Save tectonic history
tectonics.visualize_history(save_path="output/tectonic_history.png")

# Print final statistics
print("\nFinal planet statistics:")
print(f"Age: {planet.age:.1f} million years")
print(f"Elevation range: {np.min(planet.elevation):.2f} to {np.max(planet.elevation):.2f} km")

landmass = np.sum(planet.elevation >= 0) / len(planet.elevation) * 100
print(f"Land coverage: {landmass:.1f}%")
print(f"Ocean coverage: {100 - landmass:.1f}%")
print(f"Number of plates: {len(tectonics.plates)}")

# Print recent tectonic events
print("\nRecent tectonic events:")
for event in tectonics.get_recent_events():
    print(f"- {event['age']} Ma: {event['description']}")

print("\nSimulation complete! Check the output directory for visualizations.")