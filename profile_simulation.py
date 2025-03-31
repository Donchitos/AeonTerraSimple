from memory_profiler import profile
from planet_sim.core.planet import Planet
from planet_sim.core.tectonics import TectonicSimulation

@profile
def run_test():
    planet = Planet(resolution=5)  # Increase if your system can handle it
    tectonics = TectonicSimulation(planet)
    tectonics.initialize_mantle()
    tectonics.initialize_plates()
    tectonics.simulate_step(time_step=100.0)
    return planet

if __name__ == "__main__":
    planet = run_test()
    print(f"Simulation completed with {len(planet.grid.vertices)} vertices")