

                        # AeonTerra API Documentation

This document provides a reference for the classes, methods, and parameters available in the AeonTerra planet generation system.

## Core Classes

### `planet_sim.utils.SphericalGrid`

Represents the planet's surface as a spherical mesh.

#### Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `__init__` | `radius=6371.0, resolution=5` | Initialize spherical grid with given radius and detail level |
| `_create_icosahedron` | None | Creates the base icosahedron vertices and faces |
| `_subdivide` | None | Subdivides each face into 4 new faces |
| `get_vertex_neighbors` | None | Returns a list of neighboring vertices for each vertex |
| `get_lat_lon` | `vertex_idx` | Converts a vertex position to latitude/longitude |

### `planet_sim.core.Planet`

Represents a planet with physical properties and terrain data.

#### Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `__init__` | `radius=6371.0, resolution=5` | Initialize planet with given radius and detail level |
| `initialize_terrain` | None | Initialize basic terrain with random noise |
| `visualize` | `save_path=None, mode='elevation'` | Visualize the planet in 3D with specified display mode |

### `planet_sim.core.TectonicSimulation`

Simulates tectonic plate movement and interactions.

#### Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `__init__` | `planet, num_plates=12, initial_age=4500` | Initialize tectonic simulation |
| `initialize_mantle` | `num_cells=12, num_plumes=5` | Set up mantle convection system |
| `initialize_plates` | None | Create initial plates using realistic crust model |
| `simulate_step` | `time_step=10.0` | Simulate one step of plate movement |
| `visualize_plates` | `save_path=None` | Create visualization of plate configuration |
| `visualize_history` | `save_path=None` | Visualize changes in planet over time |

#### Key Parameters Explained

- `radius`: Planet radius in kilometers (Earth = 6371)
- `resolution`: Detail level of the sphere (each +1 quadruples vertices)
- `num_plates`: Number of tectonic plates to generate
- `initial_age`: Starting age of the planet in million years
- `time_step`: Amount of time to simulate in million years

## Example Usage

### Basic Planet Creation

```python
from planet_sim.core.planet import Planet

# Create a planet with Earth-like radius
planet = Planet(radius=6371.0, resolution=4)

# Initialize with random terrain
planet.initialize_terrain()

# Visualize and save to file
planet.visualize(save_path="my_planet.png")
```

### Tectonic Simulation

```python
from planet_sim.core.planet import Planet
from planet_sim.core.tectonics import TectonicSimulation

# Create a planet
planet = Planet(resolution=4)

# Initialize tectonic simulation
tectonics = TectonicSimulation(planet, num_plates=8)
tectonics.initialize_mantle(num_cells=12)
tectonics.initialize_plates()

# Run simulation for 100 million years
tectonics.simulate_step(time_step=100.0)

# Visualize results
planet.visualize(save_path="planet_after_tectonics.png")
tectonics.visualize_plates(save_path="plate_configuration.png")
```

## Data Structures

### Planet Data

The `Planet` object contains the following key attributes:

- `grid`: The `SphericalGrid` object representing the planet surface
- `elevation`: Numpy array of elevation values for each vertex
- `plate_ids`: Numpy array identifying which plate each vertex belongs to
- `temperature`: Numpy array of temperature values (if climate is simulated)
- `precipitation`: Numpy array of precipitation values (if climate is simulated)
- `biome_ids`: Numpy array of biome type IDs (if biomes are classified)
- `rivers`: List of river paths (each a list of vertex indices)

### Tectonic Plate Data

Each plate in the `TectonicSimulation.plates` list is a dictionary with:

- `id`: Unique identifier for the plate
- `center`: 3D coordinates of plate center of mass
- `vertices`: List of vertex indices in this plate
- `is_oceanic`: Boolean indicating if plate is primarily oceanic
- `velocity`: 3D vector of plate motion
- `area`: Fraction of planet surface covered by this plate
- `age`: Age of the plate in million years
- `continental_percentage`: Percentage of continental crust
- `boundaries`: List of boundary features
- `features`: Dictionary of geological features (mountain ranges, rifts, etc.)

## Extension Points

The AeonTerra API is designed to be extendable. The key extension points are:

1. **New Simulation Types**: Create new simulation classes in the `core` module by following the pattern of `TectonicSimulation`
2. **Visualization Modes**: Add new visualization modes to the `Planet.visualize()` method
3. **Custom Planet Types**: Create factory functions that configure planets with specific parameters


