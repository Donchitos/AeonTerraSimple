# AeonTerraSimple# AeonTerra: Procedural Planet Generator

AeonTerra is a scientifically grounded planet generation system that simulates the formation and evolution of Earth-like planets from first principles. It models plate tectonics, climate systems, and biome formation to create realistic, physically accurate virtual worlds.

## Project Overview

The goal of AeonTerra is to create procedurally generated planets with scientifically plausible geological features, climate patterns, and biomes. Unlike many other procedural terrain generators that create heightmaps through noise algorithms, AeonTerra simulates the actual geological processes that form planets - from mantle convection and plate tectonics to erosion and climate systems.

This system can be used by worldbuilders across different creative needs:
- Game developers creating realistic terrains
- Writers developing scientifically plausible fictional worlds
- Tabletop RPG creators making campaign settings
- Educators demonstrating geological and climate processes

## Scientific Foundation

AeonTerra is built on scientific models of planetary formation and evolution:

### Tectonic System
- **Mantle Convection**: Simulates how heat moves through the planet's mantle to drive plate movement
- **Continental vs. Oceanic Crust**: Different densities, thicknesses, and behaviors
- **Plate Boundaries**: Convergent (subduction, collision), divergent (spreading), and transform
- **Isostatic Equilibrium**: The floating of crust on the mantle based on density and thickness

### Geological Processes
- **Orogeny**: Mountain formation at convergent boundaries
- **Rift Formation**: Continental splitting and ocean basin formation
- **Volcanic Activity**: At subduction zones, rifts, and hotspots
- **Erosion**: Weathering and transport of surface materials

### Climate System
- **Atmospheric Circulation**: Prevailing winds and pressure systems
- **Ocean Currents**: Heat distribution through ocean systems
- **Precipitation Patterns**: Based on temperature, terrain, and air movement
- **Biome Formation**: Classification of regions based on temperature and precipitation

## Implementation

AeonTerra is implemented in Python using scientific computing libraries:

- **Core Language**: Python 3.x
- **Scientific Computing**: NumPy, SciPy
- **Visualization**: Matplotlib, PyVista
- **Performance Optimization**: Numba (for future implementation)

## Key Features

Current features:
- Spherical planet representation with configurable resolution
- Realistic tectonic plate generation and movement
- Geological feature formation (mountains, trenches, islands)
- Visualization of the planet's elevation and plates
- Tracking of planet evolution over time

Planned features:
- Climate simulation (temperature, precipitation, winds)
- River and lake formation
- Biome classification
- Region selection and export for game engines
- Advanced erosion simulation

## Project Structure

```
planet_sim/
├── core/
│   ├── planet.py           # Main Planet class
│   ├── tectonics.py        # Plate tectonics simulation
│   ├── erosion.py          # Erosion and weathering simulation (planned)
│   ├── climate.py          # Climate simulation (planned)
│   └── biome.py            # Biome classification (planned)
├── utils/
│   ├── sphere_grid.py      # Spherical grid representation
│   ├── export.py           # Export functionality (planned)
│   └── heightmap.py        # Heightmap operations (planned)
└── visualization/
    ├── planet_viewer.py    # 3D visualization (planned)
    ├── region_selector.py  # UI for selecting regions (planned)
    └── data_visualizer.py  # Visualize climate, biomes, etc. (planned)
```

## Usage

### Running the Basic Simulation

```bash
# Clone the repository
git clone https://github.com/yourusername/AeonTerra.git
cd AeonTerra

# Install dependencies
pip install numpy scipy matplotlib

# Run the test script
python test_planet.py
```

This will generate a planet with default settings and save several visualizations:
- `planet_initial.png` - The initial elevation map
- `plates_initial.png` - The initial plate configuration
- `planet_{age}.png` - Planet elevation at different time points
- `plates_{age}.png` - Plate configurations at different time points
- `tectonic_history.png` - Statistics about how the planet evolved

### Configuring the Simulation

You can modify parameters in `test_planet.py` to change various aspects of the simulation:

```python
# Create a planet with custom parameters
planet = Planet(resolution=5)  # Higher resolution = more detailed planet

# Initialize tectonic simulation with custom settings
tectonics = TectonicSimulation(planet, num_plates=12, initial_age=4500)

# Initialize mantle with more convection cells and plumes
tectonics.initialize_mantle(num_cells=16, num_plumes=7)
```

## Development Roadmap

### Phase 1: Core Planetary Framework (Current)
- ✅ Implement spherical grid system
- ✅ Create initial plate tectonics simulation
- ✅ Generate terrain from tectonic processes
- ✅ Develop visualization system

### Phase 2: Enhanced Geological Processes
- ⬜ Implement comprehensive erosion simulation
- ⬜ Add river systems and water flow
- ⬜ Improve mountain formation and terrain detail
- ⬜ Add volcanic features and hotspot tracks

### Phase 3: Climate System
- ⬜ Implement atmospheric circulation model
- ⬜ Add ocean currents and heat transfer
- ⬜ Model precipitation based on topography and wind
- ⬜ Create seasonal variations

### Phase 4: Biomes and Export
- ⬜ Classify biomes based on climate data
- ⬜ Create region selection and projection tool
- ⬜ Implement export to game engine formats
- ⬜ Develop user interface for planet customization

## Technical Details

### Spherical Grid

The planet is represented as a spherical grid derived from a subdivided icosahedron. This provides a nearly uniform distribution of vertices over the sphere, avoiding the distortion issues that come with latitude-longitude grids.

### Tectonic Simulation

The tectonic simulation follows these key processes:

1. **Mantle Convection**: Establishes upwelling and downwelling regions that drive plate motion
2. **Plate Generation**: Creates plates with distinct properties based on crust composition
3. **Plate Motion**: Moves plates based on mantle forces and previous momentum
4. **Boundary Processes**: Handles interactions between plates at their boundaries
5. **Crust Evolution**: Ages crust, creates new crust, and models isostatic adjustments

### Data Representation

- **Vertices**: 3D coordinates of points on the planet's surface
- **Faces**: Triangles connecting vertices to form the planet mesh
- **Plates**: Collections of vertices with shared motion and properties
- **Crust Properties**: Type, thickness, density, and age for each vertex
- **Elevation**: Height above or below sea level at each vertex

## Documentation

The following documentation is available in the `docs/` directory:

- [API Documentation](docs/api_documentation.md) - Reference for classes and methods
- [User Guide](docs/user_guide.md) - Instructions for using the system
- [Design Document](docs/design_document.md) - Technical architecture and design decisions
- [Export Formats](docs/export_formats.md) - Specification for data export formats
- [Implementation Notes](docs/implementation_notes.md) - Details on algorithms and optimization

Additional documentation:
- [Scientific Principles](docs/scientific_principles.md) - Scientific background
- [Project Plan](docs/project_plan.md) - Development roadmap and approach

## Credits

AeonTerra was inspired by academic research in geology, planetary science, and procedural generation. It draws on techniques from:

- Real-world plate tectonic models
- Geophysical principles of isostasy and crustal dynamics
- Climate system modeling
- Procedural continent and terrain generation research

## License

This project is licensed under the MIT License - see the LICENSE file for details.