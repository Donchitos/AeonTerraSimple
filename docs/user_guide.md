# AeonTerra User Guide

This guide provides detailed instructions for using the AeonTerra planet generation system, from basic setup to advanced planet creation and data export.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Creating Custom Planets](#creating-custom-planets)
4. [Visualizing Planets](#visualizing-planets)
5. [Exporting Data](#exporting-data)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7 or higher
- Git (for cloning the repository)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AeonTerra.git
   cd AeonTerra
   ```

2. Install required dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```

For optional components:
```bash
# For 3D visualization
pip install pyvista

# For advanced performance
pip install numba

# For GUI (when implemented)
pip install pyqt5
```

## Quick Start

To generate your first planet:

1. Run the included test script:
   ```bash
   python test_planet.py
   ```

2. This will create several image files showing your planet:
   - `planet_initial.png` - The initial terrain
   - `plates_initial.png` - The plate configuration
   - `planet_4600.0.png`, etc. - Planet at different times in its evolution
   - `tectonic_history.png` - Graphs showing planet development

## Creating Custom Planets

### Basic Planet Parameters

Edit the test script to customize your planet:

```python
# Create a planet with custom parameters
planet = Planet(
    radius=6371.0,    # Planet radius in km (Earth = 6371)
    resolution=4      # Detail level (4 = ~2500 vertices, 5 = ~10000)
)

# Initialize tectonic simulation with custom settings
tectonics = TectonicSimulation(
    planet,
    num_plates=8,     # Number of tectonic plates
    initial_age=4500  # Starting age in million years
)

# Configure mantle convection
tectonics.initialize_mantle(
    num_cells=12,     # Number of convection cells
    num_plumes=5      # Number of mantle plumes (hotspots)
)
```

### Simulating Planet Evolution

Control how long to simulate plate tectonics:

```python
# Run simulation for 500 million years in 5 steps
for i in range(5):
    tectonics.simulate_step(time_step=100.0)
    # Save intermediate visualizations
    planet.visualize(save_path=f"planet_{planet.age}.png")
```

Larger time steps will produce more dramatic changes but might miss some details. For higher fidelity, use smaller steps:

```python
# More detailed evolution with 20 million year steps
for i in range(25):
    tectonics.simulate_step(time_step=20.0)
```

### Advanced Planet Customization

For fine-grained control over the planet's properties:

```python
# Configure an Earth-like planet
planet = Planet(radius=6371.0, resolution=4)

# Initialize with specific settings
tectonics = TectonicSimulation(planet, num_plates=12, initial_age=4500)

# Control the mantle convection patterns
tectonics.CONTINENTAL_DENSITY = 2.7     # Density of continental crust
tectonics.OCEANIC_DENSITY = 3.0         # Density of oceanic crust
tectonics.AVG_CONTINENTAL_THICKNESS = 35.0  # Average thickness in km
tectonics.AVG_OCEANIC_THICKNESS = 7.0       # Average thickness in km

# Initialize with these settings
tectonics.initialize_mantle()
tectonics.initialize_plates()
```

## Visualizing Planets

### Visualization Modes

The `planet.visualize()` method supports different visualization modes:

```python
# Show elevation (default)
planet.visualize(save_path="elevation.png", mode='elevation')

# Show tectonic plates
tectonics.visualize_plates(save_path="plates.png")

# Show tectonic history
tectonics.visualize_history(save_path="history.png")
```

When climate simulation is implemented, additional modes will be available:

```python
# Show temperature distribution
planet.visualize(save_path="temperature.png", mode='temperature')

# Show precipitation patterns
planet.visualize(save_path="precipitation.png", mode='precipitation')

# Show biome distribution
planet.visualize(save_path="biomes.png", mode='biome')
```

### Customizing Visualizations

You can modify the visualization code in `planet.py` to customize the appearance. For example, to change the color scheme for elevation:

```python
# In Planet.visualize() method
if mode == 'elevation':
    if elevation < 0:
        # Custom ocean colors - deeper blue for deeper ocean
        colors.append([0, 0, 0.5 + 0.5 * elevation / min(self.elevation)])
    else:
        # Custom land colors - green to brown to white
        rel_height = elevation / max(0.001, max(self.elevation))
        if rel_height < 0.2:
            colors.append([0.2, 0.6, 0.2])  # Low land, green
        elif rel_height < 0.5:
            colors.append([0.6, 0.4, 0.2])  # Hills, brown
        else:
            colors.append([0.8, 0.8, 0.8])  # Mountains, light gray
```

## Exporting Data

*Note: The export functionality is planned for future implementation.*

### Exporting Heightmaps

```python
from planet_sim.utils.export import export_heightmap

# Export the full planet heightmap in equirectangular projection
export_heightmap(planet, "planet_heightmap.png", width=4096, height=2048)

# Export a specific region (lat_min, lon_min, lat_max, lon_max)
export_heightmap(planet, "region_heightmap.png", 
                 region=(30, -120, 50, -80),  # Northern US region
                 width=2048, height=2048)
```

### Exporting for Game Engines

```python
from planet_sim.utils.export import export_for_unreal

# Export heightmap and additional data for Unreal Engine
export_for_unreal(planet, "unreal_export", 
                  region=(30, -120, 50, -80),
                  resolution=1024)  # 1024x1024 heightmap
```

## Troubleshooting

### Common Issues

1. **Memory Errors**

   If you encounter memory errors with high-resolution planets:
   
   ```
   Solution: Reduce the 'resolution' parameter:
   planet = Planet(resolution=3)  # Lower resolution
   ```

2. **Scientific Libraries Missing**

   If you see errors about missing NumPy or SciPy:
   
   ```
   Solution: Ensure you've installed all dependencies:
   pip install numpy scipy matplotlib
   ```

3. **Visualization Issues**

   If visualizations fail to appear:
   
   ```
   Solution: Try saving to a file instead of displaying:
   planet.visualize(save_path="debug.png")
   ```

### Getting Help

If you encounter issues not covered here:

1. Check for existing issues on GitHub
2. Provide details when reporting problems:
   - Your Python version
   - Complete error message
   - Minimal code example that reproduces the issue