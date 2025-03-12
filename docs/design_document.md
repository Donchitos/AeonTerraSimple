
 AeonTerra System Design Document

## Overview

This document outlines the technical architecture and design decisions for the AeonTerra planet generation system. It covers the core components, data flow, algorithms, and class relationships.

## System Architecture

AeonTerra follows a modular architecture with several distinct components that work together to create a complete planetary simulation:

```
[SphericalGrid] ◄───── [Planet] ◄────┬───── [TectonicSimulation]
     ▲                     ▲          │
     │                     │          ├───── [ClimateSimulation]
     │                     │          │
     │                     │          ├───── [ErosionSimulation]
     │                     │          │
     └─────────────────────┘          └───── [BiomeClassification]
```

### Component Relationships

- **SphericalGrid**: Foundation for all spatial calculations
- **Planet**: Core data container that holds the planet's state
- **Simulation Classes**: Each applies specific physical processes to the Planet

## Data Flow

The data flow through the system follows this general pattern:

1. Planet and grid initialization
2. Tectonic simulation
3. Erosion processes
4. Climate simulation
5. Biome classification
6. Visualization and export

```
[Initial Parameters] → [Planet Creation] → [Tectonic Simulation] → [Erosion] 
                                                                     │
[Export/Visualization] ← [Biome Classification] ← [Climate Simulation] ← ┘
```

## Core Classes Design

### SphericalGrid

Represents the planet's surface as a spherical mesh, typically derived from a subdivided icosahedron.

**Key Attributes:**
- `vertices`: 3D coordinates of points on the sphere
- `faces`: Triangular faces connecting vertices
- `radius`: Planet radius in kilometers

**Design Considerations:**
- Uses an icosahedron base with subdivision to create a nearly uniform sampling
- More efficient and less distorted than lat-long grids
- Allows for easy neighbor queries and spatial operations

### Planet

The central data container for the entire system, storing all physical properties of the planet.

**Key Attributes:**
- `grid`: The SphericalGrid representing the planet's surface
- `elevation`: Height values for each vertex
- `plate_ids`: Mapping of vertices to tectonic plates
- Various climate and biome data arrays

**Design Considerations:**
- Acts as a passive data container
- Provides visualization capabilities
- Does not implement simulation logic (that's in the specialized classes)

### TectonicSimulation

Handles the simulation of tectonic plate formation and movement.

**Key Attributes:**
- `planet`: Reference to the Planet being modified
- `plates`: List of plate objects with movement and boundary data
- `crust_type`, `crust_thickness`, `crust_density`: Arrays of physical properties

**Design Considerations:**
- Pure simulation of physical processes
- Maintains its own internal state for plate configurations
- Updates the planet's elevation and other attributes based on simulation results

## Algorithm Design

### Plate Generation Algorithm

1. Generate initial crust thickness/density variation using coherent noise
2. Create Voronoi-like regions on the sphere for plates
3. Assign crust properties to plates based on dominant type
4. Calculate isostatic elevation based on crust properties

### Plate Motion Algorithm

1. Calculate forces from mantle convection
2. Move plates using rotation matrices (great circle motion)
3. Detect plate boundary interactions
4. Apply appropriate geological processes based on boundary type
5. Update elevation using isostatic equilibrium

### Climate Simulation Algorithm (Planned)

1. Calculate initial temperature distribution based on latitude
2. Simulate atmospheric and oceanic circulation
3. Calculate precipitation based on moisture transport and terrain
4. Iterate to stable state with feedback loops

## Data Structures

### Spherical Grid Representation

The spherical grid uses these core data structures:
- `vertices`: Nx3 array of 3D coordinates
- `faces`: Mx3 array of vertex indices forming triangles
- Neighbor lists: For each vertex, indices of connected vertices

### Plate Representation

Each tectonic plate is a dictionary with:
- `id`: Unique identifier
- `vertices`: List of vertex indices in the plate
- `center`: 3D coordinates of center of mass
- `velocity`: 3D vector of movement
- `is_oceanic`: Boolean flag
- Additional metadata and feature lists

## Performance Considerations

### Time Complexity

- Plate generation: O(V log P) where V = vertices, P = plates
- Simulation step: O(V + B) where B = boundary vertices
- Climate simulation: O(V log V) due to moisture transport calculations

### Space Complexity

- Base planet: O(V) for vertices, faces, and property arrays
- Simulation history: O(V * T) where T = number of time steps saved

### Optimization Strategies

1. **Spatial Partitioning**: For faster nearest-neighbor queries
2. **Adaptive Resolution**: Higher detail only where needed
3. **Parallelization**: Multi-threading for independent calculations
4. **JIT Compilation**: Using Numba for performance-critical sections

## Extensibility

The system is designed to be extensible in several ways:

1. **New Simulation Types**: Additional physical processes can be added as new Simulation classes
2. **Alternative Grid Types**: The SphericalGrid can be replaced with other representations
3. **Customization Points**: Key physical constants are exposed as parameters

## Future Architectural Considerations

1. **Multi-Scale Approach**: Hierarchical grid for varying levels of detail
2. **GPU Acceleration**: Moving computation-heavy parts to GPU
3. **Distributed Processing**: Breaking large worlds into tiles for parallel processing
4. **Real-time Feedback**: Architecture for interactive editing and visualization

## Technical Debt and Limitations

Current known limitations in the architecture:

1. Memory usage scales with square of resolution
2. Some physical processes are simplified for performance reasons
3. No built-in LOD (Level of Detail) system yet
4. Limited ability to handle very exotic planet types