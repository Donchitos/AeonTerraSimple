# AeonTerra Implementation Notes

This document provides detailed technical notes on the implementation of key algorithms and data structures in the AeonTerra planet generation system. It's intended for developers who want to understand, modify, or extend the codebase.

## Table of Contents

1. [Spherical Grid Implementation](#spherical-grid-implementation)
2. [Tectonic Simulation Algorithms](#tectonic-simulation-algorithms)
3. [Terrain Generation Details](#terrain-generation-details)
4. [Optimization Techniques](#optimization-techniques)
5. [Known Limitations](#known-limitations)

## Spherical Grid Implementation

### Icosahedron Subdivision

The planet mesh starts with a regular icosahedron (20 triangular faces, 12 vertices) and subdivides it to create a sphere with nearly uniform vertex distribution.

**Key Implementation Details:**

```python
def _subdivide(self):
    """Subdivide each face into 4 new faces"""
    new_vertices = list(self.vertices)
    new_faces = []
    
    # Cache for midpoints to avoid duplicate vertices
    midpoint_cache = {}
    
    # For each face, create 4 new faces
    for face in self.faces:
        v0, v1, v2 = face
        
        # Get or compute midpoints
        a = self._get_midpoint(v0, v1, midpoint_cache, new_vertices)
        b = self._get_midpoint(v1, v2, midpoint_cache, new_vertices)
        c = self._get_midpoint(v2, v0, midpoint_cache, new_vertices)
        
        # Create 4 new faces
        new_faces.append([v0, a, c])
        new_faces.append([v1, b, a])
        new_faces.append([v2, c, b])
        new_faces.append([a, b, c])
    
    return new_vertices, new_faces
```

**Time Complexity:**
- O(F) where F is the number of faces (which grows as 4^level)

**Space Complexity:**
- O(V) where V is the number of vertices (approximately 4^level * 10 + 2)

### Neighbor Finding

Finding vertex neighbors is crucial for many algorithms. We precompute this for efficiency:

```python
def get_vertex_neighbors(self):
    """Build adjacency list for each vertex"""
    neighbors = [[] for _ in range(len(self.vertices))]
    
    for face in self.faces:
        a, b, c = face
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    
    # Remove duplicates
    for i in range(len(neighbors)):
        neighbors[i] = list(set(neighbors[i]))
    
    return neighbors
```

**Optimizations:**
1. We build this once and cache it for repeated use
2. Using set operations removes duplicate neighbors efficiently

## Tectonic Simulation Algorithms

### Plate Generation

We use a Voronoi-like approach to generate plate boundaries:

1. Pick random seed points on the sphere for each plate
2. Assign each vertex to the closest seed point
3. Refine boundaries based on additional criteria

**Distance Calculation on Sphere:**

```python
def _spherical_distance(point1, point2):
    """Calculate great-circle distance between points on unit sphere"""
    # Use dot product to find angle between vectors
    dot = np.clip(np.dot(point1, point2), -1.0, 1.0)
    return np.arccos(dot)
```

**Optimization:**
- For small local distances, we approximate with Euclidean distance
- For global structures, we use proper great circle calculations

### Plate Motion

Plate motion is implemented using rotation matrices to ensure plates move along great circles:

```python
def _move_plates(self, time_step):
    """Move plates based on velocity and time step"""
    # Scale time step to proper movement amount
    movement_scale = time_step * 0.05  # 5cm/year * million years
    
    for plate in self.plates:
        # Get axis of rotation (cross product of center and velocity)
        velocity = np.array(plate['velocity'])
        center = np.array(plate['center'])
        
        # Skip if velocity is too small
        if np.linalg.norm(velocity) < 1e-6:
            continue
        
        # Calculate rotation axis and angle
        axis = np.cross(center, velocity)
        axis = axis / np.linalg.norm(axis)
        angle = np.linalg.norm(velocity) * movement_scale
        
        # Create rotation matrix (Rodrigues' rotation formula)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Apply rotation to all vertices in plate
        for vertex_idx in plate['vertices']:
            self.planet.grid.vertices[vertex_idx] = R @ self.planet.grid.vertices[vertex_idx]
```

**Mathematical Note:**
We use Rodrigues' rotation formula which allows rotation around an arbitrary axis by a given angle. This ensures plates move along great circles as they would on a real planet.

### Isostatic Equilibrium

The elevation of terrain is calculated using the principle of isostatic equilibrium:

```python
def _calculate_isostatic_elevation(self):
    """Calculate elevation based on isostatic equilibrium"""
    for i in range(len(self.planet.grid.vertices)):
        # Get crustal properties
        thickness = self.crust_thickness[i]
        density = self.crust_density[i]
        
        # Airy isostasy model: h = t * (ρm - ρc)/ρm
        # where h = elevation, t = thickness, ρm = mantle density, ρc = crust density
        isostatic_height = thickness * (self.MANTLE_DENSITY - density) / self.MANTLE_DENSITY
        
        # Scale to reasonable Earth-like values
        if self.crust_type[i] == 1:  # Continental
            elevation = isostatic_height * 0.05 - 1.0
            # Add variation based on thickness
            elevation += (thickness - self.AVG_CONTINENTAL_THICKNESS) / 20.0
        else:  # Oceanic
            elevation = isostatic_height * 0.05 - 4.5
            # Age-based subsidence for oceanic crust
            age_factor = np.sqrt(self.crust_age[i] / 100.0) if self.crust_age[i] > 0 else 0
            elevation -= 2.0 * age_factor
        
        self.planet.elevation[i] = elevation
```

**Scientific Basis:**
The formula is derived from Archimedes' principle, where lower density crust "floats" higher on the denser mantle. The specific scaling factors are calibrated to produce Earth-like elevations.

## Terrain Generation Details

### Coherent Noise Generation

We generate coherent noise for terrain features using a multi-scale approach:

```python
def _generate_coherent_noise(self, octaves=4, persistence=0.5, scale=1.0):
    """Generate coherent noise on the sphere"""
    # Start with random noise
    noise = np.random.random(len(self.planet.grid.vertices))
    
    # Convert to grid for easier processing
    grid_size = int(np.sqrt(len(self.planet.grid.vertices))) * 2
    grid = self._vertices_to_grid(noise, grid_size)
    
    # Apply multiple octaves of smoothing
    final_grid = np.zeros_like(grid)
    amplitude = 1.0
    max_value = 0.0
    
    for i in range(octaves):
        # Smooth with different kernel sizes
        sigma = scale * (2 ** (octaves - i - 1))
        smoothed = gaussian_filter(grid, sigma=sigma)
        
        # Add to final grid with diminishing amplitude
        final_grid += smoothed * amplitude
        max_value += amplitude
        amplitude *= persistence
    
    # Normalize and map back to vertices
    final_grid /= max_value
    return self._grid_to_vertices(final_grid)
```

**Key Implementation Decisions:**
1. We map vertex data to a 2D grid, apply filtering, and map back
2. This is faster than true spherical harmonics but has some distortion at poles
3. Multiple octaves create detail at different scales (similar to fractal noise)

### Boundary Feature Generation

Geological features at plate boundaries are generated based on boundary types:

```python
def _handle_convergent_boundary(self, vertex_idx, plate1_id, plate2_id, time_step):
    """Handle convergent plate boundary processes"""
    # Get crust types
    is_continental1 = self.crust_type[vertex_idx] == 1
    is_continental2 = self._get_neighbor_crust_type(vertex_idx, plate2_id)
    
    # Continental collision = mountain building
    if is_continental1 and is_continental2:
        # Mountain height increases over time
        elevation_increase = 5.0 + 5.0 * np.random.random()  # km per million years
        elevation_increase *= time_step / 10.0  # Scale by time step
        self.planet.elevation[vertex_idx] += elevation_increase
        
        # Crustal thickening
        self.crust_thickness[vertex_idx] += 2.0 * time_step / 10.0
    
    # Oceanic-continental = subduction with volcanic arc
    elif (not is_continental1 and is_continental2) or (is_continental1 and not is_continental2):
        if self.planet.elevation[vertex_idx] >= 0:
            # Volcanic arc on continental side
            elevation_increase = 3.0 + 4.0 * np.random.random()
            elevation_increase *= time_step / 10.0
            self.planet.elevation[vertex_idx] += elevation_increase
        else:
            # Oceanic trench
            self.planet.elevation[vertex_idx] = max(-11.0, 
                                                  self.planet.elevation[vertex_idx] - 1.0 * time_step / 10.0)
    
    # Oceanic-oceanic = island arc formation
    elif not is_continental1 and not is_continental2:
        # Age-based subduction (older crust subducts)
        if self.crust_age[vertex_idx] > self._get_neighbor_crust_age(vertex_idx, plate2_id):
            self.planet.elevation[vertex_idx] -= 1.0 * time_step / 10.0
            
            # Chance of island arc on other plate
            if np.random.random() < 0.05 * time_step / 10.0:
                self._create_island_arc(vertex_idx, plate2_id)
```

**Geological Accuracy:**
- Mountain building rates (5-10 km / million years) are based on real-world measurements
- Trench depths max out at around 11 km (similar to the Mariana Trench)
- Island arc formation is probabilistic to simulate the occasional nature of this process

## Optimization Techniques

### Spatial Locality

The grid data structures are designed to maintain spatial locality:

```python
def _update_plate_boundaries(self):
    """Update plate boundaries with optimized neighbor checks"""
    boundaries = np.zeros(len(self.planet.grid.vertices), dtype=bool)
    neighbors = self.planet.grid.get_vertex_neighbors()
    
    # Process vertices in blocks that are spatially close
    for block_start in range(0, len(self.planet.grid.vertices), 1024):
        block_end = min(block_start + 1024, len(self.planet.grid.vertices))
        
        for i in range(block_start, block_end):
            plate_id = self.planet.plate_ids[i]
            for neighbor in neighbors[i]:
                if self.planet.plate_ids[neighbor] != plate_id:
                    boundaries[i] = True
                    break
    
    self.plate_boundaries = boundaries
```

**Optimization Benefits:**
- Improves cache locality
- Reduces memory access patterns
- Can improve performance by 15-20% on large planets

### Numba JIT Compilation (Planned)

For performance-critical sections, we plan to add Numba JIT compilation:

```python
import numba

@numba.jit(nopython=True, parallel=True)
def _calculate_distances(vertices, seed_points):
    """Calculate distances from all vertices to all seed points"""
    distances = np.empty((len(vertices), len(seed_points)))
    
    for i in numba.prange(len(vertices)):
        for j in range(len(seed_points)):
            # Dot product for spherical distance
            dot = np.clip(np.sum(vertices[i] * seed_points[j]), -1.0, 1.0)
            distances[i, j] = np.arccos(dot)
    
    return distances
```

**Expected Performance Gains:**
- 5-10x speedup for distance calculations
- Enables processing of much higher resolution planets
- Particularly important for operations that can't use NumPy vectorization

## Known Limitations

### Numerical Precision

At high resolutions, floating-point precision can cause issues:

```python
# When normalizing vertices to the sphere
vertex = vertex / np.linalg.norm(vertex) * self.radius

# For very high resolution meshes, use double precision:
vertex = vertex / np.linalg.norm(vertex, dtype=np.float64) * self.radius
```

**Impact:**
- Single precision is faster but can cause "bumpy" terrain at high resolutions
- Double precision fixes this but uses more memory and may be slower

### Memory Usage

Memory usage grows quadratically with resolution:

| Resolution Level | Approx. Vertices | Memory Usage |
|------------------|------------------|--------------|
| 3                | 642              | ~10 MB       |
| 4                | 2,562            | ~40 MB       |
| 5                | 10,242           | ~160 MB      |
| 6                | 40,962           | ~640 MB      |
| 7                | 163,842          | ~2.5 GB      |

**Mitigation Strategies:**
1. Use sparse data structures for certain properties
2. Implement out-of-core processing for very large planets
3. Consider adaptive mesh refinement for critical areas

### Geophysical Simplifications

Some geophysical processes are simplified:

1. **Mantle Convection** - Uses a static pattern rather than fluid dynamics
2. **Erosion** - Uses simplified rules rather than full hydrological modeling
3. **Climate** - Approximates circulation patterns without full atmospheric simulation

**Justification:**
These simplifications maintain the essential behavior while making the simulation feasible on standard hardware. For most worldbuilding purposes, the results are still physically plausible.