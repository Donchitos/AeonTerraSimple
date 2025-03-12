import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter

class TectonicSimulation:
    """
    A comprehensive simulation of planetary tectonic processes.
    """
    
    def __init__(self, planet, num_plates=12, initial_age=4500):
        """
        Initialize the tectonic simulation.
        
        Parameters:
        - planet: A Planet object
        - num_plates: Initial number of plates
        - initial_age: Age of the planet in million years when simulation starts
        """
        self.planet = planet
        self.num_plates = num_plates
        self.initial_age = initial_age
        
        # Set planet's initial age
        self.planet.age = initial_age
        
        # Plate properties storage
        self.plates = []
        
        # Initialize plate attributes in planet
        self.planet.plate_ids = np.full(len(planet.grid.vertices), -1)
        
        # Keep history for visualization
        self.history = {
            'ages': [],
            'plate_positions': [],
            'plate_velocities': [],
            'elevations': [],
            'plate_counts': [],
            'continental_area': []
        }
        
        # Enhanced physical properties
        self.crust_age = np.zeros(len(planet.grid.vertices))
        self.crust_thickness = np.zeros(len(planet.grid.vertices))
        self.crust_density = np.zeros(len(planet.grid.vertices))
        self.crust_type = np.zeros(len(planet.grid.vertices), dtype=int)  # 0=oceanic, 1=continental
        
        # Constants for Earth-like planet (all based on scientific measurements)
        self.CONTINENTAL_DENSITY = 2.7  # g/cm³
        self.OCEANIC_DENSITY = 3.0      # g/cm³
        self.AVG_CONTINENTAL_THICKNESS = 35.0  # km
        self.AVG_OCEANIC_THICKNESS = 7.0       # km
        self.MANTLE_DENSITY = 3.3       # g/cm³
        
        # Hotspot and plume tracking
        self.mantle_plumes = []
        self.hotspots = []
        
        # Tracking of specific geological features
        self.orogenic_belts = []  # Mountain-building regions
        self.rifts = []           # Divergent boundaries on continents
        self.volcanic_arcs = []   # Volcanic chains at subduction zones
        
        # Visualization params
        self.show_plates = True
    
    def initialize_mantle(self, num_cells=12, num_plumes=5):
        """
        Initialize mantle convection with both general cells and focused plumes.
        
        Parameters:
        - num_cells: Number of major convection cells
        - num_plumes: Number of concentrated mantle plumes
        """
        print("Initializing mantle convection system...")
        
        # Create random points on sphere for convection cell centers
        cell_centers = []
        for _ in range(num_cells):
            # Random point on unit sphere
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            cell_centers.append([x, y, z])
        
        # Define upwelling/downwelling for each cell
        self.convection_cells = []
        for center in cell_centers:
            # Randomly decide if this is upwelling or downwelling
            # In reality, about 60% of mantle is in downwelling mode
            is_upwelling = np.random.random() > 0.6
            
            # Create cell with strength inversely proportional to mantle viscosity
            strength = 0.005 * (1.0 + 0.5 * np.random.random())
            if not is_upwelling:
                strength = -strength
            
            self.convection_cells.append({
                'center': center,
                'is_upwelling': is_upwelling,
                'strength': strength
            })
        
        # Add focused mantle plumes (always upwelling)
        for _ in range(num_plumes):
            # Random position for plume
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            # Plumes are stronger but more localized
            plume_strength = 0.02 * (1.0 + 0.5 * np.random.random())
            radius = 0.2 + 0.3 * np.random.random()  # Smaller influence radius
            
            self.mantle_plumes.append({
                'center': [x, y, z],
                'strength': plume_strength,
                'radius': radius,
                'age': 0  # Plumes can die and new ones can form
            })
            
            # Create a hotspot at the plume location
            self.hotspots.append({
                'position': [x, y, z],
                'strength': plume_strength * 5,  # Heat flux
                'age': 0,
                'track': []  # Will store positions of islands created over time
            })
        
        # Calculate force field from these cells
        self._calculate_mantle_forces()
        
        return self
        
    def _calculate_mantle_forces(self):
        """Calculate mantle convection forces at each vertex"""
        # Initialize force vectors
        self.mantle_forces = np.zeros((len(self.planet.grid.vertices), 3))
        
        # For each vertex, calculate combined force from all convection cells
        for i, vertex in enumerate(self.planet.grid.vertices):
            # Normalize vertex
            pos = vertex / np.linalg.norm(vertex)
            
            # Sum forces from all cells
            force = np.zeros(3)
            
            # Add forces from convection cells
            for cell in self.convection_cells:
                # Vector from cell center to vertex (on unit sphere)
                center = np.array(cell['center'])
                center = center / np.linalg.norm(center)
                
                # Calculate force based on distance
                vec_to_center = center - pos
                dist = np.linalg.norm(vec_to_center)
                
                # Skip if too far
                if dist > 2.0:
                    continue
                
                # Closer = stronger force
                strength = cell['strength'] * (1.0 - dist/2.0)
                
                # Force direction depends on upwelling/downwelling
                # Upwelling: Force pushes away from center
                # Downwelling: Force pulls toward center
                if cell['is_upwelling']:
                    force -= vec_to_center * strength
                else:
                    force += vec_to_center * strength
            
            # Add forces from mantle plumes
            for plume in self.mantle_plumes:
                # Vector from plume center to vertex
                center = np.array(plume['center'])
                center = center / np.linalg.norm(center)
                
                # Calculate force based on distance
                vec_to_center = center - pos
                dist = np.linalg.norm(vec_to_center)
                
                # Skip if outside plume radius
                if dist > plume['radius']:
                    continue
                
                # Plumes always push upward (away from center)
                # Strength decreases with distance from center
                plume_strength = plume['strength'] * (1.0 - dist/plume['radius'])
                force -= vec_to_center * plume_strength
            
            # Store force vector
            self.mantle_forces[i] = force
    
    def _generate_coherent_noise(self, octaves=4, persistence=0.5, scale=1.0):
        """Generate coherent noise on the sphere using multiple frequencies"""
        # Start with random noise
        noise = np.random.random(len(self.planet.grid.vertices))
        
        # Smooth the noise 
        # We'll approximate coherent noise by applying Gaussian blur
        # to a sphere-mapped texture multiple times
        
        # Convert to lat-lon grid for smoothing
        grid_size = int(np.sqrt(len(self.planet.grid.vertices))) * 2
        grid = np.zeros((grid_size, grid_size))
        
        # Map vertices to grid
        for i, vertex in enumerate(self.planet.grid.vertices):
            # Convert to lat-lon coordinates
            x, y, z = vertex
            lat = np.arcsin(z / np.linalg.norm(vertex))
            lon = np.arctan2(y, x)
            
            # Map to grid coordinates
            grid_x = int((lon + np.pi) / (2 * np.pi) * grid_size) % grid_size
            grid_y = int((lat + np.pi/2) / np.pi * grid_size/2) % grid_size
            
            # Set value
            grid[grid_y, grid_x] = noise[i]
        
        # Apply multiple octaves of smoothing
        final_grid = np.zeros_like(grid)
        amplitude = 1.0
        max_value = 0.0
        
        for i in range(octaves):
            # Smooth the grid
            sigma = scale * (2 ** (octaves - i - 1))
            smoothed = gaussian_filter(grid, sigma=sigma)
            
            # Add to final grid
            final_grid += smoothed * amplitude
            
            # Update amplitude for next octave
            max_value += amplitude
            amplitude *= persistence
        
        # Normalize
        final_grid /= max_value
        
        # Map back to vertices
        coherent_noise = np.zeros(len(self.planet.grid.vertices))
        for i, vertex in enumerate(self.planet.grid.vertices):
            x, y, z = vertex
            lat = np.arcsin(z / np.linalg.norm(vertex))
            lon = np.arctan2(y, x)
            
            grid_x = int((lon + np.pi) / (2 * np.pi) * grid_size) % grid_size
            grid_y = int((lat + np.pi/2) / np.pi * grid_size/2) % grid_size
            
            coherent_noise[i] = final_grid[grid_y, grid_x]
        
        return coherent_noise
    
    def initialize_plates(self):
        """Create initial plates using a more realistic model of early crust formation"""
        print("Initializing tectonic plates...")
        
        # Step 1: Create initial crustal thickness variations
        # This simulates the early formation of the crust with thickness variations
        thickness_noise = self._generate_coherent_noise(octaves=4)
        
        # Apply scientific constraints to thickness
        # Continental crust: 25-70 km thick
        # Oceanic crust: 5-10 km thick
        
        # Use a bimodal distribution to create distinct continental and oceanic crust
        continental_mask = thickness_noise > 0.65  # About 35% is continental (roughly like Earth)
        
        # Set thickness based on crust type
        self.crust_thickness = np.zeros(len(self.planet.grid.vertices))
        
        # Continental crust (25-70 km, averaging around 35 km)
        continental_thickness = self.AVG_CONTINENTAL_THICKNESS + 15 * (thickness_noise[continental_mask] - 0.65) / 0.35
        self.crust_thickness[continental_mask] = continental_thickness
        
        # Oceanic crust (5-10 km, averaging around 7 km)
        oceanic_thickness = self.AVG_OCEANIC_THICKNESS + 2 * thickness_noise[~continental_mask]
        self.crust_thickness[~continental_mask] = oceanic_thickness
        
        # Set crust type and density
        self.crust_type[continental_mask] = 1  # Continental
        self.crust_type[~continental_mask] = 0  # Oceanic
        
        # Density based on type (continental: ~2.7 g/cm³, oceanic: ~3.0 g/cm³)
        self.crust_density[continental_mask] = self.CONTINENTAL_DENSITY
        self.crust_density[~continental_mask] = self.OCEANIC_DENSITY
        
        # Initial crust age - continental crust is much older than oceanic
        self.crust_age[continental_mask] = self.initial_age * 0.8 + self.initial_age * 0.2 * np.random.random(np.sum(continental_mask))
        self.crust_age[~continental_mask] = 200 * np.random.random(np.sum(~continental_mask))  # Ocean crust younger than 200 Ma
        
        # Step 2: Define initial plate boundaries using Voronoi cells on the sphere
        self._generate_plate_boundaries()
        
        # Step 3: Initialize plate properties
        self._initialize_plate_properties()
        
        # Step 4: Set initial elevation based on isostatic equilibrium
        self._calculate_isostatic_elevation()
        
        # Record initial state in history
        self._record_state()
        
        print(f"Initialized {len(self.plates)} tectonic plates")
        return self
        
    def _generate_plate_boundaries(self):
        """Generate plate boundaries using Voronoi cells on the sphere"""
        # Step 1: Choose random seed points on the sphere
        seed_points = []
        for _ in range(self.num_plates):
            # Random point on unit sphere
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            seed_points.append([x, y, z])
        
        # Step 2: For each vertex, find closest seed point
        for i, vertex in enumerate(self.planet.grid.vertices):
            min_dist = float('inf')
            closest_plate = -1
            
            for j, seed in enumerate(seed_points):
                # Distance on the sphere (use arc length, not Euclidean)
                dot_product = np.clip(np.dot(vertex, seed) / 
                                      (np.linalg.norm(vertex) * np.linalg.norm(seed)), -1.0, 1.0)
                arc_dist = np.arccos(dot_product)
                
                if arc_dist < min_dist:
                    min_dist = arc_dist
                    closest_plate = j
            
            # Assign vertex to plate
            self.planet.plate_ids[i] = closest_plate
            
        # Step 3: Calculate plate boundaries
        # We'll identify vertices at plate boundaries for later use
        is_boundary = np.zeros(len(self.planet.grid.vertices), dtype=bool)
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        for i, neighbor_list in enumerate(neighbors):
            plate_id = self.planet.plate_ids[i]
            for neighbor in neighbor_list:
                if self.planet.plate_ids[neighbor] != plate_id:
                    is_boundary[i] = True
                    break
        
        self.plate_boundaries = is_boundary
    
    def _initialize_plate_properties(self):
        """Initialize properties for each plate with scientific parameters"""
        self.plates = []
        for i in range(self.num_plates):
            # Get all vertices in this plate
            plate_vertices = np.where(self.planet.plate_ids == i)[0]
            
            # Skip if plate is empty
            if len(plate_vertices) == 0:
                continue
            
            # Calculate center of mass
            center_pos = np.zeros(3)
            for vertex_idx in plate_vertices:
                center_pos += self.planet.grid.vertices[vertex_idx]
            center_pos /= len(plate_vertices)
            center_pos = center_pos / np.linalg.norm(center_pos)  # Normalize to sphere
            
            # Determine if plate is primarily oceanic or continental
            # This is based on the dominant crust type within the plate
            continental_crust_count = np.sum(self.crust_type[plate_vertices] == 1)
            is_oceanic = continental_crust_count < len(plate_vertices) / 2
            
            # Calculate plate velocity based on mantle forces
            plate_force = np.zeros(3)
            for vertex_idx in plate_vertices:
                plate_force += self.mantle_forces[vertex_idx]
            plate_force /= len(plate_vertices)
            
            # Add a small random component
            rand_component = np.random.normal(0, 0.002, 3)
            plate_force += rand_component
            
            # Create plate object with enhanced properties
            plate = {
                'id': i,
                'center': center_pos,
                'vertices': plate_vertices,
                'is_oceanic': is_oceanic,
                # Oceanic plates move faster (2-10 cm/yr vs 1-5 cm/yr for continental)
                'velocity': plate_force * (1.5 if is_oceanic else 0.8),
                'area': len(plate_vertices) / len(self.planet.grid.vertices),
                'age': self.initial_age,
                'continental_percentage': continental_crust_count / len(plate_vertices) * 100,
                'boundaries': [],
                'features': {
                    'orogenic_belts': [],  # Mountain ranges
                    'rifts': [],           # Spreading centers
                    'volcanic_arcs': [],   # Subduction volcanoes
                    'hotspots': []         # Intraplate volcanic centers
                }
            }
            
            self.plates.append(plate)
        
        # Update plate boundaries
        self._update_plate_boundaries()
    
    def _update_plate_boundaries(self):
        """Update the list of plate boundaries"""
        self.plate_boundaries = np.zeros(len(self.planet.grid.vertices), dtype=bool)
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        for i, neighbor_list in enumerate(neighbors):
            plate_id = self.planet.plate_ids[i]
            for neighbor in neighbor_list:
                if self.planet.plate_ids[neighbor] != plate_id:
                    self.plate_boundaries[i] = True
                    break
                    
    def _calculate_isostatic_elevation(self):
        """
        Calculate elevation based on isostatic equilibrium.
        
        This follows Airy isostasy model where thicker/less dense crust
        floats higher on the mantle, like blocks of wood on water.
        """
        # Calculate elevation based on density and thickness (isostatic compensation)
        # Formula derived from Archimedes' principle
        for i in range(len(self.planet.grid.vertices)):
            if self.crust_type[i] == 1:  # Continental
                # Continental crust rises higher due to lower density
                # Isostatic equation: e = h * (ρm - ρc)/ρm
                elevation = self.crust_thickness[i] * (self.MANTLE_DENSITY - self.crust_density[i]) / self.MANTLE_DENSITY
                
                # Adjust to Earth-like elevations (continents average ~0.8 km above sea level)
                elevation = elevation * 0.05 - 1.0
                
                # Add local variation based on thickness
                # Shields (thick old continental crust) are often higher
                thickness_variation = (self.crust_thickness[i] - self.AVG_CONTINENTAL_THICKNESS) / 20.0
                elevation += thickness_variation
                
                self.planet.elevation[i] = elevation
            else:  # Oceanic
                # Oceanic crust starts lower due to higher density
                # Isostatic equation adjusted for sea floor elevation
                elevation = self.crust_thickness[i] * (self.MANTLE_DENSITY - self.crust_density[i]) / self.MANTLE_DENSITY
                
                # Adjust to Earth-like elevations (ocean floor averages ~3.7 km below sea level)
                elevation = elevation * 0.05 - 4.5
                
                # Age-based subsidence: as oceanic crust ages, it cools and sinks
                # This follows the square root of age (GDH1 thermal subsidence model)
                age_factor = np.sqrt(self.crust_age[i] / 100.0) if self.crust_age[i] > 0 else 0
                subsidence = -2.0 * age_factor
                
                elevation += subsidence
                elevation = max(elevation, -11.0)  # Maximum ocean depth constraint
                
                self.planet.elevation[i] = elevation
        
        # Add local terrain variation
        terrain_noise = self._generate_coherent_noise(octaves=6, scale=0.5) * 0.5
        
        # Apply noise, more to land than ocean (erosion creates more varied terrain on land)
        for i in range(len(self.planet.elevation)):
            if self.planet.elevation[i] >= 0:
                # Land gets more variation
                self.planet.elevation[i] += terrain_noise[i] * 0.8
            else:
                # Ocean gets less variation
                self.planet.elevation[i] += terrain_noise[i] * 0.2
                
    def simulate_step(self, time_step=10.0):
        """
        Simulate one step of tectonic plate movement with enhanced scientific accuracy.
        
        Parameters:
        - time_step: Time in million years to simulate
        """
        print(f"Simulating plate tectonics from year {self.planet.age} to {self.planet.age + time_step} million...")
        
        # Step 1: Move plates based on velocities
        self._move_plates(time_step)
        
        # Step 2: Handle crust creation and destruction
        self._handle_crust_processes(time_step)
        
        # Step 3: Update plate properties and boundaries
        self._update_plate_motion()
        
        # Step 4: Process hotspots and plumes
        self._process_mantle_plumes(time_step)
        
        # Step 5: Update elevation based on isostatic equilibrium
        self._calculate_isostatic_elevation()
        
        # Step 6: Age the crust and apply subsidence
        self._age_crust(time_step)
        
        # Step 7: Check for plate reorganization events
        if np.random.random() < 0.1:  # 10% chance per step
            self._reorganize_plates()
        
        # Update planet age
        self.planet.age += time_step
        
        # Record state for history
        self._record_state()
        
        return self
    
    def _move_plates(self, time_step):
        """Move plates based on velocity and time step"""
        # Convert time_step from million years to simulation units
        # Assuming 1.0 velocity = 5cm/year
        movement_scale = time_step * 0.05  # 5cm/year * million years
        
        # For each plate, calculate rotation matrix
        for plate in self.plates:
            # Get the axis of rotation (perpendicular to velocity)
            velocity = np.array(plate['velocity'])
            center = np.array(plate['center'])
            
            # Skip if velocity is too small
            if np.linalg.norm(velocity) < 1e-6:
                continue
            
            # Calculate axis of rotation (cross product of center and velocity)
            axis = np.cross(center, velocity)
            axis = axis / np.linalg.norm(axis)
            
            # Calculate angle of rotation based on velocity magnitude and time step
            angle = np.linalg.norm(velocity) * movement_scale
            
            # Create rotation matrix (Rodrigues' rotation formula)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            
            # Apply rotation to all vertices in the plate
            for vertex_idx in plate['vertices']:
                vertex = self.planet.grid.vertices[vertex_idx]
                rotated_vertex = R @ vertex
                self.planet.grid.vertices[vertex_idx] = rotated_vertex
        
        # Renormalize all vertices to the sphere surface
        for i in range(len(self.planet.grid.vertices)):
            self.planet.grid.vertices[i] = self.planet.grid.vertices[i] / np.linalg.norm(self.planet.grid.vertices[i]) * self.planet.radius
    
    def _handle_crust_processes(self, time_step):
        """Handle crust creation and destruction processes with scientific accuracy"""
        # Process boundaries to determine plate interactions
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        # Step 1: Find all boundary vertices
        boundary_vertices = np.where(self.plate_boundaries)[0]
        
        # Step 2: Process each boundary vertex
        for i in boundary_vertices:
            my_plate_id = self.planet.plate_ids[i]
            
            # Skip if vertex isn't assigned to a plate
            if my_plate_id == -1 or my_plate_id >= len(self.plates):
                continue
                
            # Find neighboring plates
            neighbor_plate_ids = set()
            for neighbor in neighbors[i]:
                neighbor_id = self.planet.plate_ids[neighbor]
                if neighbor_id != my_plate_id and neighbor_id != -1 and neighbor_id < len(self.plates):
                    neighbor_plate_ids.add(neighbor_id)
            
            # Process each plate interaction
            for neighbor_plate_id in neighbor_plate_ids:
                # Get both plates
                my_plate = self.plates[my_plate_id]
                neighbor_plate = self.plates[neighbor_plate_id]
                
                # Calculate relative motion
                rel_velocity = my_plate['velocity'] - neighbor_plate['velocity']
                
                # Get normal vector to boundary
                normal = self.planet.grid.vertices[i] / np.linalg.norm(self.planet.grid.vertices[i])
                
                # Dot product tells us if plates are converging or diverging
                convergence = -np.dot(rel_velocity, normal)
                
                # Handle different boundary types
                if convergence > 0.001:  # Convergent boundary
                    self._handle_convergent_boundary(i, my_plate_id, neighbor_plate_id, time_step)
                elif convergence < -0.001:  # Divergent boundary
                    self._handle_divergent_boundary(i, my_plate_id, neighbor_plate_id, time_step)
                else:  # Transform boundary
                    self._handle_transform_boundary(i, my_plate_id, neighbor_plate_id, time_step)
    
    def _handle_convergent_boundary(self, vertex_idx, plate1_id, plate2_id, time_step):
        """
        Handle convergent plate boundary processes with scientific accuracy
        
        Parameters:
        - vertex_idx: Index of boundary vertex
        - plate1_id, plate2_id: IDs of interacting plates
        - time_step: Time step in million years
        """
        # Get plate info
        plate1 = self.plates[plate1_id]
        plate2 = self.plates[plate2_id]
        
        # Get crust type at this location
        is_continental1 = self.crust_type[vertex_idx] == 1
        
        # Check neighboring vertices for their crust type
        neighbors = self.planet.grid.get_vertex_neighbors()
        neighbor_vertices = neighbors[vertex_idx]
        
        # Find a neighbor from the other plate
        other_plate_vertex = None
        for neighbor in neighbor_vertices:
            if self.planet.plate_ids[neighbor] == plate2_id:
                other_plate_vertex = neighbor
                break
        
        if other_plate_vertex is None:
            return
            
        is_continental2 = self.crust_type[other_plate_vertex] == 1
        
        # Handle different convergence scenarios
        
        # Case 1: Continental-continental collision
        if is_continental1 and is_continental2:
            # Continental collision creates mountain ranges
            # Increase elevation by a rate proportional to convergence
            # Typically 0.5-1 cm/year or 5-10 km per million years
            elevation_increase = 5.0 + 5.0 * np.random.random()  # km per million years
            elevation_increase *= time_step / 10.0  # Scale by time step
            self.planet.elevation[vertex_idx] += elevation_increase
            
            # Thicken the crust (due to compression)
            # Continental crust can double in thickness at collision zones
            self.crust_thickness[vertex_idx] += 2.0 * time_step / 10.0
            
            # Add to orogenic belts (mountain building regions)
            if plate1_id not in plate2['features']['orogenic_belts']:
                plate1['features']['orogenic_belts'].append(plate2_id)
            if plate2_id not in plate1['features']['orogenic_belts']:
                plate2['features']['orogenic_belts'].append(plate1_id)
        
        # Case 2: Oceanic-continental subduction
        elif (not is_continental1 and is_continental2) or (is_continental1 and not is_continental2):
            # Determine subducting and overriding plates
            if (not is_continental1 and is_continental2):
                subducting_plate = plate1
                overriding_plate = plate2
                overriding_vertex = other_plate_vertex
            else:
                subducting_plate = plate2
                overriding_plate = plate1
                overriding_vertex = vertex_idx
            
            # Create coastal mountains on overriding plate
            # Typically 0.3-0.7 cm/year or 3-7 km per million years
            if self.planet.elevation[overriding_vertex] >= 0:
                elevation_increase = 3.0 + 4.0 * np.random.random()  # km per million years
                elevation_increase *= time_step / 10.0  # Scale by time step
                self.planet.elevation[overriding_vertex] += elevation_increase
            
            # Create trench on subducting plate
            # Trenches are typically 8-11 km deep
            if self.planet.elevation[vertex_idx] < 0:
                # Deepen the trench
                self.planet.elevation[vertex_idx] = max(-11.0, self.planet.elevation[vertex_idx] - 1.0 * time_step / 10.0)
            
            # Add volcanic arc to features
            if subducting_plate['id'] not in overriding_plate['features']['volcanic_arcs']:
                overriding_plate['features']['volcanic_arcs'].append(subducting_plate['id'])
        
        # Case 3: Oceanic-oceanic subduction
        elif not is_continental1 and not is_continental2:
            # Older plate subducts under younger plate
            age1 = self.crust_age[vertex_idx]
            age2 = self.crust_age[other_plate_vertex]
            
            if age1 > age2:
                # Plate 1 subducts
                self.planet.elevation[vertex_idx] -= 1.0 * time_step / 10.0
                
                # Chance of island arc formation on plate 2
                if np.random.random() < 0.05 * time_step / 10.0:  # 5% chance per 10 My
                    self.planet.elevation[other_plate_vertex] = max(0, self.planet.elevation[other_plate_vertex])
                    self.crust_type[other_plate_vertex] = 1  # Convert to continental crust
                    self.crust_thickness[other_plate_vertex] = 15.0  # Initial island arc thickness
                    self.crust_density[other_plate_vertex] = self.CONTINENTAL_DENSITY
            else:
                # Plate 2 subducts
                self.planet.elevation[other_plate_vertex] -= 1.0 * time_step / 10.0
                
                # Chance of island arc formation on plate 1
                if np.random.random() < 0.05 * time_step / 10.0:  # 5% chance per 10 My
                    self.planet.elevation[vertex_idx] = max(0, self.planet.elevation[vertex_idx])
                    self.crust_type[vertex_idx] = 1  # Convert to continental crust
                    self.crust_thickness[vertex_idx] = 15.0  # Initial island arc thickness
                    self.crust_density[vertex_idx] = self.CONTINENTAL_DENSITY
    
    def _handle_divergent_boundary(self, vertex_idx, plate1_id, plate2_id, time_step):
        """
        Handle divergent plate boundary processes with scientific accuracy
        
        Parameters:
        - vertex_idx: Index of boundary vertex
        - plate1_id, plate2_id: IDs of interacting plates
        - time_step: Time step in million years
        """
        # Get crust type at this location
        is_continental = self.crust_type[vertex_idx] == 1
        
        # Case 1: Continental rifting
        if is_continental:
            # Continental rifting thins the crust
            # Rate: about 2-3 km per million years
            thinning_rate = 2.5 * time_step / 10.0
            self.crust_thickness[vertex_idx] = max(self.AVG_OCEANIC_THICKNESS, 
                                                  self.crust_thickness[vertex_idx] - thinning_rate)
            
            # Lower elevation due to crustal thinning
            # Rate: about 0.5-1 km per million years
            subsidence_rate = 0.75 * time_step / 10.0
            self.planet.elevation[vertex_idx] -= subsidence_rate
            
            # If crust thins enough, it transitions to oceanic
            if self.crust_thickness[vertex_idx] <= 15.0:
                # Transition to oceanic conditions
                if np.random.random() < 0.2 * time_step / 10.0:  # 20% chance per 10 My
                    self.crust_type[vertex_idx] = 0  # Convert to oceanic
                    self.crust_density[vertex_idx] = self.OCEANIC_DENSITY
                    self.crust_age[vertex_idx] = 0  # New crust
                    self.planet.elevation[vertex_idx] = -1.5  # Initial seafloor depth
                    
                    # Add to rifts feature
                    if plate1_id not in self.plates[plate2_id]['features']['rifts']:
                        self.plates[plate2_id]['features']['rifts'].append(plate1_id)
                    if plate2_id not in self.plates[plate1_id]['features']['rifts']:
                        self.plates[plate1_id]['features']['rifts'].append(plate2_id)
        
        # Case 2: Seafloor spreading
        else:
            # Create new oceanic crust at boundary
            self.crust_age[vertex_idx] = 0  # Brand new crust
            self.crust_thickness[vertex_idx] = self.AVG_OCEANIC_THICKNESS + 1.0 * np.random.random()
            
            # New seafloor is higher due to thermal buoyancy
            # Mid-ocean ridges are typically 2-3 km above surrounding seafloor
            self.planet.elevation[vertex_idx] = -2.0 + 0.5 * np.random.random()
    
    def _handle_transform_boundary(self, vertex_idx, plate1_id, plate2_id, time_step):
        """
        Handle transform plate boundary processes
        
        Parameters:
        - vertex_idx: Index of boundary vertex
        - plate1_id, plate2_id: IDs of interacting plates
        - time_step: Time step in million years
        """
        # Transform boundaries primarily create faults without major elevation changes
        # Occasionally they can create pull-apart basins or push-up ranges
        
        # Small chance of minor terrain features
        if np.random.random() < 0.1 * time_step / 10.0:  # 10% chance per 10 My
            feature_type = np.random.random()
            
            if feature_type < 0.5:  # Pull-apart basin (50% chance)
                # Small depression (typically 1-2 km deep)
                self.planet.elevation[vertex_idx] -= 0.5 * np.random.random()
            else:  # Push-up range (50% chance)
                # Small uplift (typically 0.5-1 km high)
                self.planet.elevation[vertex_idx] += 0.3 * np.random.random()
                
    def _update_plate_motion(self):
        """Update plate motion vectors based on forces and constraints"""
        # Update center of mass for each plate
        for plate in self.plates:
            center_pos = np.zeros(3)
            for vertex_idx in plate['vertices']:
                center_pos += self.planet.grid.vertices[vertex_idx]
            center_pos /= len(plate['vertices'])
            plate['center'] = center_pos / np.linalg.norm(center_pos)
        
        # Apply constraints to plate motions
        for plate in self.plates:
            # Apply damping (plate motion slows over time)
            plate['velocity'] *= 0.9
            
            # Apply mantle forces
            plate_force = np.zeros(3)
            for vertex_idx in plate['vertices']:
                plate_force += self.mantle_forces[vertex_idx]
            plate_force /= len(plate['vertices'])
            
            plate['velocity'] += plate_force * 0.1
            
            # Ensure velocity is tangent to surface
            center = plate['center']
            vel_dot_center = np.dot(plate['velocity'], center)
            plate['velocity'] -= vel_dot_center * center
    
    def _process_mantle_plumes(self, time_step):
        """Process mantle plumes and hotspot volcanism with scientific accuracy"""
        # Plumes are relatively fixed in the mantle
        # Hotspots remain in place while plates move over them
        
        # Update each hotspot
        for hotspot in self.hotspots:
            hotspot['age'] += time_step
            
            # Find vertices near this hotspot
            for i, vertex in enumerate(self.planet.grid.vertices):
                # Distance from hotspot to vertex
                dist = np.arccos(np.clip(np.dot(hotspot['position'], vertex) / 
                              (np.linalg.norm(hotspot['position']) * np.linalg.norm(vertex)), -1.0, 1.0))
                
                # If vertex is close to hotspot, create volcanic features
                if dist < 0.1:  # About 600 km at Earth's radius
                    # Higher chance of volcanism for stronger hotspots and vertices directly above
                    volcano_chance = hotspot['strength'] * (1.0 - dist/0.1) * 0.1
                    
                    if np.random.random() < volcano_chance * time_step / 10.0:
                        # Create volcanic feature
                        if self.planet.elevation[i] < 0:
                            # Undersea volcano - can form island
                            if np.random.random() < 0.5:  # 50% chance of breaching surface
                                self.planet.elevation[i] = 0.5 + 1.0 * np.random.random()  # Island with 0.5-1.5 km elevation
                                
                                # Islands have thin continental crust
                                self.crust_type[i] = 1
                                self.crust_thickness[i] = 15.0 + 5.0 * np.random.random()
                                self.crust_density[i] = self.CONTINENTAL_DENSITY
                                
                                # Record island position in hotspot track
                                hotspot['track'].append(list(vertex))
                            else:
                                # Seamount (underwater volcano)
                                self.planet.elevation[i] += 1.0 + 2.0 * np.random.random()
                        else:
                            # Continental volcano - increase elevation
                            self.planet.elevation[i] += 0.5 + 1.0 * np.random.random()
        
        # Occasional new plumes form and old ones die
        if np.random.random() < 0.05 * time_step / 10.0:  # 5% chance per 10 My
            # Create a new plume
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            # Plumes are stronger but more localized
            plume_strength = 0.02 * (1.0 + 0.5 * np.random.random())
            radius = 0.2 + 0.3 * np.random.random()
            
            self.mantle_plumes.append({
                'center': [x, y, z],
                'strength': plume_strength,
                'radius': radius,
                'age': 0
            })
            
            # Create a hotspot at the plume location
            self.hotspots.append({
                'position': [x, y, z],
                'strength': plume_strength * 5,
                'age': 0,
                'track': []
            })
            
            print(f"New mantle plume formed at {self.planet.age + time_step} million years.")
        
        # Old plumes can die out
        for i in range(len(self.mantle_plumes) - 1, -1, -1):
            plume = self.mantle_plumes[i]
            plume['age'] += time_step
            
            if plume['age'] > 100:  # Plumes typically last 100-200 Myr
                # 5% chance of dying per 10 My after reaching 100 My age
                if np.random.random() < 0.05 * time_step / 10.0:
                    # Remove the plume and its hotspot
                    del self.mantle_plumes[i]
                    del self.hotspots[i]
                    print(f"Mantle plume died at {self.planet.age + time_step} million years.")
    
    def _age_crust(self, time_step):
        """
        Age the crust and apply scientific processes related to aging
        
        Parameters:
        - time_step: Time step in million years
        """
        # Age all crust
        self.crust_age += time_step
        
        # Process oceanic crust - it subducts after about 180-200 My
        # In reality, the oldest oceanic crust on Earth is only about 180-200 million years old
        for i in range(len(self.crust_age)):
            if self.crust_type[i] == 0:  # Oceanic crust
                if self.crust_age[i] > 180 and np.random.random() < 0.1 * time_step / 10.0:
                    # Very old oceanic crust has a chance to begin subduction
                    # This creates new trenches and can initiate new subduction zones
                    self.planet.elevation[i] -= 1.0 * np.random.random()
        
        # New crust at divergent boundaries remains young
        for i, is_boundary in enumerate(self.plate_boundaries):
            if not is_boundary:
                continue
                
            plate_id = self.planet.plate_ids[i]
            
            # Skip if vertex isn't assigned to a plate
            if plate_id == -1 or plate_id >= len(self.plates):
                continue
                
            plate = self.plates[plate_id]
            
            # Check if this is at a divergent boundary
            for boundary in plate['boundaries']:
                if 'vertex' in boundary and boundary['vertex'] == i and boundary['type'] == 'divergent':
                    if 'subtype' in boundary and boundary['subtype'] == 'oceanic-oceanic':
                        # Fresh oceanic crust at mid-ocean ridges
                        self.crust_age[i] = 0
                        self.crust_thickness[i] = self.AVG_OCEANIC_THICKNESS + np.random.random() * 2.0
    
    def _reorganize_plates(self):
        """Handle major plate reorganization events"""
        # Possibility 1: Break up a large plate
        for plate in self.plates:
            # Only consider large plates
            if plate['area'] > 0.25:  # 25% of planet surface
                # 30% chance to break up
                if np.random.random() < 0.3:
                    print(f"Major tectonic event: Breaking up large plate {plate['id']}")
                    self._break_plate(plate)
                    return  # Only one major event per step
        
        # Possibility 2: Merge small plates
        small_plates = [p for p in self.plates if p['area'] < 0.05]
        if len(small_plates) >= 2:
            # 20% chance to merge
            if np.random.random() < 0.2:
                # Choose two small plates to merge
                p1, p2 = np.random.choice(small_plates, 2, replace=False)
                print(f"Major tectonic event: Merging small plates {p1['id']} and {p2['id']}")
                self._merge_plates(p1, p2)
                return  # Only one major event per step
    
    def _break_plate(self, plate):
        """Break a large plate into two smaller plates"""
        # Create a new plate ID
        new_plate_id = len(self.plates)
        
        # Decide where to split (random great circle)
        # Choose a random axis as the splitting plane
        split_axis = np.random.normal(0, 1, 3)
        split_axis = split_axis / np.linalg.norm(split_axis)
        
        # Assign vertices to new plate if they're on one side of the plane
        new_plate_vertices = []
        for vertex_idx in plate['vertices']:
            vertex = self.planet.grid.vertices[vertex_idx]
            # Check which side of the plane the vertex is on
            if np.dot(vertex, split_axis) > 0:
                self.planet.plate_ids[vertex_idx] = new_plate_id
                new_plate_vertices.append(vertex_idx)
        
        # Remove these vertices from the original plate
        plate['vertices'] = [v for v in plate['vertices'] if v not in new_plate_vertices]
        
        # Update plate area
        plate['area'] = len(plate['vertices']) / len(self.planet.grid.vertices)
        
        # Create the new plate
        # Calculate center
        center_pos = np.zeros(3)
        for vertex_idx in new_plate_vertices:
            center_pos += self.planet.grid.vertices[vertex_idx]
        center_pos /= len(new_plate_vertices)
        center_pos = center_pos / np.linalg.norm(center_pos)
        
        # Determine if plate is oceanic (based on crust type)
        continental_count = sum(1 for v in new_plate_vertices if self.crust_type[v] == 1)
        is_oceanic = continental_count < len(new_plate_vertices) / 2
        
        # Create velocity similar to parent but with some deviation
        parent_vel = plate['velocity']
        new_vel = parent_vel + np.random.normal(0, 0.002, 3)
        
        # Create the new plate
        new_plate = {
            'id': new_plate_id,
            'center': center_pos,
            'vertices': new_plate_vertices,
            'is_oceanic': is_oceanic,
            'velocity': new_vel,
            'area': len(new_plate_vertices) / len(self.planet.grid.vertices),
            'age': plate['age'],
            'continental_percentage': continental_count / len(new_plate_vertices) * 100,
            'boundaries': [],
            'features': {
                'orogenic_belts': [],  # Mountain ranges
                'rifts': [],           # Spreading centers
                'volcanic_arcs': [],   # Subduction volcanoes
                'hotspots': []         # Intraplate volcanic centers
            }
        }
        
        self.plates.append(new_plate)
        
        # Update boundaries
        self._update_plate_boundaries()
    
    def _merge_plates(self, plate1, plate2):
        """Merge two plates into one"""
        # Keep the first plate, reassign vertices from second plate
        for vertex_idx in plate2['vertices']:
            self.planet.plate_ids[vertex_idx] = plate1['id']
            plate1['vertices'].append(vertex_idx)
        
        # Update area
        plate1['area'] = len(plate1['vertices']) / len(self.planet.grid.vertices)
        
        # Update velocity (weighted average)
        total_weight = plate1['area'] + plate2['area']
        plate1['velocity'] = (plate1['velocity'] * plate1['area'] + 
                              plate2['velocity'] * plate2['area']) / total_weight
        
        # Determine if merged plate is oceanic based on majority
        continental_count = sum(1 for v in plate1['vertices'] if self.crust_type[v] == 1)
        plate1['is_oceanic'] = continental_count < len(plate1['vertices']) / 2
        plate1['continental_percentage'] = continental_count / len(plate1['vertices']) * 100
        
        # Remove plate2 from the list
        self.plates.remove(plate2)
        
        # Update IDs for all plates
        for i, plate in enumerate(self.plates):
            plate['id'] = i
            for vertex_idx in plate['vertices']:
                self.planet.plate_ids[vertex_idx] = i
        
        # Update boundaries
        self._update_plate_boundaries()
    
    def _record_state(self):
        """Record current state for history with enhanced statistics"""
        self.history['ages'].append(self.planet.age)
        
        # Record plate positions
        positions = []
        for plate in self.plates:
            positions.append(plate['center'])
        self.history['plate_positions'].append(positions)
        
        # Record plate velocities
        velocities = []
        for plate in self.plates:
            velocities.append(plate['velocity'])
        self.history['plate_velocities'].append(velocities)
        
        # Record elevation
        self.history['elevations'].append(np.copy(self.planet.elevation))
        
        # Record plate count
        self.history['plate_counts'].append(len(self.plates))
        
        # Record continental area percentage
        continental_area = 100 * np.sum(self.crust_type == 1) / len(self.crust_type)
        self.history['continental_area'].append(continental_area)
    
    def visualize_history(self, save_path=None):
        """
        Visualize the tectonic plate history over time.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        """
        if len(self.history['ages']) < 2:
            print("Not enough history data to visualize")
            return
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Plate count over time
        axs[0, 0].plot(self.history['ages'], self.history['plate_counts'])
        axs[0, 0].set_xlabel('Age (million years)')
        axs[0, 0].set_ylabel('Number of Plates')
        axs[0, 0].set_title('Plate Count Over Time')
        
        # Plot 2: Elevation distribution over time
        times = np.array(self.history['ages'])
        times_idx = np.linspace(0, len(times)-1, min(10, len(times))).astype(int)
        
        for idx in times_idx:
            elevations = self.history['elevations'][idx]
            hist, bins = np.histogram(elevations, bins=50, range=(-10, 10))
            axs[0, 1].plot(bins[:-1], hist, label=f'{times[idx]}Ma')
        
        axs[0, 1].set_xlabel('Elevation (km)')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Elevation Distribution Over Time')
        axs[0, 1].legend()
        
        # Plot 3: Plate velocity magnitudes
        vel_magnitudes = []
        for velocities in self.history['plate_velocities']:
            magnitudes = [np.linalg.norm(v) for v in velocities]
            vel_magnitudes.append(np.mean(magnitudes))
        
        axs[1, 0].plot(self.history['ages'], vel_magnitudes)
        axs[1, 0].set_xlabel('Age (million years)')
        axs[1, 0].set_ylabel('Average Plate Velocity')
        axs[1, 0].set_title('Plate Velocities Over Time')
        
        # Plot 4: Continent vs Ocean area
        if 'continental_area' in self.history:
            axs[1, 1].plot(self.history['ages'], self.history['continental_area'], label='Continental')
            axs[1, 1].plot(self.history['ages'], [100 - x for x in self.history['continental_area']], label='Oceanic')
        else:
            land_area = []
            ocean_area = []
            
            for elevations in self.history['elevations']:
                land = np.sum(elevations >= 0) / len(elevations) * 100
                ocean = 100 - land
                land_area.append(land)
                ocean_area.append(ocean)
            
            axs[1, 1].plot(self.history['ages'], land_area, label='Land')
            axs[1, 1].plot(self.history['ages'], ocean_area, label='Ocean')
            
        axs[1, 1].set_xlabel('Age (million years)')
        axs[1, 1].set_ylabel('Percentage of Surface Area')
        axs[1, 1].set_title('Land vs Ocean Area Over Time')
        axs[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"History visualization saved to {save_path}")
        else:
            plt.show()
            
    def visualize_plates(self, save_path=None):
        """
        Create a visualization of the current plate configuration.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create arrays for plotting
        xs, ys, zs = [], [], []
        colors = []
        
        # Assign a unique color to each plate
        plate_colors = {}
        for plate in self.plates:
            # Generate a unique color based on plate ID
            hue = (plate['id'] * 0.618033988749895) % 1.0  # Golden ratio distribution
            saturation = 0.8
            value = 0.8 if not plate['is_oceanic'] else 0.5
            
            # Convert HSV to RGB
            import colorsys
            plate_colors[plate['id']] = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Prepare data for plotting
        for i, vertex in enumerate(self.planet.grid.vertices):
            xs.append(vertex[0])
            ys.append(vertex[1])
            zs.append(vertex[2])
            
            plate_id = self.planet.plate_ids[i]
            
            if plate_id in plate_colors:
                colors.append(plate_colors[plate_id])
            else:
                colors.append((0.7, 0.7, 0.7))  # Default gray
        
        # Plot plates
        ax.scatter(xs, ys, zs, c=colors, s=10)
        
        # Plot plate boundaries
        boundary_xs, boundary_ys, boundary_zs = [], [], []
        for i, is_boundary in enumerate(self.plate_boundaries):
            if is_boundary:
                vertex = self.planet.grid.vertices[i]
                boundary_xs.append(vertex[0])
                boundary_ys.append(vertex[1])
                boundary_zs.append(vertex[2])
        
        ax.scatter(boundary_xs, boundary_ys, boundary_zs, c='black', s=10)
        
        # Plot plate motion vectors
        for plate in self.plates:
            center = plate['center']
            velocity = plate['velocity']
            
            # Scale velocity for visibility
            arrow_scale = 50.0
            end_point = center + velocity * arrow_scale
            
            ax.quiver(center[0], center[1], center[2],
                      velocity[0], velocity[1], velocity[2],
                      color='red', arrow_length_ratio=0.3, length=arrow_scale)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        ax.set_axis_off()
        
        plt.title(f'Tectonic Plate Configuration at {self.planet.age} million years')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Plate visualization saved to {save_path}")
        else:
            plt.show()