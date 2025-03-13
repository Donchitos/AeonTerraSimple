import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter
import datetime
import os
from ..utils.map_visualizer import MapVisualizer

class TectonicSimulation:
    """
    A comprehensive simulation of planetary tectonic processes with enhanced 
    scientific accuracy based on Earth observations.
    """
    
    def __init__(self, planet, num_plates=12, initial_age=4500, 
                 tectonic_activity=1.0, mantle_viscosity=1.0,
                 continental_fraction=0.3, plate_thickness_variation=1.0):
        """
        Initialize the tectonic simulation with customizable parameters.
        
        Parameters:
        - planet: A Planet object
        - num_plates: Initial number of plates
        - initial_age: Age of the planet in million years when simulation starts
        - tectonic_activity: 0.0-2.0, controls overall rate of tectonic processes
        - mantle_viscosity: 0.5-2.0, affects plate speeds (lower = faster plates)
        - continental_fraction: 0.1-0.9, target continental crust percentage
        - plate_thickness_variation: 0.5-2.0, controls how varied plate thickness is
        """
        self.planet = planet
        self.num_plates = num_plates
        self.initial_age = initial_age
        
        # Customization parameters
        self.tectonic_activity = tectonic_activity  # Overall rate multiplier
        self.mantle_viscosity = mantle_viscosity  # Controls resistance to flow
        self.target_continental_fraction = continental_fraction  # Target %
        self.plate_thickness_variation = plate_thickness_variation  # Variation amount
        
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
            'continental_area': [],
            'supercontinent_index': []  # New metric for continent aggregation
        }
        
        # Enhanced physical properties
        self.crust_age = np.zeros(len(planet.grid.vertices))
        self.crust_thickness = np.zeros(len(planet.grid.vertices))
        self.crust_density = np.zeros(len(planet.grid.vertices))
        self.crust_type = np.zeros(len(planet.grid.vertices), dtype=int)  # 0=oceanic, 1=continental
        self.boundary_stress = np.zeros(len(planet.grid.vertices))  # Stress at boundaries
        self.boundary_age = np.zeros(len(planet.grid.vertices))  # Age of boundaries
        
        # Constants calibrated for Earth-like planet
        # Based on scientific observations with appropriate scaling
        self.CONTINENTAL_DENSITY = 2.7  # g/cm³
        self.OCEANIC_DENSITY = 3.0      # g/cm³
        self.AVG_CONTINENTAL_THICKNESS = 35.0  # km
        self.AVG_OCEANIC_THICKNESS = 7.0       # km
        self.MANTLE_DENSITY = 3.3       # g/cm³
        
        # Plate motion constants (calibrated from observed Earth rates)
        self.PLATE_VELOCITY_OCEANIC_MAX = 10.0 * self.tectonic_activity / self.mantle_viscosity  # cm/year
        self.PLATE_VELOCITY_CONTINENTAL_MAX = 5.0 * self.tectonic_activity / self.mantle_viscosity  # cm/year
        self.SEAFLOOR_SPREADING_RATE_FAST = 15.0 * self.tectonic_activity  # cm/year (East Pacific Rise)
        self.SEAFLOOR_SPREADING_RATE_SLOW = 2.0 * self.tectonic_activity   # cm/year (Mid-Atlantic Ridge)
        self.MOUNTAIN_BUILDING_RATE_MAX = 1.0 * self.tectonic_activity     # cm/year (Himalayas)
        self.OCEANIC_CRUST_LIFESPAN_MAX = 200.0 / self.tectonic_activity  # million years
        
        # Collision and subduction physics
        self.SUBDUCTION_ANGLE_BASE = np.radians(30)  # Base subduction angle
        self.SUBDUCTION_ANGLE_VAR = np.radians(20)   # Variation in subduction angle
        self.COLLISION_THICKENING_RATE = 0.5 * self.tectonic_activity  # km/My
        self.STRESS_FACTOR = 0.2 * self.tectonic_activity  # Stress accumulation rate
        
        # Supercontinent cycle parameters
        self.SUPERCONTINENT_THRESHOLD = 0.3  # Radian distance threshold for supercontinent
        self.last_supercontinent_breakup = -1000  # Time since last breakup (My)
        
        # Planetary core and mantle properties
        self.core_temp = 5000  # K, approximate Earth core temperature
        self.mantle_heat = 1500  # Relative units
        self.radiogenic_heat = 1.0  # Relative units for radioactive elements
        self.CORE_HEAT_TRANSFER = 0.0001  # Heat transfer coefficient
        self.LOWER_MANTLE_FACTOR = 10.0  # Increased viscosity of lower mantle
        
        # Hotspot and plume tracking
        self.mantle_plumes = []
        self.hotspots = []
        
        # Tracking of specific geological features
        self.orogenic_belts = []  # Mountain-building regions
        self.rifts = []           # Divergent boundaries on continents
        self.volcanic_arcs = []   # Volcanic chains at subduction zones
        self.transform_faults = [] # Major transform faults
        self.back_arc_basins = [] # Back-arc spreading centers
        
        # Event tracking for reporting
        self.recent_events = []
        self.all_events = []
        
        # Visualization params
        self.show_plates = True
        
        # Allow the planet to access this simulation
        self.planet._tectonics = self
    
    def initialize_planetary_evolution(self, evolution_stages=3):
        """
        Simulate early planetary evolution before fully developed plate tectonics.
        
        Parameters:
        - evolution_stages: Number of early evolution stages to simulate
        """
        print("Simulating early planetary evolution...")
        
        # Stage 1: Magma ocean and primary crust formation
        self._simulate_magma_ocean_cooling()
        
        # Stage 2: Mantle differentiation and initial density stratification
        self._simulate_mantle_differentiation()
        
        # Stage 3: Proto-plate formation via mantle overturn events
        self._simulate_early_tectonics()
        
        print("Early planetary evolution complete.")
        return self
    
    def _simulate_magma_ocean_cooling(self):
        """Simulate cooling of initial magma ocean and primary crust formation"""
        print("  Stage 1: Magma ocean cooling and primary crust formation")
        
        # Generate primary crust thickness variations
        # Initially thinner crust with high variation
        base_thickness = self.AVG_OCEANIC_THICKNESS * 0.7
        
        # Create coherent noise for initial crustal variations
        thickness_noise = self._generate_coherent_noise(octaves=3, persistence=0.6)
        
        # Apply scientific constraints to initial crust
        # Primary crust was likely thin and relatively uniform
        self.crust_thickness = base_thickness + thickness_noise * 3.0 * self.plate_thickness_variation
        
        # Set initial crust density (higher than modern continental crust)
        self.crust_density.fill(self.OCEANIC_DENSITY + 0.1)
        
        # Set initial crust type (all primary undifferentiated crust)
        self.crust_type.fill(0)  # Start with primary oceanic-like crust
        
        # Set initial crust age (all formed around the same time)
        age_variation = 100 * np.random.random(len(self.crust_age))
        self.crust_age = self.initial_age - age_variation
    
    def _simulate_mantle_differentiation(self):
        """Simulate mantle differentiation and initial crustal dichotomy"""
        print("  Stage 2: Mantle differentiation and crustal evolution")
        
        # Generate regions where early continental crust forms
        # Continental crust forms through partial melting and reprocessing
        proto_continents = self._generate_coherent_noise(octaves=3, scale=3.0)
        
        # Higher values = more likely to form continental crust
        # Target the desired continental percentage
        continental_threshold = np.percentile(proto_continents, 
                                             (1 - self.target_continental_fraction) * 100)
        
        continental_mask = proto_continents > continental_threshold
        
        # Convert these regions to early continental crust (cratons)
        self.crust_type[continental_mask] = 1  # Continental
        
        # Early continental crust was thick but density similar to modern
        self.crust_thickness[continental_mask] = (self.AVG_CONTINENTAL_THICKNESS * 1.1 + 
                                                 5 * np.random.random(np.sum(continental_mask)))
        self.crust_density[continental_mask] = self.CONTINENTAL_DENSITY
        
        # Continental crust is older than surrounding oceanic crust
        self.crust_age[continental_mask] = self.initial_age * 0.9 + self.initial_age * 0.1 * np.random.random(np.sum(continental_mask))
    
    def _simulate_early_tectonics(self):
        """Simulate early tectonic processes before stable plate tectonics"""
        print("  Stage 3: Proto-plate formation via early tectonic processes")
        
        # Create small proto-plates using Voronoi cells
        # Early Earth likely had many small plates
        proto_plates = self.num_plates * 2  # More, smaller plates
        self._generate_plate_boundaries(proto_plates)
        
        # Initialize mantle convection system
        # Higher vigor convection in early Earth due to higher heat
        early_plumes = max(5, int(7 * self.tectonic_activity))
        self.initialize_mantle(num_cells=12, num_plumes=early_plumes)
        
        # Set up early plate motions
        # Early Earth had faster plates due to higher heat flow
        self._initialize_plate_properties(velocity_scale=1.5)
        
        # Early "simmer-down" phase - run a short simulation of rapid tectonics
        early_time = 100.0  # 100 My of early rapid tectonics
        print(f"  Simulating {early_time} million years of early rapid tectonics...")
        
        # Temporarily increase tectonic activity
        saved_activity = self.tectonic_activity
        self.tectonic_activity *= 1.5
        
        # Run rapid early tectonics
        self._move_plates(early_time)
        self._handle_crust_processes(early_time)
        self._age_crust(early_time)
        
        # Reset tectonic activity
        self.tectonic_activity = saved_activity
    
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
            strength = 0.005 * (1.0 + 0.5 * np.random.random()) / self.mantle_viscosity
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
            plume_strength = 0.02 * (1.0 + 0.5 * np.random.random()) / self.mantle_viscosity
            radius = 0.2 + 0.3 * np.random.random()  # Smaller influence radius
            
            self.mantle_plumes.append({
                'center': [x, y, z],
                'base_strength': plume_strength,
                'current_strength': plume_strength,  # Can vary over time
                'radius': radius,
                'age': 0,  # Plumes can die and new ones can form
                'pulse_phase': np.random.random() * 2 * np.pi,  # Random starting phase
                'pulse_period': 50 + 50 * np.random.random()  # 50-100 My oscillation
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
        """Calculate mantle convection forces at each vertex with scientific calibration"""
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
                plume_strength = plume['current_strength'] * (1.0 - dist/plume['radius'])
                force -= vec_to_center * plume_strength
            
            # Scale force based on tectonic activity
            force *= self.tectonic_activity
            
            # Store force vector
            self.mantle_forces[i] = force
    
    def _update_mantle_plumes(self, time_step):
        """Update mantle plumes with more realistic behavior"""
        # Plumes periodically pulsate and can migrate
        for plume in self.mantle_plumes:
            # Update age
            plume['age'] += time_step
            
            # Update pulse phase
            phase_change = time_step / plume['pulse_period'] * 2 * np.pi
            plume['pulse_phase'] = (plume['pulse_phase'] + phase_change) % (2 * np.pi)
            
            # Calculate current strength based on pulse phase
            strength_factor = 0.7 + 0.6 * np.sin(plume['pulse_phase'])
            plume['current_strength'] = plume['base_strength'] * strength_factor
            
            # Plumes can migrate slightly
            if np.random.random() < 0.1 * time_step / 50:
                # Random walk on sphere surface
                deviation = np.random.normal(0, 0.01, 3)
                center = np.array(plume['center'])
                # Make deviation tangent to sphere surface
                deviation = deviation - np.dot(deviation, center) * center
                new_pos = center + deviation * 0.1
                plume['center'] = list(new_pos / np.linalg.norm(new_pos))
    
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
        continental_threshold = 1.0 - self.target_continental_fraction
        continental_mask = thickness_noise > continental_threshold
        
        # Set thickness based on crust type
        self.crust_thickness = np.zeros(len(self.planet.grid.vertices))
        
        # Continental crust (25-70 km, averaging around 35 km)
        # Apply variation based on the customization parameter
        thickness_var = 15 * self.plate_thickness_variation
        continental_thickness = self.AVG_CONTINENTAL_THICKNESS + thickness_var * (thickness_noise[continental_mask] - continental_threshold) / (1.0 - continental_threshold)
        self.crust_thickness[continental_mask] = continental_thickness
        
        # Oceanic crust (5-10 km, averaging around 7 km)
        # Apply variation based on the customization parameter
        oceanic_thickness = self.AVG_OCEANIC_THICKNESS + 2 * self.plate_thickness_variation * thickness_noise[~continental_mask]
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
        
        # Add initial event
        self._add_event("Initial planetary configuration established")
        
        print(f"Initialized {len(self.plates)} tectonic plates")
        return self
        
    def _generate_plate_boundaries(self, custom_num_plates=None):
        """Generate plate boundaries using Voronoi cells on the sphere"""
        num_plates = custom_num_plates if custom_num_plates is not None else self.num_plates
        
        # Step 1: Choose random seed points on the sphere
        seed_points = []
        for _ in range(num_plates):
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
        
        # Initialize boundary stress and age to zero
        self.boundary_stress = np.zeros(len(self.planet.grid.vertices))
        self.boundary_age = np.zeros(len(self.planet.grid.vertices))
    
    def _initialize_plate_properties(self, velocity_scale=1.0):
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
            
            # Scale velocity based on plate type (oceanic moves faster)
            # And apply the velocity scale parameter
            if is_oceanic:
                vel_factor = 1.5 * velocity_scale
                max_vel = self.PLATE_VELOCITY_OCEANIC_MAX / 100.0  # Convert cm/year to simulation units
            else:
                vel_factor = 0.8 * velocity_scale
                max_vel = self.PLATE_VELOCITY_CONTINENTAL_MAX / 100.0  # Convert cm/year to simulation units
            
            # Apply velocity but cap at maximum based on plate type
            velocity = plate_force * vel_factor
            vel_mag = np.linalg.norm(velocity)
            if vel_mag > max_vel:
                velocity = velocity * (max_vel / vel_mag)
            
            # Create plate object with enhanced properties
            plate = {
                'id': i,
                'center': center_pos,
                'vertices': plate_vertices,
                'is_oceanic': is_oceanic,
                'velocity': velocity,
                'area': len(plate_vertices) / len(self.planet.grid.vertices),
                'age': self.initial_age,
                'continental_percentage': continental_crust_count / len(plate_vertices) * 100,
                'boundaries': [],
                'features': {
                    'orogenic_belts': [],  # Mountain ranges
                    'rifts': [],           # Spreading centers
                    'volcanic_arcs': [],   # Subduction volcanoes
                    'hotspots': [],        # Intraplate volcanic centers
                    'transform_zones': []  # Transform fault zones
                }
            }
            
            self.plates.append(plate)
        
        # Update plate boundaries
        self._update_plate_boundaries()
    
    def _update_plate_boundaries(self):
        """Update the list of plate boundaries and classify boundary types"""
        # Reset boundary markers
        self.plate_boundaries = np.zeros(len(self.planet.grid.vertices), dtype=bool)
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        # Clear boundary lists for each plate
        for plate in self.plates:
            plate['boundaries'] = []
        
        # Process vertices in blocks for better performance
        for block_start in range(0, len(self.planet.grid.vertices), 1024):
            block_end = min(block_start + 1024, len(self.planet.grid.vertices))
            
            for i in range(block_start, block_end):
                plate_id = self.planet.plate_ids[i]
                
                # Skip if vertex isn't assigned to a plate
                if plate_id == -1 or plate_id >= len(self.plates):
                    continue
                    
                # Check neighbors for different plates
                for neighbor in neighbors[i]:
                    neighbor_plate = self.planet.plate_ids[neighbor]
                    
                    if neighbor_plate != plate_id and neighbor_plate != -1 and neighbor_plate < len(self.plates):
                        # This is a boundary vertex
                        self.plate_boundaries[i] = True
                        
                        # Calculate boundary type based on relative motion
                        boundary_type = self._calculate_boundary_type(i, plate_id, neighbor_plate)
                        
                        # Add to plate's boundary list if not already there
                        boundary_exists = False
                        for boundary in self.plates[plate_id]['boundaries']:
                            if boundary.get('neighbor_plate') == neighbor_plate:
                                boundary_exists = True
                                break
                                
                        if not boundary_exists:
                            self.plates[plate_id]['boundaries'].append({
                                'neighbor_plate': neighbor_plate,
                                'type': boundary_type,
                                'vertices': [i]
                            })
                        else:
                            # Add vertex to existing boundary
                            for boundary in self.plates[plate_id]['boundaries']:
                                if boundary.get('neighbor_plate') == neighbor_plate:
                                    if i not in boundary['vertices']:
                                        boundary['vertices'].append(i)
                                    break
                        
                        break  # Only mark once per vertex
    
    def _calculate_boundary_type(self, vertex_idx, plate1_id, plate2_id):
        """Calculate the type of boundary based on relative plate motion"""
        # Handle the case where plate2_id is None
        if plate2_id is None:
            # In this case, we need to determine if this is at a boundary
            # Let's check if any neighbor is from a different plate
            neighbors = self.planet.grid.get_vertex_neighbors()[vertex_idx]
            different_plate_neighbors = []
            
            for n in neighbors:
                if self.planet.plate_ids[n] != plate1_id and self.planet.plate_ids[n] != -1 and self.planet.plate_ids[n] < len(self.plates):
                    different_plate_neighbors.append(n)
            
            if not different_plate_neighbors:
                # No different plate neighbors, not a boundary
                return "none"
            
            # Use the first different neighbor as plate2
            plate2_id = self.planet.plate_ids[different_plate_neighbors[0]]
        
        # Get plate velocities
        v1 = self.plates[plate1_id]['velocity']
        v2 = self.plates[plate2_id]['velocity']
        
        # Calculate relative velocity
        rel_velocity = v1 - v2
        
        # Get normal vector to boundary
        normal = self.planet.grid.vertices[vertex_idx] / np.linalg.norm(self.planet.grid.vertices[vertex_idx])
        
        # Dot product tells us if plates are converging or diverging
        convergence = -np.dot(rel_velocity, normal)
        
        # Calculate tangential component for transform motion
        tangential = rel_velocity - convergence * normal
        tangential_mag = np.linalg.norm(tangential)
        
        # Determine boundary type based on motion components
        if abs(convergence) < 0.001 and tangential_mag > 0.001:
            return "transform"
        elif convergence < -0.001:
            return "divergent"
        elif convergence > 0.001:
            return "convergent"
        else:
            return "complex"  # Mixed or indeterminate motion
    
    def _calculate_boundary_forces(self, vertex_idx, plate1_id, plate2_id, time_step):
        """Calculate precise forces and stress at plate boundaries"""
        # Get plate velocities
        v1 = self.plates[plate1_id]['velocity']
        v2 = self.plates[plate2_id]['velocity']
        
        # Calculate relative velocity
        rel_velocity = v1 - v2
        
        # Get normal vector to boundary
        normal = self.planet.grid.vertices[vertex_idx] / np.linalg.norm(self.planet.grid.vertices[vertex_idx])
        
        # Dot product tells us if plates are converging or diverging
        normal_component = -np.dot(rel_velocity, normal)
        
        # Calculate tangential component
        tangential_component = rel_velocity - normal_component * normal
        tangential_mag = np.linalg.norm(tangential_component)
        
        # Calculate stress based on motion
        # Convergent and transform boundaries accumulate stress
        if normal_component > 0 or tangential_mag > 0.001:
            stress_increase = (abs(normal_component) + tangential_mag * 0.5) * self.STRESS_FACTOR
            self.boundary_stress[vertex_idx] += stress_increase * time_step
        else:
            # Divergent boundaries don't accumulate stress
            self.boundary_stress[vertex_idx] = 0
            
        # Age the boundary
        self.boundary_age[vertex_idx] += time_step
        
        # Determine boundary type based on motion components
        if abs(normal_component) < 0.001 and tangential_mag > 0.001:
            return "transform", self.boundary_stress[vertex_idx]
        elif normal_component < -0.001:
            return "divergent", self.boundary_stress[vertex_idx]
        elif normal_component > 0.001:
            return "convergent", self.boundary_stress[vertex_idx]
        else:
            return "complex", self.boundary_stress[vertex_idx]
                    
    def _calculate_isostatic_elevation(self):
        """
        Calculate elevation based on isostatic equilibrium with enhanced precision
        based on the Airy isostasy model where thicker/less dense crust
        floats higher on the mantle, like blocks of wood on water.
        """
        # Calculate elevation based on density and thickness (isostatic compensation)
        # Formula derived from Archimedes' principle
        for i in range(len(self.planet.grid.vertices)):
            # Get crustal properties
            thickness = self.crust_thickness[i]
            density = self.crust_density[i]
            
            # Airy isostasy model: h = t * (ρm - ρc)/ρm
            # where h = elevation, t = thickness, ρm = mantle density, ρc = crust density
            isostatic_height = thickness * (self.MANTLE_DENSITY - density) / self.MANTLE_DENSITY
            
            if self.crust_type[i] == 1:  # Continental
                # Continental crust rises higher due to lower density
                # Scale to realistic Earth-like elevations
                # Average continent is ~0.8 km above sea level, but with variation
                elevation = isostatic_height * 0.05 - 1.0
                
                # Add local variation based on thickness
                # Shields (thick old continental crust) are often higher
                thickness_variation = (thickness - self.AVG_CONTINENTAL_THICKNESS) / 20.0
                elevation += thickness_variation
                
                # Adjust for cratons (very old continental crust)
                if self.crust_age[i] > self.initial_age * 0.8:
                    elevation += 0.3  # Cratons tend to be higher
                
                self.planet.elevation[i] = elevation
            else:  # Oceanic
                # Oceanic crust starts lower due to higher density
                elevation = isostatic_height * 0.05 - 4.5
                
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
        
        # Update physical parameters that change with time
        self._update_planet_heat(time_step)
        
        # Update mantle plume behavior
        self._update_mantle_plumes(time_step)
        
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
        self._check_for_supercontinent_cycle(time_step)
        
        # Update planet age
        self.planet.age += time_step
        
        # Record state for history
        self._record_state()
        
        # Clear recent events list (will be filled by functions during this step)
        self.recent_events = []
        
        return self
        
    def _update_planet_heat(self, time_step):
        """Update the planet's heat budget"""
        # Core cooling
        cooling_rate = 0.0001 * time_step  # Degrees K per million years
        self.core_temp -= cooling_rate
        
        # Radioactive decay
        decay_factor = np.exp(-0.000000001 * time_step)  # Approximating radioactive decay
        self.radiogenic_heat *= decay_factor
        
        # Update mantle heat
        core_heat = self.core_temp * self.CORE_HEAT_TRANSFER * time_step
        self.mantle_heat = max(0, self.mantle_heat - 0.1 * time_step + core_heat)
        
        # Adjust mantle viscosity based on heat
        heat_factor = max(0.5, min(2.0, 1.0 + (1500 - self.mantle_heat) / 1500))
        self.mantle_viscosity = self.mantle_viscosity * 0.9 + heat_factor * 0.1
        
        # Recalculate mantle forces with updated parameters
        self._calculate_mantle_forces()
    
    def _move_plates(self, time_step):
        """Move plates based on velocity and time step with realistic rates"""
        # Convert time_step from million years to simulation units
        # Assuming 1.0 velocity unit = 5cm/year
        movement_scale = time_step * 0.05  # 5cm/year * million years
        
        # Scale by tectonic activity
        movement_scale *= self.tectonic_activity
        
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
                # Calculate boundary type and stress
                boundary_type, stress = self._calculate_boundary_forces(i, my_plate_id, neighbor_plate_id, time_step)
                
                # Handle different boundary types with enhanced scientific detail
                if boundary_type == "convergent":
                    self._handle_convergent_boundary(i, my_plate_id, neighbor_plate_id, time_step)
                elif boundary_type == "divergent":
                    self._handle_divergent_boundary(i, my_plate_id, neighbor_plate_id, time_step)
                elif boundary_type == "transform":
                    self._handle_transform_boundary(i, my_plate_id, neighbor_plate_id, time_step)
                else:  # Complex or indeterminate
                    # Complex boundaries can develop unpredictable features
                    if np.random.random() < 0.01 * time_step / 10.0:
                        if self.crust_type[i] == 1:  # Continental
                            # Intraplate deformation is common in complex boundaries
                            self.crust_thickness[i] += 1.0 * np.random.random() * time_step / 10.0
                            self.planet.elevation[i] += 0.5 * np.random.random() * time_step / 10.0
                        else:  # Oceanic
                            # Complex oceanic boundaries can develop seamounts
                            self.planet.elevation[i] += 0.3 * np.random.random() * time_step / 10.0
                
                # Stress-based feature formation
                # Major earthquakes, rifts, etc. occur when stress exceeds thresholds
                if stress > 5.0 and np.random.random() < 0.1 * time_step / 10.0:
                    # Reset stress after a major event
                    self.boundary_stress[i] *= 0.2
                    
                    # Different features form based on boundary type
                    if boundary_type == "transform":
                        # Major transform fault slippage
                        if i not in self.transform_faults:
                            self.transform_faults.append(i)
                    elif boundary_type == "convergent" and self.crust_type[i] == 1:
                        # Major earthquake in continental collision zone
                        # Can sometimes cause rapid elevation changes
                        self.planet.elevation[i] += 0.2 * np.random.random()
                        
                        # Record a major earthquake event
                        if np.random.random() < 0.2:  # Only record some of them to avoid spamming
                            x, y, z = self.planet.grid.vertices[i]
                            lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[i])) * 180 / np.pi
                            lon = np.arctan2(y, x) * 180 / np.pi
                            self._add_event(f"Major earthquake at convergent boundary (lat: {lat:.1f}, lon: {lon:.1f})")
    
    def _handle_convergent_boundary(self, vertex_idx, plate1_id, plate2_id, time_step):
        """
        Handle convergent plate boundary processes with enhanced scientific accuracy
        
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
        
        # Calculate convergence angle for more realistic physics
        # Get the plate velocities
        v1 = plate1['velocity']
        v2 = plate2['velocity']
        rel_velocity = v1 - v2
        
        # Get normal vector to boundary
        normal = self.planet.grid.vertices[vertex_idx] / np.linalg.norm(self.planet.grid.vertices[vertex_idx])
        
        # Calculate convergence components
        normal_component = np.dot(rel_velocity, normal)
        tangential_component = rel_velocity - normal_component * normal
        
        # Calculate angle between convergence direction and pure normal
        convergence_angle = np.arctan2(np.linalg.norm(tangential_component), abs(normal_component))
        
        # Get the boundary age - older boundaries behave differently
        boundary_age = self.boundary_age[vertex_idx]
        
        # Handle different convergence scenarios with scientific accuracy
        
        # Case 1: Continental-continental collision
        if is_continental1 and is_continental2:
            # Collision creates mountain ranges - intensity depends on angle
            collision_intensity = max(0.5, np.cos(convergence_angle))
            
            # Elevation increase follows real mountain growth rates
            # Typically 0.5-1 cm/year or 5-10 km per million years
            base_rate = self.MOUNTAIN_BUILDING_RATE_MAX * 100 * time_step / 10.0  # Convert from cm/year to km/My
            elevation_increase = base_rate * collision_intensity * np.random.random()
            
            # Scale by tectonic activity parameter
            elevation_increase *= self.tectonic_activity
            
            self.planet.elevation[vertex_idx] += elevation_increase
            
            # Thicken the crust (due to compression)
            # Continental crust can double in thickness at collision zones
            thickness_increase = self.COLLISION_THICKENING_RATE * time_step / 10.0 * collision_intensity
            self.crust_thickness[vertex_idx] += thickness_increase
            
            # Mature collision zones develop plateaus
            if boundary_age > 50:  # After 50 My, start forming plateaus
                plateau_factor = min(1.0, (boundary_age - 50) / 50)
                
                # Expand mountain range to neighbors
                for neighbor in neighbor_vertices:
                    if self.planet.plate_ids[neighbor] == plate1_id and self.crust_type[neighbor] == 1:
                        self.planet.elevation[neighbor] += elevation_increase * plateau_factor * 0.7
                        self.crust_thickness[neighbor] += thickness_increase * plateau_factor * 0.7
            
            # Add to orogenic belts (mountain building regions)
            if plate2_id not in plate1['features']['orogenic_belts']:
                plate1['features']['orogenic_belts'].append(plate2_id)
                
                # Add event for new mountain range
                if np.random.random() < 0.5:  # 50% chance to record this event
                    # Get location in lat/lon for better descriptions
                    x, y, z = self.planet.grid.vertices[vertex_idx]
                    lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[vertex_idx])) * 180 / np.pi
                    lon = np.arctan2(y, x) * 180 / np.pi
                    
                    self._add_event(f"New mountain range forming from continental collision at lat: {lat:.1f}, lon: {lon:.1f}")
            
            if plate1_id not in plate2['features']['orogenic_belts']:
                plate2['features']['orogenic_belts'].append(plate1_id)
        
        # Case 2: Oceanic-continental subduction
        elif (not is_continental1 and is_continental2) or (is_continental1 and not is_continental2):
            # Determine subducting and overriding plates
            if (not is_continental1 and is_continental2):
                subducting_plate = plate1
                subducting_vertex = vertex_idx
                overriding_plate = plate2
                overriding_vertex = other_plate_vertex
            else:
                subducting_plate = plate2
                subducting_vertex = other_plate_vertex
                overriding_plate = plate1
                overriding_vertex = vertex_idx
            
            # Calculate subduction angle - varies by location and age
            # Older, colder slabs tend to subduct at steeper angles
            subducting_age = self.crust_age[subducting_vertex]
            age_factor = min(1.0, subducting_age / 100.0)
            
            # Base angle plus variation (measured from horizontal)
            subduction_angle = self.SUBDUCTION_ANGLE_BASE + self.SUBDUCTION_ANGLE_VAR * np.random.random() * age_factor
            
            # Create coastal mountains on overriding plate
            # Typically 0.3-0.7 cm/year or 3-7 km per million years
            if self.planet.elevation[overriding_vertex] >= 0:
                elevation_increase = 0.3 + 0.4 * np.random.random()  # km per million years
                elevation_increase *= time_step
                self.planet.elevation[overriding_vertex] += elevation_increase
                
                # Thicken crust at continental margin
                self.crust_thickness[overriding_vertex] += elevation_increase * 0.5
            
            # Create trench on subducting plate
            # Trenches are typically 8-11 km deep
            if self.planet.elevation[subducting_vertex] < 0:
                # Deepen the trench
                trench_deepening = 1.0 * time_step / 10.0
                self.planet.elevation[subducting_vertex] = max(-11.0, 
                                                             self.planet.elevation[subducting_vertex] - trench_deepening)
            
            # Calculate volcanic arc position based on subduction angle and crust thickness
            arc_distance = self.crust_thickness[subducting_vertex] / np.tan(subduction_angle)
            
            # Find vertices that might be in the volcanic arc
            # This is simplified - in reality, would need a more precise spatial search
            potential_arc_vertices = []
            for v in overriding_plate['vertices']:
                if self.planet.elevation[v] >= 0:  # Only on land
                    # Get vector from trench to this point
                    trench_pos = self.planet.grid.vertices[subducting_vertex]
                    vertex_pos = self.planet.grid.vertices[v]
                    
                    # Project onto surface
                    trench_pos = trench_pos / np.linalg.norm(trench_pos)
                    vertex_pos = vertex_pos / np.linalg.norm(vertex_pos)
                    
                    # Calculate great-circle distance
                    dot = np.clip(np.dot(trench_pos, vertex_pos), -1.0, 1.0)
                    dist = np.arccos(dot) * self.planet.radius
                    
                    # Check if in the approximate arc zone (80-120% of expected distance)
                    if 0.8 * arc_distance < dist < 1.2 * arc_distance:
                        potential_arc_vertices.append(v)
            
            # Create volcanic arc features
            arc_chance = 0.01 * time_step
            for v in potential_arc_vertices:
                if np.random.random() < arc_chance:
                    # Create volcanic mountain
                    self.planet.elevation[v] += 0.5 + 1.0 * np.random.random()
                    
                    # Add volcanic arc to features
                    if subducting_plate['id'] not in overriding_plate['features']['volcanic_arcs']:
                        overriding_plate['features']['volcanic_arcs'].append(subducting_plate['id'])
                        
                        # Add event for new volcanic arc
                        x, y, z = self.planet.grid.vertices[v]
                        lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[v])) * 180 / np.pi
                        lon = np.arctan2(y, x) * 180 / np.pi
                        
                        self._add_event(f"Volcanic arc forming at subduction zone near lat: {lat:.1f}, lon: {lon:.1f}")
            
            # Create back-arc basin with certain probability
            # Older subduction zones often develop back-arc spreading
            if boundary_age > 30 and np.random.random() < 0.05 * time_step / 10.0:
                self._create_back_arc_basin(subducting_vertex, overriding_plate['id'], arc_distance * 1.5)
        
        # Case 3: Oceanic-oceanic subduction
        elif not is_continental1 and not is_continental2:
            # Older plate subducts under younger plate
            age1 = self.crust_age[vertex_idx]
            age2 = self.crust_age[other_plate_vertex]
            
            # Determine which plate subducts
            if age1 > age2:
                # Plate 1 subducts
                self.planet.elevation[vertex_idx] -= 1.0 * time_step / 10.0
                subducting_plate = plate1
                overriding_plate = plate2
                
                # Island arc formation on plate 2
                # Island arcs form at distances controlled by subduction angle
                island_arc_chance = 0.05 * time_step / 10.0
                
                if np.random.random() < island_arc_chance:
                    self.planet.elevation[other_plate_vertex] = max(0, self.planet.elevation[other_plate_vertex])
                    self.crust_type[other_plate_vertex] = 1  # Convert to continental crust
                    self.crust_thickness[other_plate_vertex] = 15.0  # Initial island arc thickness
                    self.crust_density[other_plate_vertex] = self.CONTINENTAL_DENSITY
                    
                    # Reset age for new arc material
                    self.crust_age[other_plate_vertex] = 0
                    
                    # Record island arc formation
                    x, y, z = self.planet.grid.vertices[other_plate_vertex]
                    lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[other_plate_vertex])) * 180 / np.pi
                    lon = np.arctan2(y, x) * 180 / np.pi
                    
                    self._add_event(f"Island arc forming at oceanic-oceanic subduction zone near lat: {lat:.1f}, lon: {lon:.1f}")
            else:
                # Plate 2 subducts
                self.planet.elevation[other_plate_vertex] -= 1.0 * time_step / 10.0
                subducting_plate = plate2
                overriding_plate = plate1
                
                # Island arc formation on plate 1
                island_arc_chance = 0.05 * time_step / 10.0
                
                if np.random.random() < island_arc_chance:
                    self.planet.elevation[vertex_idx] = max(0, self.planet.elevation[vertex_idx])
                    self.crust_type[vertex_idx] = 1  # Convert to continental crust
                    self.crust_thickness[vertex_idx] = 15.0  # Initial island arc thickness
                    self.crust_density[vertex_idx] = self.CONTINENTAL_DENSITY
                    
                    # Reset age for new arc material
                    self.crust_age[vertex_idx] = 0
                    
                    # Record island arc formation
                    x, y, z = self.planet.grid.vertices[vertex_idx]
                    lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[vertex_idx])) * 180 / np.pi
                    lon = np.arctan2(y, x) * 180 / np.pi
                    
                    self._add_event(f"Island arc forming at oceanic-oceanic subduction zone near lat: {lat:.1f}, lon: {lon:.1f}")
            
            # Record the subduction relationship
            if overriding_plate['id'] not in subducting_plate['features']['volcanic_arcs']:
                subducting_plate['features']['volcanic_arcs'].append(overriding_plate['id'])
    
    def _create_back_arc_basin(self, trench_vertex, overriding_plate_id, distance):
        """Create a back-arc basin behind a volcanic arc"""
        # Get location in lat/lon for event description
        x, y, z = self.planet.grid.vertices[trench_vertex]
        lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[trench_vertex])) * 180 / np.pi
        lon = np.arctan2(y, x) * 180 / np.pi
        
        self._add_event(f"Back-arc basin forming near lat: {lat:.1f}, lon: {lon:.1f}")
        
        # Find potential basin locations
        basin_vertices = []
        
        # Trench position
        trench_pos = self.planet.grid.vertices[trench_vertex]
        trench_pos = trench_pos / np.linalg.norm(trench_pos)
        
        # Check all vertices in the overriding plate
        for v in self.plates[overriding_plate_id]['vertices']:
            vertex_pos = self.planet.grid.vertices[v]
            vertex_pos = vertex_pos / np.linalg.norm(vertex_pos)
            
            # Calculate distance from trench
            dot = np.clip(np.dot(trench_pos, vertex_pos), -1.0, 1.0)
            dist = np.arccos(dot) * self.planet.radius
            
            # If this vertex is at approximately the back-arc distance
            if 0.8 * distance < dist < 1.2 * distance:
                basin_vertices.append(v)
        
        # Convert some of these to oceanic crust (rifting)
        for v in basin_vertices:
            if np.random.random() < 0.2:
                # Convert to oceanic crust
                if self.crust_type[v] == 1:  # Only convert continental to oceanic
                    self.crust_type[v] = 0  # Oceanic
                    self.crust_thickness[v] = self.AVG_OCEANIC_THICKNESS
                    self.crust_density[v] = self.OCEANIC_DENSITY
                    self.crust_age[v] = 0  # Brand new crust
                    
                    # Lower elevation to create a basin
                    self.planet.elevation[v] = -2.0 + 0.5 * np.random.random()
        
        # Add to back arc basin list
        self.back_arc_basins.append({
            'trench_vertex': trench_vertex,
            'overriding_plate': overriding_plate_id,
            'basin_vertices': basin_vertices,
            'age': 0
        })
    
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
        
        # Get boundary age
        boundary_age = self.boundary_age[vertex_idx]
        
        # Calculate spreading rate
        # Get plate velocities
        v1 = self.plates[plate1_id]['velocity']
        v2 = self.plates[plate2_id]['velocity']
        rel_velocity = v1 - v2
        
        # Get normal vector to boundary
        normal = self.planet.grid.vertices[vertex_idx] / np.linalg.norm(self.planet.grid.vertices[vertex_idx])
        
        # Calculate divergence rate
        divergence_rate = abs(np.dot(rel_velocity, normal))
        
        # Scale to realistic values (0 to 15 cm/year)
        spreading_rate = min(self.SEAFLOOR_SPREADING_RATE_FAST, 
                            self.SEAFLOOR_SPREADING_RATE_SLOW + 
                            divergence_rate * (self.SEAFLOOR_SPREADING_RATE_FAST - self.SEAFLOOR_SPREADING_RATE_SLOW) / 0.02)
        
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
            
            # Rifts follow distinct stages as they mature
            rift_stage = min(3, int(boundary_age / 20))  # 3 stages over 60 My
            
            if rift_stage == 0:  # Early rifting
                # Initial rift valley forms, minimal subsidence
                if np.random.random() < 0.1 * time_step / 10.0:
                    # Fault formation - creates escarpments
                    self.planet.elevation[vertex_idx] -= 0.2 + 0.3 * np.random.random()
            
            elif rift_stage == 1:  # Developing rift
                # Continued thinning, more subsidence, some volcanism
                if np.random.random() < 0.15 * time_step / 10.0:
                    # Rift volcanism
                    if np.random.random() < 0.3:
                        self.planet.elevation[vertex_idx] += 0.3 + 0.5 * np.random.random()
                    else:
                        self.planet.elevation[vertex_idx] -= 0.4 + 0.4 * np.random.random()
            
            elif rift_stage >= 2:  # Mature rift
                # If crust thins enough, it transitions to oceanic
                if self.crust_thickness[vertex_idx] <= 15.0:
                    # Transition to oceanic conditions
                    if np.random.random() < 0.2 * time_step / 10.0:  # 20% chance per 10 My
                        self.crust_type[vertex_idx] = 0  # Convert to oceanic
                        self.crust_density[vertex_idx] = self.OCEANIC_DENSITY
                        self.crust_age[vertex_idx] = 0  # New crust
                        self.planet.elevation[vertex_idx] = -1.5  # Initial seafloor depth
                        
                        # Add to rifts feature and record event
                        if rift_stage == 2 and plate1_id not in self.plates[plate2_id]['features']['rifts']:
                            # Only record when a continental rift first becomes oceanic
                            self.plates[plate2_id]['features']['rifts'].append(plate1_id)
                            
                            # Get location for event
                            x, y, z = self.planet.grid.vertices[vertex_idx]
                            lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[vertex_idx])) * 180 / np.pi
                            lon = np.arctan2(y, x) * 180 / np.pi
                            
                            self._add_event(f"Continental rift opening to form new ocean basin at lat: {lat:.1f}, lon: {lon:.1f}")
                        
                        if rift_stage == 2 and plate2_id not in self.plates[plate1_id]['features']['rifts']:
                            self.plates[plate1_id]['features']['rifts'].append(plate2_id)
        
        # Case 2: Seafloor spreading
        else:
            # Create new oceanic crust at boundary
            self.crust_age[vertex_idx] = 0  # Brand new crust
            self.crust_thickness[vertex_idx] = self.AVG_OCEANIC_THICKNESS + 1.0 * np.random.random()
            
            # Spreading centers produce different topography based on rate
            if spreading_rate <= 5.0:  # Slow spreading
                # Slow spreading has rough topography, deep rift valleys
                self.planet.elevation[vertex_idx] = -2.5 + 0.5 * np.random.random()
            else:  # Fast spreading
                # Fast spreading has smooth topography, shallow axial high
                self.planet.elevation[vertex_idx] = -2.0 + 0.5 * np.random.random()
    
    def _handle_transform_boundary(self, vertex_idx, plate1_id, plate2_id, time_step):
        """
        Handle transform plate boundary processes with enhanced physics
        
        Parameters:
        - vertex_idx: Index of boundary vertex
        - plate1_id, plate2_id: IDs of interacting plates
        - time_step: Time step in million years
        """
        # Get plate velocities for detailed motion analysis
        v1 = self.plates[plate1_id]['velocity']
        v2 = self.plates[plate2_id]['velocity']
        rel_velocity = v1 - v2
        
        # Get normal vector to boundary
        normal = self.planet.grid.vertices[vertex_idx] / np.linalg.norm(self.planet.grid.vertices[vertex_idx])
        
        # Calculate transform motion vector (tangential component)
        tangential_component = rel_velocity - np.dot(rel_velocity, normal) * normal
        slip_rate = np.linalg.norm(tangential_component)
        
        # Stress accumulation is proportional to slip rate
        stress_accumulation = slip_rate * self.STRESS_FACTOR * time_step
        self.boundary_stress[vertex_idx] += stress_accumulation
        
        # Major strike-slip features form based on accumulated stress
        if self.boundary_stress[vertex_idx] > 5.0 and np.random.random() < 0.1 * time_step / 10.0:
            # Major earthquake occurs - stress release
            stress_release = self.boundary_stress[vertex_idx] * 0.8
            self.boundary_stress[vertex_idx] -= stress_release
            
            # Determine feature type based on crust and slip rate
            if self.crust_type[vertex_idx] == 1:  # Continental
                if slip_rate > 0.01:  # Fast slip
                    # Major continental transform (like San Andreas)
                    # Creates ridges and valleys along fault
                    feature_type = np.random.random()
                    
                    if feature_type < 0.5:  # Pull-apart basin
                        # Depression (typically 1-2 km deep)
                        self.planet.elevation[vertex_idx] -= 0.5 * np.random.random()
                    else:  # Push-up range
                        # Small uplift (typically 0.5-1 km high)
                        self.planet.elevation[vertex_idx] += 0.3 * np.random.random()
                        
                    # Record significant transform fault activity
                    if np.random.random() < 0.3:  # 30% chance to record this event
                        x, y, z = self.planet.grid.vertices[vertex_idx]
                        lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[vertex_idx])) * 180 / np.pi
                        lon = np.arctan2(y, x) * 180 / np.pi
                        self._add_event(f"Major transform fault activity at lat: {lat:.1f}, lon: {lon:.1f}")
                else:  # Slow slip
                    # Minor terrain modifications
                    if np.random.random() < 0.3:
                        self.planet.elevation[vertex_idx] += 0.1 * (np.random.random() - 0.5)
            else:  # Oceanic
                if slip_rate > 0.01:  # Fast slip
                    # Oceanic transform fault (like Mendocino)
                    # Creates offset ridge segments
                    self.planet.elevation[vertex_idx] -= 0.2 * np.random.random()
                # Otherwise minimal effect on seafloor topography
    
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
            # Apply damping (plate motion slows over time due to resistance)
            plate['velocity'] *= 0.9
            
            # Apply mantle forces
            plate_force = np.zeros(3)
            for vertex_idx in plate['vertices']:
                plate_force += self.mantle_forces[vertex_idx]
            plate_force /= len(plate['vertices'])
            
            # Scale force with plate properties
            if plate['is_oceanic']:
                force_scale = 0.12
            else:
                force_scale = 0.08
            
            plate['velocity'] += plate_force * force_scale
            
            # Ensure velocity is tangent to surface
            center = plate['center']
            vel_dot_center = np.dot(plate['velocity'], center)
            plate['velocity'] -= vel_dot_center * center
            
            # Apply velocity limits based on plate type
            vel_mag = np.linalg.norm(plate['velocity'])
            if plate['is_oceanic']:
                max_vel = self.PLATE_VELOCITY_OCEANIC_MAX / 100.0  # Convert to simulation units
            else:
                max_vel = self.PLATE_VELOCITY_CONTINENTAL_MAX / 100.0
                
            if vel_mag > max_vel:
                plate['velocity'] = plate['velocity'] * (max_vel / vel_mag)
    
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
                    plate_id = self.planet.plate_ids[i]
                    
                    # Skip if vertex isn't assigned to a plate
                    if plate_id == -1 or plate_id >= len(self.plates):
                        continue
                    
                    # Higher chance of volcanism for stronger hotspots and vertices directly above
                    volcano_chance = hotspot['strength'] * (1.0 - dist/0.1) * 0.1
                    volcano_chance *= self.tectonic_activity
                    
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
                                
                                # Add hotspot to plate features
                                if hotspot not in self.plates[plate_id]['features']['hotspots']:
                                    self.plates[plate_id]['features']['hotspots'].append(hotspot)
                                
                                # Record hotspot island formation
                                x, y, z = vertex
                                lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
                                lon = np.arctan2(y, x) * 180 / np.pi
                                
                                self._add_event(f"Hotspot volcanic island forming at lat: {lat:.1f}, lon: {lon:.1f}")
                            else:
                                # Seamount (underwater volcano)
                                self.planet.elevation[i] += 1.0 + 2.0 * np.random.random()
                        else:
                            # Continental volcano - increase elevation
                            self.planet.elevation[i] += 0.5 + 1.0 * np.random.random()
                            
                            # Add hotspot to plate features
                            if hotspot not in self.plates[plate_id]['features']['hotspots']:
                                self.plates[plate_id]['features']['hotspots'].append(hotspot)
        
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
                'base_strength': plume_strength,
                'current_strength': plume_strength,
                'radius': radius,
                'age': 0,
                'pulse_phase': np.random.random() * 2 * np.pi,
                'pulse_period': 50 + 50 * np.random.random()
            })
            
            # Create a hotspot at the plume location
            self.hotspots.append({
                'position': [x, y, z],
                'strength': plume_strength * 5,
                'age': 0,
                'track': []
            })
            
            # Record new plume formation
            lat = np.arcsin(z) * 180 / np.pi
            lon = np.arctan2(y, x) * 180 / np.pi
            
            self._add_event(f"New mantle plume formed near lat: {lat:.1f}, lon: {lon:.1f}")
        
        # Old plumes can die out
        for i in range(len(self.mantle_plumes) - 1, -1, -1):
            plume = self.mantle_plumes[i]
            plume['age'] += time_step
            
            if plume['age'] > 100:  # Plumes typically last 100-200 Myr
                # 5% chance of dying per 10 My after reaching 100 My age
                if np.random.random() < 0.05 * time_step / 10.0:
                    # Get location for event
                    x, y, z = plume['center']
                    lat = np.arcsin(z / np.linalg.norm(plume['center'])) * 180 / np.pi
                    lon = np.arctan2(y, x) * 180 / np.pi
                    
                    self._add_event(f"Mantle plume died near lat: {lat:.1f}, lon: {lon:.1f}")
                    
                    # Remove the plume and its hotspot
                    del self.mantle_plumes[i]
                    if i < len(self.hotspots):  # Safety check
                        del self.hotspots[i]
    
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
                # Maximum age based on oceanic crust lifespan parameter
                if self.crust_age[i] > self.OCEANIC_CRUST_LIFESPAN_MAX:
                    # Very old oceanic crust has higher chance to begin subduction
                    if np.random.random() < 0.2 * time_step / 10.0:
                        # Create a new trench or deepen existing
                        self.planet.elevation[i] -= 1.0 * np.random.random()
                        
                        # Increase density to promote subduction
                        self.crust_density[i] += 0.05
        
        # New crust at divergent boundaries remains young
        for i, is_boundary in enumerate(self.plate_boundaries):
            if not is_boundary:
                continue
                
            plate_id = self.planet.plate_ids[i]
            
            # Skip if vertex isn't assigned to a plate
            if plate_id == -1 or plate_id >= len(self.plates):
                continue
            
            # Find a neighbor from a different plate
            neighbors = self.planet.grid.get_vertex_neighbors()[i]
            different_plate_neighbor = None
            
            for n in neighbors:
                if self.planet.plate_ids[n] != plate_id and self.planet.plate_ids[n] != -1 and self.planet.plate_ids[n] < len(self.plates):
                    different_plate_neighbor = n
                    break
            
            if different_plate_neighbor is not None:
                neighbor_plate_id = self.planet.plate_ids[different_plate_neighbor]
                # Check boundary type
                boundary_type = self._calculate_boundary_type(i, plate_id, neighbor_plate_id)
            else:
                # No different plate neighbor found
                boundary_type = "none"
                
            if boundary_type == "divergent":
                if self.crust_type[i] == 0:  # Only oceanic crust gets reset
                    # Reset age at divergent boundaries (new crust forms)
                    self.crust_age[i] = 0

    def _check_for_supercontinent_cycle(self, time_step):
        """Check for supercontinent cycles and handle major tectonic events"""
        # Calculate continental dispersion index
        continental_coords = []
        continental_indices = []
        
        for i in range(len(self.planet.elevation)):
            if self.crust_type[i] == 1:  # Continental
                continental_coords.append(self.planet.grid.vertices[i])
                continental_indices.append(i)
        
        # Calculate mean position of all continental crust
        if continental_coords:
            mean_pos = np.mean(continental_coords, axis=0)
            mean_pos = mean_pos / np.linalg.norm(mean_pos)
            
            # Calculate average distance from mean
            distances = []
            for pos in continental_coords:
                norm_pos = pos / np.linalg.norm(pos)
                dot = np.clip(np.dot(norm_pos, mean_pos), -1.0, 1.0)
                distances.append(np.arccos(dot))
            
            dispersion = np.mean(distances)
            
            # Check if we're in assembly or dispersal phase
            if dispersion < self.SUPERCONTINENT_THRESHOLD:
                # We have a supercontinent - increased chance of breakup
                # But only if the last breakup was at least 300 Ma ago
                time_since_breakup = self.planet.age - self.last_supercontinent_breakup
                
                if time_since_breakup > 300 and np.random.random() < 0.05 * time_step / 10.0:
                    self._add_event(f"Major tectonic event: Supercontinent breakup at {self.planet.age} million years")
                    self._initiate_supercontinent_breakup(mean_pos)
                    
                    # Record this breakup
                    self.last_supercontinent_breakup = self.planet.age
            else:
                # Continents are dispersed - possibility of starting convergence
                if np.random.random() < 0.03 * time_step / 10.0:
                    self._increase_continental_convergence()
        
        # Possibility 1: Break up a large plate
        for plate in self.plates:
            # Only consider large plates
            if plate['area'] > 0.25:  # 25% of planet surface
                # 30% chance to break up over 10 My
                if np.random.random() < 0.03 * time_step:
                    self._add_event(f"Major tectonic event: Breaking up large plate {plate['id']}")
                    self._break_plate(plate)
                    return  # Only one major event per step
        
        # Possibility 2: Merge small plates
        small_plates = [p for p in self.plates if p['area'] < 0.05]
        if len(small_plates) >= 2:
            # 20% chance to merge over 10 My
            if np.random.random() < 0.02 * time_step:
                # Choose two small plates to merge
                p1, p2 = np.random.choice(small_plates, 2, replace=False)
                self._add_event(f"Major tectonic event: Merging small plates {p1['id']} and {p2['id']}")
                self._merge_plates(p1, p2)
                return  # Only one major event per step
    
    def _initiate_supercontinent_breakup(self, continent_center):
        """Initialize the breakup of a supercontinent with rifts"""
        # Create a breakup axis - a great circle through the supercontinent
        # First, get a random direction perpendicular to the center
        perp1 = np.random.normal(0, 1, 3)
        # Make it perpendicular to the center
        perp1 = perp1 - np.dot(perp1, continent_center) * continent_center
        perp1 = perp1 / np.linalg.norm(perp1)
        
        # Second perpendicular vector to define a plane
        perp2 = np.cross(continent_center, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Find all continental vertices
        continental_vertices = []
        for i in range(len(self.crust_type)):
            if self.crust_type[i] == 1:
                continental_vertices.append(i)
        
        # Find vertices near the breakup axis
        rift_vertices = []
        
        for i in continental_vertices:
            vertex = self.planet.grid.vertices[i]
            
            # Project onto the plane defined by perp1 and perp2
            # This creates a great circle path for rifting
            proj1 = np.dot(vertex, perp1)
            proj2 = np.dot(vertex, perp2)
            
            # If close to the plane, mark for rifting
            if abs(proj1) < 0.1:
                rift_vertices.append(i)
        
        # Create rifts along the breakup axis
        for i in rift_vertices:
            # Lower elevation to create rift valley
            self.planet.elevation[i] -= 1.0 + 0.5 * np.random.random()
            
            # Thin the crust
            self.crust_thickness[i] *= 0.8
            
            # Increase boundary stress to encourage plate separation
            self.boundary_stress[i] += 5.0
            
            # Some locations convert to oceanic crust as breakup progresses
            if np.random.random() < 0.2:
                self.crust_type[i] = 0  # Oceanic
                self.crust_thickness[i] = self.AVG_OCEANIC_THICKNESS
                self.crust_density[i] = self.OCEANIC_DENSITY
                self.crust_age[i] = 0
        
        # Break affected plates along the rift
        for plate in self.plates:
            # Check if this plate has rift vertices
            plate_rift_vertices = [v for v in rift_vertices if v in plate['vertices']]
            
            if len(plate_rift_vertices) > 10 and len(plate_rift_vertices) < len(plate['vertices']) * 0.8:
                self._break_plate_along_rift(plate, plate_rift_vertices, perp1)
    
    def _break_plate_along_rift(self, plate, rift_vertices, rift_direction):
        """Break a plate along a specified rift direction"""
        # Create a new plate ID
        new_plate_id = len(self.plates)
        
        # Divide vertices into two groups based on which side of the rift they're on
        side1_vertices = []
        side2_vertices = []
        
        for vertex_idx in plate['vertices']:
            if vertex_idx in rift_vertices:
                # On the rift itself - randomly assign
                if np.random.random() < 0.5:
                    side1_vertices.append(vertex_idx)
                else:
                    side2_vertices.append(vertex_idx)
                continue
                
            # Not on rift - check which side it's on
            vertex = self.planet.grid.vertices[vertex_idx]
            
            # Dot product with rift direction determines side
            side = np.dot(vertex, rift_direction)
            
            if side > 0:
                side1_vertices.append(vertex_idx)
            else:
                side2_vertices.append(vertex_idx)
        
        # Ensure both sides have vertices
        if len(side1_vertices) == 0 or len(side2_vertices) == 0:
            return
        
        # Assign new plate ID to side2
        for vertex_idx in side2_vertices:
            self.planet.plate_ids[vertex_idx] = new_plate_id
        
        # Update the original plate to side1 only
        plate['vertices'] = side1_vertices
        plate['area'] = len(side1_vertices) / len(self.planet.grid.vertices)
        
        # Calculate continental percentage
        continental_count = sum(1 for v in side1_vertices if self.crust_type[v] == 1)
        plate['continental_percentage'] = continental_count / len(side1_vertices) * 100
        
        # Create the new plate
        # Calculate center
        center_pos = np.zeros(3)
        for vertex_idx in side2_vertices:
            center_pos += self.planet.grid.vertices[vertex_idx]
        center_pos /= len(side2_vertices)
        center_pos = center_pos / np.linalg.norm(center_pos)
        
        # Determine if plate is oceanic
        continental_count = sum(1 for v in side2_vertices if self.crust_type[v] == 1)
        is_oceanic = continental_count < len(side2_vertices) / 2
        
        # Create velocity that diverges from the rift
        # First get current velocity
        parent_vel = np.array(plate['velocity'])
        # Add component away from rift
        rift_component = rift_direction * 0.01 * np.random.random()
        if np.dot(center_pos, rift_direction) < 0:
            rift_component = -rift_component
        
        new_vel = parent_vel + rift_component
        
        # Create the new plate
        new_plate = {
            'id': new_plate_id,
            'center': center_pos,
            'vertices': side2_vertices,
            'is_oceanic': is_oceanic,
            'velocity': new_vel,
            'area': len(side2_vertices) / len(self.planet.grid.vertices),
            'age': plate['age'],
            'continental_percentage': continental_count / len(side2_vertices) * 100,
            'boundaries': [],
            'features': {
                'orogenic_belts': [],  # Mountain ranges
                'rifts': [],           # Spreading centers
                'volcanic_arcs': [],   # Subduction volcanoes
                'hotspots': [],        # Intraplate volcanic centers
                'transform_zones': []  # Transform fault zones
            }
        }
        
        # Also adjust the original plate's velocity
        plate['velocity'] = parent_vel - rift_component
        
        self.plates.append(new_plate)
        
        # Update boundaries
        self._update_plate_boundaries()
    
    def _increase_continental_convergence(self):
        """Increase continental convergence to promote supercontinent assembly"""
        # Find continental plates
        continental_plates = []
        for plate in self.plates:
            if plate['continental_percentage'] > 50:
                continental_plates.append(plate)
        
        if len(continental_plates) < 2:
            return
        
        # Calculate mean continental position
        mean_pos = np.zeros(3)
        for plate in continental_plates:
            mean_pos += plate['center']
        mean_pos = mean_pos / np.linalg.norm(mean_pos)
        
        # Adjust plate velocities to increase convergence
        for plate in continental_plates:
            # Calculate vector toward mean
            toward_mean = mean_pos - plate['center']
            toward_mean = toward_mean - np.dot(toward_mean, plate['center']) * plate['center']
            toward_mean = toward_mean / np.linalg.norm(toward_mean) * 0.002
            
            # Adjust velocity to move more toward mean
            plate['velocity'] = plate['velocity'] * 0.9 + toward_mean
        
        self._add_event(f"Adjusted continental plate motions to promote convergence at {self.planet.age} Ma")
    
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
        remaining_vertices = []
        
        for vertex_idx in plate['vertices']:
            vertex = self.planet.grid.vertices[vertex_idx]
            # Check which side of the plane the vertex is on
            if np.dot(vertex, split_axis) > 0:
                self.planet.plate_ids[vertex_idx] = new_plate_id
                new_plate_vertices.append(vertex_idx)
            else:
                remaining_vertices.append(vertex_idx)
        
        # Make sure we have vertices in both plates
        if len(new_plate_vertices) == 0 or len(remaining_vertices) == 0:
            return  # Can't split, skip this operation
        
        # Update the original plate
        plate['vertices'] = remaining_vertices
        plate['area'] = len(remaining_vertices) / len(self.planet.grid.vertices)
        
        # Update continental percentage for original plate
        continental_count = sum(1 for v in remaining_vertices if self.crust_type[v] == 1)
        plate['continental_percentage'] = continental_count / len(remaining_vertices) * 100 if remaining_vertices else 0
        
        # Calculate center for new plate
        center_pos = np.zeros(3)
        for vertex_idx in new_plate_vertices:
            center_pos += self.planet.grid.vertices[vertex_idx]
        center_pos /= len(new_plate_vertices)
        center_pos = center_pos / np.linalg.norm(center_pos)
        
        # Determine if new plate is oceanic
        continental_count = sum(1 for v in new_plate_vertices if self.crust_type[v] == 1)
        is_oceanic = continental_count < len(new_plate_vertices) / 2
        
        # Create velocity similar to parent but with some deviation
        parent_vel = np.array(plate['velocity'])
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
                'hotspots': [],        # Intraplate volcanic centers
                'transform_zones': []  # Transform fault zones
            }
        }
        
        self.plates.append(new_plate)
        
        # Update boundaries
        self._update_plate_boundaries()
        
        self._add_event(f"Plate {plate['id']} split into plates {plate['id']} and {new_plate_id}")
        
    def _merge_plates(self, plate1, plate2):
        """Merge two plates into one (smaller one gets absorbed)"""
        # Determine which plate absorbs the other
        if len(plate1['vertices']) >= len(plate2['vertices']):
            main_plate = plate1
            absorbed_plate = plate2
        else:
            main_plate = plate2
            absorbed_plate = plate1
        
        # Update plate IDs for all vertices in the absorbed plate
        for vertex_idx in absorbed_plate['vertices']:
            self.planet.plate_ids[vertex_idx] = main_plate['id']
        
        # Add vertices to the main plate
        main_plate['vertices'].extend(absorbed_plate['vertices'])
        
        # Update main plate properties
        main_plate['area'] = len(main_plate['vertices']) / len(self.planet.grid.vertices)
        
        # Update continental percentage
        continental_count = sum(1 for v in main_plate['vertices'] if self.crust_type[v] == 1)
        main_plate['continental_percentage'] = continental_count / len(main_plate['vertices']) * 100
        
        # Update center of mass
        center_pos = np.zeros(3)
        for vertex_idx in main_plate['vertices']:
            center_pos += self.planet.grid.vertices[vertex_idx]
        center_pos /= len(main_plate['vertices'])
        main_plate['center'] = center_pos / np.linalg.norm(center_pos)
        
        # Modify velocity (weighted average)
        weight1 = len(plate1['vertices']) / (len(plate1['vertices']) + len(plate2['vertices']))
        weight2 = 1.0 - weight1
        main_plate['velocity'] = (weight1 * np.array(plate1['velocity']) + 
                               weight2 * np.array(plate2['velocity']))
        
        # Remove the absorbed plate from the plates list
        self.plates = [p for p in self.plates if p['id'] != absorbed_plate['id']]
        
        # Update boundaries
        self._update_plate_boundaries()
        
        self._add_event(f"Plates {plate1['id']} and {plate2['id']} merged")
    
    def _record_state(self):
        """Record the current state of the planet for history tracking"""
        self.history['ages'].append(self.planet.age)
        
        # Record plate positions and velocities
        plate_positions = []
        plate_velocities = []
        
        for plate in self.plates:
            plate_positions.append(plate['center'])
            plate_velocities.append(plate['velocity'])
        
        self.history['plate_positions'].append(plate_positions)
        self.history['plate_velocities'].append(plate_velocities)
        
        # Track elevation extremes
        self.history['elevations'].append([float(np.min(self.planet.elevation)), 
                                        float(np.max(self.planet.elevation))])
        
        # Count number of plates
        self.history['plate_counts'].append(len(self.plates))
        
        # Calculate continental area
        continental_vertices = np.sum(self.crust_type == 1)
        continental_area = continental_vertices / len(self.crust_type) * 100
        self.history['continental_area'].append(float(continental_area))
        
        # Calculate supercontinent index - ADDED/MODIFIED
        continental_coords = []
        for i in range(len(self.planet.elevation)):
            if self.crust_type[i] == 1:  # Continental
                continental_coords.append(self.planet.grid.vertices[i])
        
        # Calculate mean position of all continental crust
        if continental_coords:
            mean_pos = np.mean(continental_coords, axis=0)
            mean_pos = mean_pos / np.linalg.norm(mean_pos)
            
            # Calculate average distance from mean
            distances = []
            for pos in continental_coords:
                norm_pos = pos / np.linalg.norm(pos)
                dot = np.clip(np.dot(norm_pos, mean_pos), -1.0, 1.0)
                distances.append(np.arccos(dot))
            
            dispersion = np.mean(distances)
            self.history['supercontinent_index'].append(1.0 - dispersion / np.pi)
        else:
            # No continental crust, use default value
            self.history['supercontinent_index'].append(0.0)
    
    def _add_event(self, description):
        """Record a significant geological event"""
        event = {
            'age': float(self.planet.age),
            'description': description,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add to both lists
        self.recent_events.append(event)
        self.all_events.append(event)
        
        # Limit total events stored to avoid memory issues
        if len(self.all_events) > 1000:
            self.all_events = self.all_events[-1000:]
    
    def get_recent_events(self):
        """Get list of recent geological events"""
        return self.recent_events
    
    def get_all_events(self):
        """Get list of all recorded geological events"""
        return self.all_events
    
    def visualize_plates(self, save_path=None, show=False, show_features=False):
        """
        Visualize the tectonic plates configuration in 3D.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - show: Whether to display the plot
        - show_features: Whether to highlight geological features
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a unique color for each plate
        plate_colors = {}
        for plate in self.plates:
            hue = (plate['id'] * 0.618033988749895) % 1.0  # Golden ratio produces nicely distributed colors
            plate_colors[plate['id']] = plt.cm.hsv(hue)
        
        # Create arrays for plotting
        xs, ys, zs = [], [], []
        colors = []
        
        # Map vertices to positions and colors
        for i, vertex in enumerate(self.planet.grid.vertices):
            # Apply elevation to position
            elevation = self.planet.elevation[i]
            scaled_vertex = vertex * (1 + elevation/self.planet.radius)
            
            xs.append(scaled_vertex[0])
            ys.append(scaled_vertex[1])
            zs.append(scaled_vertex[2])
            
            # Color based on plate ID
            plate_id = self.planet.plate_ids[i]
            if plate_id >= 0 and plate_id < len(self.plates):
                colors.append(plate_colors[plate_id])
            else:
                colors.append([0.7, 0.7, 0.7])  # Gray for unassigned
        
        # Plot the vertices
        ax.scatter(xs, ys, zs, c=colors, s=10, alpha=0.8)
        
        # Mark plate boundaries
        if self.plate_boundaries is not None:
            boundary_xs = []
            boundary_ys = []
            boundary_zs = []
            
            for i, is_boundary in enumerate(self.plate_boundaries):
                if is_boundary:
                    vertex = self.planet.grid.vertices[i]
                    elevation = self.planet.elevation[i]
                    scaled_vertex = vertex * (1 + elevation/self.planet.radius)
                    
                    boundary_xs.append(scaled_vertex[0])
                    boundary_ys.append(scaled_vertex[1])
                    boundary_zs.append(scaled_vertex[2])
            
            ax.scatter(boundary_xs, boundary_ys, boundary_zs, c='black', s=15, alpha=0.5)
        
        # Show plate motion vectors if requested
        if show_features:
            for plate in self.plates:
                # Get the plate center and velocity
                center = np.array(plate['center']) * self.planet.radius
                vel = np.array(plate['velocity']) * self.planet.radius * 20  # Scale for visibility
                
                # Plot the velocity vector
                ax.quiver(center[0], center[1], center[2], 
                        vel[0], vel[1], vel[2], 
                        color='red', length=1.0, normalize=True, arrow_length_ratio=0.3)
                
                # Mark special features
                for feature_type, features in plate['features'].items():
                    if not features:
                        continue
                        
                    if feature_type == 'orogenic_belts':
                        # Mark mountain ranges
                        for other_plate_id in features:
                            # Find mountain vertices at the boundary between plates
                            for i, is_boundary in enumerate(self.plate_boundaries):
                                if is_boundary and self.planet.plate_ids[i] == plate['id']:
                                    # Check if this is a boundary with the other plate
                                    neighbors = self.planet.grid.get_vertex_neighbors()[i]
                                    for n in neighbors:
                                        if self.planet.plate_ids[n] == other_plate_id:
                                            # This is part of the mountain range
                                            vertex = self.planet.grid.vertices[i]
                                            elevation = self.planet.elevation[i]
                                            scaled_vertex = vertex * (1 + elevation/self.planet.radius)
                                            
                                            # Only show if it's actually a mountain
                                            if elevation > 1.0:
                                                ax.scatter([scaled_vertex[0]], [scaled_vertex[1]], [scaled_vertex[2]], 
                                                         color='white', s=20, alpha=0.7, marker='^')
                                            break
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        ax.set_axis_off()
        
        # Add title
        plt.title(f'Tectonic Plates at {self.planet.age:.1f} million years', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            if not show:
                plt.close(fig)
            print(f"Plate visualization saved to {save_path}")
            
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_plates_2d(self, save_path=None, show=False, projection='mercator', show_features=False):
        """
        Visualize tectonic plates as a 2D map using enhanced visualization.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - show: Whether to display the plot
        - projection: Map projection ('mercator', 'equirectangular', etc.)
        - show_features: Whether to highlight geological features
        """
        visualizer = MapVisualizer(self.planet)
        return visualizer.visualize_plates(
            save_path=save_path,
            show=show,
            projection=projection,
            show_boundaries=True,
            show_arrows=show_features
        )
    
    def visualize_history(self, save_path=None, show=False):
        """
        Visualize the history of tectonic evolution.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - show: Whether to display the plot
        """
        import matplotlib.pyplot as plt
        
        # Check if we have enough history data
        if len(self.history['ages']) < 2:
            print("Not enough history data for visualization.")
            return None
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot 1: Elevation extremes
        elevations = np.array(self.history['elevations'])
        axes[0].fill_between(self.history['ages'], elevations[:, 0], elevations[:, 1], 
                           color='skyblue', alpha=0.5)
        axes[0].plot(self.history['ages'], elevations[:, 0], 'b-', label='Minimum elevation')
        axes[0].plot(self.history['ages'], elevations[:, 1], 'r-', label='Maximum elevation')
        axes[0].set_ylabel('Elevation (km)')
        axes[0].set_title('Elevation Range Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Number of plates and continental area
        ax2 = axes[1]
        ax2.plot(self.history['ages'], self.history['plate_counts'], 'g-', label='Number of plates')
        ax2.set_ylabel('Number of plates', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(True, alpha=0.3)
        
        # Add continental area on secondary y-axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.history['ages'], self.history['continental_area'], 'm-', label='Continental area')
        ax2_twin.set_ylabel('Continental area (%)', color='m')
        ax2_twin.tick_params(axis='y', labelcolor='m')
        
        # Add combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 3: Supercontinent index
        if self.history['supercontinent_index']:
            axes[2].plot(self.history['ages'], self.history['supercontinent_index'], 'k-')
            axes[2].set_ylabel('Supercontinent index')
            axes[2].set_title('Continental Aggregation')
            axes[2].grid(True, alpha=0.3)
            
            # Add horizontal line at supercontinent threshold
            axes[2].axhline(y=1.0 - self.SUPERCONTINENT_THRESHOLD/np.pi, 
                          color='r', linestyle='--', alpha=0.5, 
                          label='Supercontinent threshold')
            axes[2].legend()
        
        # Add events as vertical lines with annotations
        for i, event in enumerate(self.all_events[-10:]):  # Show last 10 events
            for ax in axes:
                ax.axvline(x=event['age'], color='gray', linestyle=':', alpha=0.5)
            
            # Add annotation to top subplot, alternating top/bottom for readability
            y_pos = 0.9 - (i % 3) * 0.1
            axes[0].annotate(event['description'], 
                           xy=(event['age'], axes[0].get_ylim()[1] * y_pos),
                           xytext=(10, 0), textcoords='offset points',
                           fontsize=8, rotation=90, color='gray')
        
        # Add axis labels
        axes[2].set_xlabel('Age (million years ago)')
        
        # Set x-axis to run from oldest to newest
        for ax in axes:
            ax.invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            if not show:
                plt.close(fig)
            print(f"Tectonic history visualization saved to {save_path}")
            
        if show:
            plt.show()
        
        return save_path if save_path else None
    
def visualize_plates_2d(self, save_path=None, show=False, projection='mercator', show_features=False):
    """
    Visualize tectonic plates as a 2D map projection with enhanced clarity.
    
    Parameters:
    - save_path: If provided, save the visualization to this file path
    - show: Whether to display the plot
    - projection: Map projection ('mercator', 'equirectangular', 'mollweide')
    - show_features: Whether to highlight geological features
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    # Setup the projection
    width, height = 1200, 600  # Pixels for projection calculation
    
    # Create a unique color for each plate
    plate_colors = {}
    for plate in self.plates:
        hue = (plate['id'] * 0.618033988749895) % 1.0  # Golden ratio for color distribution
        plate_colors[plate['id']] = plt.cm.hsv(hue)
    
    # Draw each plate with a unique color
    for i, vertex in enumerate(self.planet.grid.vertices):
        plate_id = self.planet.plate_ids[i]
        if plate_id >= 0 and plate_id < len(self.plates):
            # Convert vertex to lat/lon
            x, y, z = vertex
            lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
            lon = np.arctan2(y, x) * 180 / np.pi
            
            # Project to 2D based on projection type
            if projection == 'mercator':
                px, py = self._lat_lon_to_mercator(lat, lon, width, height)
            else:  # Default to equirectangular
                px = (lon + 180) * width / 360
                py = (90 - lat) * height / 180
            
            # Plot point with plate color
            if px >= 0 and px < width and py >= 0 and py < height:
                ax.scatter(px, py, c=[plate_colors[plate_id]], s=5, alpha=0.8)
    
    # Draw plate boundaries
    self._draw_plate_boundaries_2d(ax, width, height, projection, plate_colors)
    
    # Draw plate velocity vectors
    if show_features:
        self._draw_plate_velocities_2d(ax, width, height, projection, plate_colors)
    
    # Set axis limits
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis for standard map orientation
    
    # Add title and grid
    plt.title(f'Tectonic Plates at {self.planet.age:.1f} million years')
    plt.grid(True, alpha=0.3)
    
    # Add legend for plate types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Convergent Boundary'),
        Line2D([0], [0], color='blue', lw=2, label='Divergent Boundary'),
        Line2D([0], [0], color='purple', linestyle='--', lw=2, label='Transform Boundary')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        if not show:
            plt.close(fig)
        print(f"2D plate visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    return save_path if save_path else None

def _draw_plate_boundaries_2d(self, ax, width, height, projection, plate_colors):
    """Draw plate boundaries on 2D map with correct styling by boundary type"""
    
    # Get neighbors for boundary detection
    neighbors = self.planet.grid.get_vertex_neighbors()
    
    # For each vertex, check if it's at a boundary
    for i, vertex in enumerate(self.planet.grid.vertices):
        plate_id = self.planet.plate_ids[i]
        
        # Skip if not assigned to a plate
        if plate_id == -1 or plate_id >= len(self.plates):
            continue
        
        # Check neighbors for different plates
        for neighbor in neighbors[i]:
            neighbor_plate = self.planet.plate_ids[neighbor]
            
            if neighbor_plate != plate_id and neighbor_plate != -1 and neighbor_plate < len(self.plates):
                # This is a boundary - determine type based on relative motion
                boundary_type = self._calculate_boundary_type(i, plate_id, neighbor_plate)
                
                # Get coordinates for both vertices
                x1, y1, z1 = self.planet.grid.vertices[i]
                x2, y2, z2 = self.planet.grid.vertices[neighbor]
                
                lat1 = np.arcsin(z1 / np.linalg.norm(self.planet.grid.vertices[i])) * 180 / np.pi
                lon1 = np.arctan2(y1, x1) * 180 / np.pi
                
                lat2 = np.arcsin(z2 / np.linalg.norm(self.planet.grid.vertices[neighbor])) * 180 / np.pi
                lon2 = np.arctan2(y2, x2) * 180 / np.pi
                
                # Project to 2D based on projection type
                if projection == 'mercator':
                    px1, py1 = self._lat_lon_to_mercator(lat1, lon1, width, height)
                    px2, py2 = self._lat_lon_to_mercator(lat2, lon2, width, height)
                else:  # Default to equirectangular
                    px1 = (lon1 + 180) * width / 360
                    py1 = (90 - lat1) * height / 180
                    px2 = (lon2 + 180) * width / 360
                    py2 = (90 - lat2) * height / 180
                
                # Skip if coordinates are outside the bounds
                if (px1 < 0 or px1 >= width or py1 < 0 or py1 >= height or 
                    px2 < 0 or px2 >= width or py2 < 0 or py2 >= height):
                    continue
                
                # Draw line with appropriate style based on boundary type
                if boundary_type == "convergent":
                    ax.plot([px1, px2], [py1, py2], 'r-', linewidth=1.5, alpha=0.8)
                elif boundary_type == "divergent":
                    ax.plot([px1, px2], [py1, py2], 'b-', linewidth=1.5, alpha=0.8)
                elif boundary_type == "transform":
                    ax.plot([px1, px2], [py1, py2], 'purple', linestyle='--', linewidth=1.5, alpha=0.8)
                else:
                    ax.plot([px1, px2], [py1, py2], 'k-', linewidth=0.5, alpha=0.5)

def _draw_plate_velocities_2d(self, ax, width, height, projection, plate_colors):
    """Draw plate velocity vectors on 2D map"""
    import numpy as np
    
    for plate in self.plates:
        # Convert center to lat/lon
        x, y, z = plate['center']
        lat = np.arcsin(z / np.linalg.norm(plate['center'])) * 180 / np.pi
        lon = np.arctan2(y, x) * 180 / np.pi
        
        # Project to 2D based on projection type
        if projection == 'mercator':
            px, py = self._lat_lon_to_mercator(lat, lon, width, height)
        else:  # Default to equirectangular
            px = (lon + 180) * width / 360
            py = (90 - lat) * height / 180
        
        # Skip if outside bounds
        if px < 0 or px >= width or py < 0 or py >= height:
            continue
        
        # Calculate velocity components in 3D
        vel_mag = np.linalg.norm(plate['velocity']) * 5000  # Scale for visibility
        vel_dir = plate['velocity'] / np.linalg.norm(plate['velocity']) if np.linalg.norm(plate['velocity']) > 0 else np.array([0, 0, 0])
        
        # Project velocity to 2D (simplified approach)
        vx = vel_dir[0] * vel_mag
        vy = vel_dir[1] * vel_mag
        
        # Draw arrow
        ax.arrow(px, py, vx, -vy, head_width=10, head_length=15, 
                 fc=plate_colors[plate['id']], ec=plate_colors[plate['id']], 
                 length_includes_head=True)
        
        # Add plate label
        is_oceanic = plate['is_oceanic'] if 'is_oceanic' in plate else False
        type_label = " (Oceanic)" if is_oceanic else " (Continental)"
        ax.text(px, py-20, f"Plate {plate['id']}{type_label}", 
                fontsize=8, ha='center', va='center', 
                color=plate_colors[plate['id']])

def _lat_lon_to_mercator(self, lat, lon, width, height):
    """Convert latitude/longitude to Mercator projection x,y coordinates"""
    import math
    
    # Constrain latitude to prevent infinite scaling near poles
    lat = max(min(lat, 85), -85)
    
    # Mercator projection formulas
    x = (lon + 180) * (width / 360)
    
    # Convert latitude to radians
    lat_rad = math.radians(lat)
    
    # Mercator y-coordinate formula
    merc_n = math.log(math.tan((math.pi / 4) + (lat_rad / 2)))
    y = (height / 2) - (width * merc_n / (2 * math.pi))
    
    return x, y