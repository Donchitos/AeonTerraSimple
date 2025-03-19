import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import datetime
import math

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
        self.plate_boundaries = None
        
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
    
    # Generate regions where early continental crust forms using coherent noise
    # Continental crust forms through partial melting and reprocessing
    proto_continents = self._generate_coherent_noise(octaves=3, scale=3.0)
    
    # Add bias to create more coherent continents (rather than scattered fragments)
    # Apply a smoothing filter to the noise
    from scipy.ndimage import gaussian_filter
    
    # Convert to grid for smoothing
    grid_size = int(np.sqrt(len(proto_continents))) * 2
    grid = np.zeros((grid_size, grid_size))
    
    # Map values to grid
    for i, val in enumerate(proto_continents):
        # Convert vertex to lat-lon coordinates
        x, y, z = self.planet.grid.vertices[i]
        lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[i]))
        lon = np.arctan2(y, x)
        
        # Map to grid coordinates
        grid_x = int((lon + np.pi) / (2 * np.pi) * grid_size) % grid_size
        grid_y = int((lat + np.pi/2) / np.pi * grid_size/2) % grid_size
        
        # Set value
        grid[grid_y, grid_x] = val
    
    # Apply smoothing to create more coherent continents
    smoothed_grid = gaussian_filter(grid, sigma=3.0)
    
    # Map back to vertices
    for i in range(len(proto_continents)):
        x, y, z = self.planet.grid.vertices[i]
        lat = np.arcsin(z / np.linalg.norm(self.planet.grid.vertices[i]))
        lon = np.arctan2(y, x)
        
        grid_x = int((lon + np.pi) / (2 * np.pi) * grid_size) % grid_size
        grid_y = int((lat + np.pi/2) / np.pi * grid_size/2) % grid_size
        
        proto_continents[i] = smoothed_grid[grid_y, grid_x]
    
    # Create 2-4 major continental blocks
    num_blocks = np.random.randint(2, 5)
    continental_seeds = []
    
    for _ in range(num_blocks):
        # Random point on sphere
        theta = np.random.random() * 2 * np.pi
        phi = np.arccos(2 * np.random.random() - 1)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        continental_seeds.append(np.array([x, y, z]))
    
    # Boost continental crust formation near seeds
    for i, vertex in enumerate(self.planet.grid.vertices):
        # Find distance to closest continental seed
        min_dist = float('inf')
        
        for seed in continental_seeds:
            # Calculate angular distance
            dot = np.clip(np.dot(vertex, seed) / 
                        (np.linalg.norm(vertex) * np.linalg.norm(seed)), -1.0, 1.0)
            dist = np.arccos(dot)
            min_dist = min(min_dist, dist)
        
        # Boost proto_continents value based on proximity to continental seed
        # This makes continents more likely to form near the seeds
        proximity_factor = max(0, 1.0 - min_dist / np.pi * 2)
        proto_continents[i] += proximity_factor * 0.5
    
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
    
    # Print continental coverage
    continental_percentage = np.sum(continental_mask) / len(continental_mask) * 100
    print(f"  Generated {continental_percentage:.1f}% continental crust")
    
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
    
    # Step 5: Ensure continental crust emerges properly
    self.ensure_continental_emergence()
    
    # Record initial state in history
    self._record_state()
    
    # Add initial event
    self._add_event("Initial planetary configuration established")
    
    print(f"Initialized {len(self.plates)} tectonic plates")
    return self
        
def _generate_plate_boundaries(self, custom_num_plates=None):
    """Generate plate boundaries using Voronoi cells on the sphere with continental bias"""
    num_plates = custom_num_plates if custom_num_plates is not None else self.num_plates
    
    # Calculate desired number of continental and oceanic plates
    target_continental_plates = max(1, int(num_plates * self.target_continental_fraction * 1.5))
    oceanic_plates = num_plates - target_continental_plates
    
    print(f"  Targeting {target_continental_plates} continental plates and {oceanic_plates} oceanic plates")
    
    # Get indices of continental and oceanic crust
    continental_indices = np.where(self.crust_type == 1)[0]
    oceanic_indices = np.where(self.crust_type == 0)[0]
    
    # Ensure we have continental crust
    if len(continental_indices) == 0:
        print("  Warning: No continental crust found. Generating continental crust...")
        # Force creation of some continental crust
        random_vertices = np.random.choice(
            range(len(self.planet.grid.vertices)), 
            size=int(len(self.planet.grid.vertices) * self.target_continental_fraction),
            replace=False
        )
        self.crust_type[random_vertices] = 1
        self.crust_thickness[random_vertices] = self.AVG_CONTINENTAL_THICKNESS
        self.crust_density[random_vertices] = self.CONTINENTAL_DENSITY
        continental_indices = random_vertices
    
    # Choose seed points for continental plates from continental crust regions
    seed_points = []
    
    # Continental plate seeds first
    for _ in range(target_continental_plates):
        if len(continental_indices) > 0:
            # Select a random continental vertex
            idx = np.random.choice(continental_indices)
            seed_points.append(self.planet.grid.vertices[idx])
            
            # Remove nearby continental vertices to avoid clustering
            pos = self.planet.grid.vertices[idx]
            distances = np.array([
                np.arccos(np.clip(np.dot(pos, self.planet.grid.vertices[i]) / 
                               (np.linalg.norm(pos) * np.linalg.norm(self.planet.grid.vertices[i])), 
                               -1.0, 1.0))
                for i in continental_indices
            ])
            mask = distances > 0.5  # Remove continental vertices within ~30° arc distance
            if np.any(mask):  # Check if mask has any True values
                continental_indices = continental_indices[mask]
        else:
            # If we run out of continental vertices, just choose randomly
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            seed_points.append([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
    
    # Oceanic plate seeds
    for _ in range(oceanic_plates):
        if len(oceanic_indices) > 0:
            # Select a random oceanic vertex
            idx = np.random.choice(oceanic_indices)
            seed_points.append(self.planet.grid.vertices[idx])
            
            # Remove nearby oceanic vertices to avoid clustering
            pos = self.planet.grid.vertices[idx]
            distances = np.array([
                np.arccos(np.clip(np.dot(pos, self.planet.grid.vertices[i]) / 
                               (np.linalg.norm(pos) * np.linalg.norm(self.planet.grid.vertices[i])), 
                               -1.0, 1.0))
                for i in oceanic_indices
            ])
            mask = distances > 0.5  # Remove oceanic vertices within ~30° arc distance
            if np.any(mask):  # Check if mask has any True values
                oceanic_indices = oceanic_indices[mask]
        else:
            # If we run out of oceanic vertices, just choose randomly
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            seed_points.append([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
    
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
    
    # Metrics for monitoring
    continental_plates_count = 0
    oceanic_plates_count = 0
    
    # Iterate through possible plate IDs and create plates for ones with vertices
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
        continental_percentage = continental_crust_count / len(plate_vertices) * 100
        
        # More balanced classification - use a lower threshold for continents
        # A plate with 30% or more continental crust is considered continental
        # This represents mixed plates like Eurasia which have significant oceanic components
        is_oceanic = continental_percentage < 30.0
        
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
            oceanic_plates_count += 1
        else:
            vel_factor = 0.8 * velocity_scale
            max_vel = self.PLATE_VELOCITY_CONTINENTAL_MAX / 100.0  # Convert cm/year to simulation units
            continental_plates_count += 1
        
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
            'continental_percentage': continental_percentage,
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
    
    print(f"  Created {continental_plates_count} continental plates and {oceanic_plates_count} oceanic plates")
    
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
                # Simplified boundary handling to reduce code complexity
                
                # Calculate boundary type with stress
                boundary_type, stress = self._calculate_boundary_forces(i, my_plate_id, neighbor_plate_id, time_step)
                
                # Apply different effects based on boundary type
                if boundary_type == "convergent":
                    # Convergent boundaries create mountains or trenches
                    is_continental1 = self.crust_type[i] == 1
                    
                    # Find neighbor vertex from other plate
                    other_vertex = None
                    for n in neighbors[i]:
                        if self.planet.plate_ids[n] == neighbor_plate_id:
                            other_vertex = n
                            break
                    
                    if other_vertex is not None:
                        is_continental2 = self.crust_type[other_vertex] == 1
                        
                        # Continental-continental collision forms mountains
                        if is_continental1 and is_continental2:
                            # Increase elevation (mountain building)
                            self.planet.elevation[i] += 0.2 * time_step / 10.0
                            # Thicken crust
                            self.crust_thickness[i] += 0.5 * time_step / 10.0
                        
                        # Oceanic-continental creates subduction zone
                        elif not is_continental1 and is_continental2:
                            # Create trench
                            self.planet.elevation[i] -= 0.2 * time_step / 10.0
                        
                        # Continental-oceanic also creates subduction zone, but on the other plate
                        elif is_continental1 and not is_continental2:
                            # Chance for volcanic uplift
                            if np.random.random() < 0.1 * time_step / 10.0:
                                self.planet.elevation[i] += 0.3 * np.random.random()
                        
                        # Oceanic-oceanic creates island arc system
                        else:
                            # Older crust subducts
                            if self.crust_age[i] > self.crust_age[other_vertex]:
                                self.planet.elevation[i] -= 0.3 * time_step / 10.0
                                
                                # Small chance for volcanic island formation
                                if np.random.random() < 0.02 * time_step / 10.0:
                                    self.planet.elevation[other_vertex] = max(0, self.planet.elevation[other_vertex])
                                    self.crust_type[other_vertex] = 1  # Convert to continental
                            else:
                                self.planet.elevation[other_vertex] -= 0.3 * time_step / 10.0
                                
                                # Small chance for volcanic island formation
                                if np.random.random() < 0.02 * time_step / 10.0:
                                    self.planet.elevation[i] = max(0, self.planet.elevation[i])
                                    self.crust_type[i] = 1  # Convert to continental
                
                elif boundary_type == "divergent":
                    # Divergent boundaries create new crust
                    if self.crust_type[i] == 0:  # Oceanic
                        # Reset age for new crust
                        self.crust_age[i] = 0
                    else:  # Continental
                        # Continental rifting - thinning crust
                        self.crust_thickness[i] = max(self.AVG_OCEANIC_THICKNESS, 
                                                     self.crust_thickness[i] - 0.5 * time_step / 10.0)
                        
                        # If sufficiently thin, convert to oceanic
                        if self.crust_thickness[i] < 15.0 and np.random.random() < 0.1 * time_step / 10.0:
                            self.crust_type[i] = 0  # Oceanic
                            self.crust_density[i] = self.OCEANIC_DENSITY
                            self.crust_age[i] = 0
                            self.planet.elevation[i] = -1.5  # Initial seafloor
                
                elif boundary_type == "transform":
                    # Transform boundaries just accumulate stress
                    # Occasionally release as earthquakes
                    if stress > 5.0 and np.random.random() < 0.05 * time_step / 10.0:
                        # Stress release
                        self.boundary_stress[i] *= 0.2
    
    def _calculate_boundary_forces(self, vertex_idx, plate1_id, plate2_id, time_step):
        """Calculate forces and stress at plate boundaries"""
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
        """Process mantle plumes and hotspot volcanism"""
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
                                self.planet.elevation[i] = 0.5 + 1.0 * np.random.random()  # Island
                                
                                # Islands have thin continental crust
                                self.crust_type[i] = 1
                                self.crust_thickness[i] = 15.0 + 5.0 * np.random.random()
                                self.crust_density[i] = self.CONTINENTAL_DENSITY
                                
                                # Record island position in hotspot track
                                hotspot['track'].append(list(vertex))
                                
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
    
    def _age_crust(self, time_step):
        """Age the crust and apply scientific processes related to aging"""
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
                    
                    # Record this breakup
                    self.last_supercontinent_breakup = self.planet.age
            else:
                # Continents are dispersed - possibility of starting convergence
                if np.random.random() < 0.03 * time_step / 10.0:
                    self._add_event(f"Continents beginning to converge at {self.planet.age} million years")
    
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
    
    def lat_lon_to_mercator(self, lat, lon, width, height):
        """Convert latitude/longitude to Mercator projection x,y coordinates"""
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
    
def ensure_continental_emergence(self):
    """
    Ensure that continental crust emerges above sea level
    by adjusting isostatic equilibrium calculations
    """
    print("Ensuring continental crust emerges above sea level...")
    
    # Get continental vertices
    continental_vertices = np.where(self.crust_type == 1)[0]
    
    if len(continental_vertices) == 0:
        print("  No continental crust found, cannot adjust elevations")
        return
    
    # Get current elevation stats
    cont_elevations = self.planet.elevation[continental_vertices]
    min_cont_elev = np.min(cont_elevations)
    max_cont_elev = np.max(cont_elevations)
    mean_cont_elev = np.mean(cont_elevations)
    
    print(f"  Continental elevation before adjustment: {min_cont_elev:.2f} to {max_cont_elev:.2f}, mean: {mean_cont_elev:.2f}")
    
    # If average continental elevation is below sea level, adjust
    if mean_cont_elev < 0.2:  # We want continents to be significantly above sea level
        # Calculate adjustment needed - we want mean elevation around 0.8 km
        adjustment = 0.8 - mean_cont_elev
        
        # Apply adjustment to all continental crust
        self.planet.elevation[continental_vertices] += adjustment
        
        # Recalculate stats
        cont_elevations = self.planet.elevation[continental_vertices]
        min_cont_elev = np.min(cont_elevations)
        max_cont_elev = np.max(cont_elevations)
        mean_cont_elev = np.mean(cont_elevations)
        
        print(f"  Continental elevation after adjustment: {min_cont_elev:.2f} to {max_cont_elev:.2f}, mean: {mean_cont_elev:.2f}")
        
    # Ensure some parts of each continental plate are above water
    for plate in self.plates:
        if not plate['is_oceanic']:
            # Get continental vertices in this plate
            plate_cont_vertices = np.intersect1d(plate['vertices'], continental_vertices)
            
            if len(plate_cont_vertices) > 0:
                plate_elevations = self.planet.elevation[plate_cont_vertices]
                max_plate_elev = np.max(plate_elevations)
                
                # If the highest point is underwater, bring it above
                if max_plate_elev < 0:
                    # Find highest points (top 10%)
                    high_threshold = np.percentile(plate_elevations, 90)
                    high_points = plate_cont_vertices[plate_elevations >= high_threshold]
                    
                    # Lift these points above sea level
                    adjustment = 0.5 - max_plate_elev  # Raise to at least 0.5 km
                    self.planet.elevation[high_points] += adjustment
                    
                    print(f"  Raised continental plate {plate['id']} above sea level")
    
    # Apply terrain variations for realism
    self._apply_terrain_variations()

def _apply_terrain_variations(self):
    """Apply realistic terrain variations to the elevation model"""
    # Create high-frequency noise
    terrain_noise = self._generate_coherent_noise(octaves=6, persistence=0.5, scale=0.5)
    
    # Apply noise differently to different regions
    for i in range(len(self.planet.elevation)):
        if self.planet.elevation[i] > 0:  # Land
            # More varied terrain on land
            # Variation increases with elevation (mountains more rugged)
            base_variation = 0.3 * terrain_noise[i]
            elevation_factor = min(1.0, self.planet.elevation[i] / 3.0)
            variation = base_variation * (1.0 + elevation_factor)
            
            self.planet.elevation[i] += variation
        else:  # Ocean
            # Smoother but still varied ocean floor
            variation = 0.1 * terrain_noise[i]
            self.planet.elevation[i] += variation
    
    # Add mountain ranges along convergent boundaries
    if self.plate_boundaries is not None:
        boundary_vertices = np.where(self.plate_boundaries)[0]
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        for i in boundary_vertices:
            # Skip underwater boundaries
            if self.planet.elevation[i] < 0:
                continue
                
            # Check if this is a convergent boundary
            plate_id = self.planet.plate_ids[i]
            if plate_id == -1 or plate_id >= len(self.plates):
                continue
                
            # Find neighbor from different plate
            neighbor_plate = None
            for n in neighbors[i]:
                if self.planet.plate_ids[n] != plate_id and self.planet.plate_ids[n] != -1:
                    neighbor_plate = self.planet.plate_ids[n]
                    break
            
            if neighbor_plate is not None and neighbor_plate < len(self.plates):
                # Check if both are continental
                if (not self.plates[plate_id]['is_oceanic'] and 
                    not self.plates[neighbor_plate]['is_oceanic']):
                    # Continental collision = higher mountains
                    self.planet.elevation[i] += np.random.random() * 2.0

def _apply_terrain_variations(self):
    """Apply realistic terrain variations to the elevation model"""
    # Create high-frequency noise
    terrain_noise = self._generate_coherent_noise(octaves=6, persistence=0.5, scale=0.5)
    
    # Apply noise differently to different regions
    for i in range(len(self.planet.elevation)):
        if self.planet.elevation[i] > 0:  # Land
            # More varied terrain on land
            # Variation increases with elevation (mountains more rugged)
            base_variation = 0.3 * terrain_noise[i]
            elevation_factor = min(1.0, self.planet.elevation[i] / 3.0)
            variation = base_variation * (1.0 + elevation_factor)
            
            self.planet.elevation[i] += variation
        else:  # Ocean
            # Smoother but still varied ocean floor
            variation = 0.1 * terrain_noise[i]
            self.planet.elevation[i] += variation
    
    # Add mountain ranges along convergent boundaries
    if self.plate_boundaries is not None:
        boundary_vertices = np.where(self.plate_boundaries)[0]
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        for i in boundary_vertices:
            # Skip underwater boundaries
            if self.planet.elevation[i] < 0:
                continue
                
            # Check if this is a convergent boundary
            plate_id = self.planet.plate_ids[i]
            if plate_id == -1 or plate_id >= len(self.plates):
                continue
                
            # Find neighbor from different plate
            neighbor_plate = None
            for n in neighbors[i]:
                if self.planet.plate_ids[n] != plate_id and self.planet.plate_ids[n] != -1:
                    neighbor_plate = self.planet.plate_ids[n]
                    break
            
            if neighbor_plate is not None and neighbor_plate < len(self.plates):
                # Check if both are continental
                if (not self.plates[plate_id]['is_oceanic'] and 
                    not self.plates[neighbor_plate]['is_oceanic']):
                    # Continental collision = higher mountains
                    self.planet.elevation[i] += np.random.random() * 2.0