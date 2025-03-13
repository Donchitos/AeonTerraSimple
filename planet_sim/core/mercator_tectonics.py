import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter
import datetime
import os
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
        
        # Allow the planet to access this simulation
        self.planet._tectonics = self
        
    # ... [keep all the existing methods] ...
    
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
                    px, py = self.lat_lon_to_mercator(lat, lon, width, height)
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
                        px1, py1 = self.lat_lon_to_mercator(lat1, lon1, width, height)
                        px2, py2 = self.lat_lon_to_mercator(lat2, lon2, width, height)
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
        for plate in self.plates:
            # Convert center to lat/lon
            x, y, z = plate['center']
            lat = np.arcsin(z / np.linalg.norm(plate['center'])) * 180 / np.pi
            lon = np.arctan2(y, x) * 180 / np.pi
            
            # Project to 2D based on projection type
            if projection == 'mercator':
                px, py = self.lat_lon_to_mercator(lat, lon, width, height)
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

    # Keep all other original methods from tectonics.py