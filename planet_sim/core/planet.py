import numpy as np
from ..utils.sphere_grid import SphericalGrid

class Planet:
    """
    Represents a planet with its physical properties and surface data.
    """
    
    def __init__(self, radius=6371.0, resolution=5, name="AeonTerra"):
        """
        Initialize a planet.
        
        Parameters:
        - radius: Planet radius in kilometers
        - resolution: Detail level of the terrain grid
        - name: Planet name for identification
        """
        self.radius = radius
        self.grid = SphericalGrid(radius, resolution)
        self.name = name
        
        # Initialize data arrays for different planetary properties
        num_vertices = len(self.grid.vertices)
        self.elevation = np.zeros(num_vertices)
        self.plate_ids = np.full(num_vertices, -1)
        self.temperature = None
        self.precipitation = None
        self.biome_ids = None
        
        # Planet properties
        self.age = 0  # Simulation age in million years
        self.rotation_period = 24.0  # hours
        self.axial_tilt = 23.5  # degrees
        self.gravity = 9.8  # m/s²
        
        # Store rivers (each river is a list of vertex indices)
        self.rivers = []
        
        # History tracking
        self.history = {
            'elevation': [],
            'metadata': {
                'creation_time': None,
                'last_modified': None,
                'simulation_steps': 0
            },
            'stats': {
                'min_elevation': [],
                'max_elevation': [],
                'land_percentage': [],
                'ocean_percentage': []
            }
        }
        
        # For visualization
        self.last_visualization = {
            'type': None,
            'mode': None,
            'path': None,
            'timestamp': None
        }
        
        print(f"Planet {self.name} created with {num_vertices} vertices at resolution {resolution}")
    
    def initialize_terrain(self, noise_scale=0.02, continent_percentage=30):
        """
        Initialize a basic terrain with some noise.
        
        Parameters:
        - noise_scale: Scale of noise as a fraction of planet radius
        - continent_percentage: Percentage of surface to elevate above sea level
        
        Returns:
        - self (for method chaining)
        """
        # Simple random noise for demonstration
        noise_scale = self.radius * noise_scale  # Scale noise to a percentage of radius
        self.elevation = np.random.uniform(-noise_scale, noise_scale, len(self.grid.vertices))
        
        # Ensure some landmass by setting desired percentage to be above sea level
        threshold = np.percentile(self.elevation, 100 - continent_percentage)
        self.elevation = np.where(self.elevation > threshold, 
                                 self.elevation + noise_scale*0.5, 
                                 self.elevation)
        
        # Update history and stats
        self._update_stats()
        
        return self
    
    def _update_stats(self):
        """Update the planet's statistics for history tracking"""
        # Calculate basic statistics
        min_elev = float(np.min(self.elevation))
        max_elev = float(np.max(self.elevation))
        land_percentage = float(np.sum(self.elevation >= 0) / len(self.elevation) * 100)
        ocean_percentage = 100.0 - land_percentage
        
        # Update history
        self.history['stats']['min_elevation'].append(min_elev)
        self.history['stats']['max_elevation'].append(max_elev)
        self.history['stats']['land_percentage'].append(land_percentage)
        self.history['stats']['ocean_percentage'].append(ocean_percentage)
        
        # Update metadata
        self.history['metadata']['simulation_steps'] += 1
        self.history['metadata']['last_modified'] = self.age
    
    def get_lat_lon(self, vertex_idx):
        """Convert vertex coordinates to latitude/longitude (in degrees)"""
        return self.grid.get_lat_lon(vertex_idx)
    
    def summarize(self):
        """
        Generate a text summary of the planet's current state.
        
        Returns:
        - String with planet summary
        """
        landmass = np.sum(self.elevation >= 0) / len(self.elevation) * 100
        ocean = 100 - landmass
        
        summary = [
            f"Planet: {self.name}",
            f"Age: {self.age:.1f} million years",
            f"Radius: {self.radius:.1f} km",
            f"Resolution: {len(self.grid.vertices)} vertices",
            f"Elevation Range: {np.min(self.elevation):.1f} to {np.max(self.elevation):.1f} km",
            f"Land Coverage: {landmass:.1f}%",
            f"Ocean Coverage: {ocean:.1f}%",
            f"Simulation Steps: {self.history['metadata']['simulation_steps']}"
        ]
        
        # Add info about other available data
        if self.temperature is not None:
            summary.append(f"Temperature Range: {np.min(self.temperature):.1f} to {np.max(self.temperature):.1f} °C")
        
        if self.precipitation is not None:
            summary.append(f"Precipitation Range: {np.min(self.precipitation):.1f} to {np.max(self.precipitation):.1f} mm/year")
        
        if self.biome_ids is not None:
            unique_biomes = len(np.unique(self.biome_ids))
            summary.append(f"Biomes: {unique_biomes} different biome types")
        
        if self.rivers:
            summary.append(f"Rivers: {len(self.rivers)} river systems")
        
        return "\n".join(summary)