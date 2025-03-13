import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..utils.sphere_grid import SphericalGrid
import matplotlib.colors as colors
from matplotlib import cm
import cartopy.crs as ccrs
import os

class Planet:
    """
    Represents a planet with its physical properties and surface data.
    Enhanced with improved visualization capabilities and data tracking.
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
        
        # Add this line to store rivers
        self.rivers = []  # Each river is a list of vertex indices
        
        # Data tracking
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
        
        # Advanced properties 
        self.crust_age = None        # Age of crust at each vertex
        self.crust_thickness = None  # Thickness of crust at each vertex
        self.crust_strain = None     # Accumulated strain at each vertex
        self.magnetic_field = None   # Magnetic field strength (0-1)
        self.ocean_depth = None      # Depth of ocean in km (negative elevation doesn't give depth)
        
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
        # We would use datetime here, but keeping it simple
        self.history['metadata']['last_modified'] = self.age
    
    def visualize(self, save_path=None, mode='elevation', show=False, title=None,
                 custom_colormap=None, figure_size=(12, 10), dpi=100, 
                 view_angle=(30, 45), show_features=False, feature_size=20):
        """
        Visualize the planet in 3D with matplotlib.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - mode: Visualization mode ('elevation', 'temperature', 'precipitation', 'biome', 'plates', 'age')
        - show: Whether to display the plot (should be False for non-interactive environments)
        - title: Custom title for the plot
        - custom_colormap: Optional custom colormap
        - figure_size: Size of the figure in inches
        - dpi: DPI for the saved image
        - view_angle: Tuple of (elevation, azimuth) in degrees for 3D view
        - show_features: Whether to highlight special features like hotspots, ridges, etc.
        - feature_size: Size of feature markers
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create arrays for plotting
        xs, ys, zs = [], [], []
        point_colors = []
        
        # Map data to colors and position based on mode
        for i, vertex in enumerate(self.grid.vertices):
            # Apply elevation to position
            elevation = self.elevation[i]
            norm = np.linalg.norm(vertex)
            scaled_vertex = vertex * (1 + elevation/self.radius)
            
            xs.append(scaled_vertex[0])
            ys.append(scaled_vertex[1])
            zs.append(scaled_vertex[2])
            
            # Color based on selected mode
            if mode == 'elevation':
                point_colors.append(self._get_elevation_color(elevation))
            elif mode == 'temperature' and self.temperature is not None:
                point_colors.append(self._get_temperature_color(self.temperature[i]))
            elif mode == 'precipitation' and self.precipitation is not None:
                point_colors.append(self._get_precipitation_color(self.precipitation[i]))
            elif mode == 'biome' and self.biome_ids is not None:
                from ..core.biome import BiomeClassifier
                biome_classifier = BiomeClassifier(self)
                point_colors.append(biome_classifier.get_biome_color(self.biome_ids[i]))
            elif mode == 'plates' and hasattr(self, 'plate_ids'):
                # Unique color for each plate
                plate_id = self.plate_ids[i]
                if plate_id >= 0:
                    # Use a hash function to get a consistent color for each plate
                    hue = (plate_id * 0.618033988749895) % 1.0  # Golden ratio to spread colors
                    point_colors.append(plt.cm.hsv(hue))
                else:
                    point_colors.append([0.7, 0.7, 0.7])  # Gray for unassigned
            elif mode == 'age' and hasattr(self, 'crust_age') and self.crust_age is not None:
                point_colors.append(self._get_age_color(self.crust_age[i]))
            else:
                # Default coloring
                point_colors.append([0.7, 0.7, 0.7])  # Gray
        
        # Plot the vertices
        ax.scatter(xs, ys, zs, c=point_colors, s=10, alpha=0.8)
        
        # Plot rivers if available
        if hasattr(self, 'rivers') and self.rivers and mode in ['elevation', 'biome']:
            for river in self.rivers:
                if len(river) > 1:
                    river_xs = []
                    river_ys = []
                    river_zs = []
                    
                    for idx in river:
                        vertex = self.grid.vertices[idx]
                        elevation = self.elevation[idx]
                        scaled_vertex = vertex * (1 + elevation/self.radius)
                        
                        river_xs.append(scaled_vertex[0])
                        river_ys.append(scaled_vertex[1])
                        river_zs.append(scaled_vertex[2])
                    
                    # Plot the river as a blue line
                    ax.plot(river_xs, river_ys, river_zs, color='blue', linewidth=1.5, alpha=0.7)
        
        # Plot special features if requested
        if show_features:
            self._add_special_features(ax, feature_size, mode)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        ax.set_axis_off()
        
        # Set the view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Add title based on mode
        if title:
            plt.title(title, fontsize=14)
        else:
            title_map = {
                'elevation': f'Terrain Elevation (Age: {self.age:.1f} million years)',
                'temperature': 'Surface Temperature (°C)',
                'precipitation': 'Annual Precipitation (mm)',
                'biome': 'Biome Distribution',
                'plates': 'Tectonic Plates',
                'age': 'Crust Age (million years)'
            }
            plt.title(title_map.get(mode, f'Planet Visualization - {mode.capitalize()}'), fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
            # Update visualization history
            self.last_visualization = {
                'type': '3d',
                'mode': mode,
                'path': save_path,
                'timestamp': self.age
            }
            
            if not show:
                plt.close(fig)
            print(f"3D visualization saved to {save_path}")
            
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_2d(self, save_path=None, mode='elevation', projection='equirectangular',
                    show=False, title=None, custom_colormap=None, figure_size=(12, 6), dpi=100,
                    show_features=False, feature_size=20, grid_lines=True, coastlines=True):
        """
        Visualize the planet as a 2D map projection.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - mode: Visualization mode ('elevation', 'temperature', 'precipitation', 'biome', 'plates', 'age')
        - projection: Map projection ('equirectangular', 'mollweide', 'orthographic')
        - show: Whether to display the plot
        - title: Custom title for the plot
        - custom_colormap: Optional custom colormap
        - figure_size: Size of the figure in inches
        - dpi: DPI for the saved image
        - show_features: Whether to highlight special features
        - feature_size: Size of feature markers
        - grid_lines: Whether to show latitude/longitude grid lines
        - coastlines: Whether to show coastlines
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        # Set up the projection
        if projection == 'equirectangular':
            proj = ccrs.PlateCarree()
        elif projection == 'mollweide':
            proj = ccrs.Mollweide()
        elif projection == 'orthographic':
            proj = ccrs.Orthographic(central_longitude=0, central_latitude=30)
        else:
            proj = ccrs.PlateCarree()  # Default
        
        # Create figure and axis
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        
        # Enable coastlines if requested
        if coastlines:
            self._add_coastlines(ax)
        
        # Add grid lines if requested
        if grid_lines:
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        
        # Convert 3D coordinates to lat/lon for mapping
        lats = []
        lons = []
        values = []
        
        for i, vertex in enumerate(self.grid.vertices):
            # Convert to lat/lon
            x, y, z = vertex
            lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
            lon = np.arctan2(y, x) * 180 / np.pi
            
            lats.append(lat)
            lons.append(lon)
            
            # Get value based on selected mode
            if mode == 'elevation':
                values.append(self.elevation[i])
            elif mode == 'temperature' and self.temperature is not None:
                values.append(self.temperature[i])
            elif mode == 'precipitation' and self.precipitation is not None:
                values.append(self.precipitation[i])
            elif mode == 'biome' and self.biome_ids is not None:
                values.append(self.biome_ids[i])
            elif mode == 'plates' and hasattr(self, 'plate_ids'):
                values.append(self.plate_ids[i])
            elif mode == 'age' and hasattr(self, 'crust_age') and self.crust_age is not None:
                values.append(self.crust_age[i])
            else:
                values.append(0)  # Default
        
        # Set up colormap based on mode
        if custom_colormap:
            cmap = custom_colormap
        elif mode == 'elevation':
            # Create a custom colormap for elevation with land/sea distinction
            cmap = self._create_elevation_colormap()
            # Normalize based on elevation range
            norm = colors.Normalize(vmin=min(self.elevation), vmax=max(self.elevation))
        elif mode == 'temperature':
            cmap = plt.cm.coolwarm
            # Normalize temperature
            min_temp = min(self.temperature) if self.temperature is not None else -20
            max_temp = max(self.temperature) if self.temperature is not None else 40
            norm = colors.Normalize(vmin=min_temp, vmax=max_temp)
        elif mode == 'precipitation':
            cmap = plt.cm.Blues
            # Normalize precipitation
            min_precip = 0
            max_precip = max(self.precipitation) if self.precipitation is not None else 3000
            norm = colors.Normalize(vmin=min_precip, vmax=max_precip)
        elif mode == 'biome':
            # For biomes, use a qualitative colormap
            cmap = plt.cm.tab20
            norm = colors.Normalize(vmin=0, vmax=19)  # Assuming up to 20 biome types
        elif mode == 'plates':
            # For plates, use a cyclic colormap
            cmap = plt.cm.hsv
            norm = colors.Normalize(vmin=0, vmax=max(self.plate_ids) if hasattr(self, 'plate_ids') else 20)
        elif mode == 'age':
            # For crust age, use a sequential colormap
            cmap = plt.cm.plasma_r
            min_age = 0
            max_age = max(self.crust_age) if self.crust_age is not None else self.age
            norm = colors.Normalize(vmin=min_age, vmax=max_age)
        else:
            cmap = plt.cm.viridis
            norm = colors.Normalize(vmin=min(values), vmax=max(values))
        
        # Scatter plot with appropriate coloring
        scatter = ax.scatter(lons, lats, c=values, cmap=cmap, norm=norm, 
                           s=5, transform=ccrs.PlateCarree(), alpha=0.8)
        
        # Add colorbar
        if mode != 'biome' and mode != 'plates':  # Don't add colorbar for categorical data
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
            
            # Set colorbar label based on mode
            if mode == 'elevation':
                cbar.set_label('Elevation (km)')
            elif mode == 'temperature':
                cbar.set_label('Temperature (°C)')
            elif mode == 'precipitation':
                cbar.set_label('Precipitation (mm/year)')
            elif mode == 'age':
                cbar.set_label('Crust Age (million years)')
        
        # Add rivers if available and appropriate
        if hasattr(self, 'rivers') and self.rivers and mode in ['elevation', 'biome']:
            for river in self.rivers:
                if len(river) > 1:
                    river_lons = []
                    river_lats = []
                    
                    for idx in river:
                        vertex = self.grid.vertices[idx]
                        x, y, z = vertex
                        lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
                        lon = np.arctan2(y, x) * 180 / np.pi
                        
                        river_lons.append(lon)
                        river_lats.append(lat)
                    
                    # Plot the river
                    ax.plot(river_lons, river_lats, color='blue', linewidth=1, 
                          transform=ccrs.PlateCarree(), alpha=0.7)
        
        # Plot special features if requested
        if show_features:
            self._add_special_features_2d(ax, feature_size, mode)
        
        # Add title
        if title:
            plt.title(title, fontsize=14)
        else:
            title_map = {
                'elevation': f'Terrain Elevation (Age: {self.age:.1f} million years)',
                'temperature': 'Surface Temperature (°C)',
                'precipitation': 'Annual Precipitation (mm)',
                'biome': 'Biome Distribution',
                'plates': 'Tectonic Plates',
                'age': 'Crust Age (million years)'
            }
            plt.title(title_map.get(mode, f'Planet Visualization - {mode.capitalize()}'), fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
            # Update visualization history
            self.last_visualization = {
                'type': '2d',
                'mode': mode,
                'path': save_path,
                'timestamp': self.age
            }
            
            if not show:
                plt.close(fig)
            print(f"2D visualization saved to {save_path}")
            
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def _add_coastlines(self, ax):
        """Add coastlines to a 2D map"""
        # Convert 3D coordinates to lat/lon for coastlines
        coast_lats = []
        coast_lons = []
        
        # Find coastal vertices (where elevation is close to 0)
        for i, vertex in enumerate(self.grid.vertices):
            # Skip non-coastal points
            if not (-0.1 < self.elevation[i] < 0.1):
                continue
                
            # Find if this vertex is at a land-sea boundary
            neighbors = self.grid.get_vertex_neighbors()[i]
            is_coast = False
            
            for n in neighbors:
                if (self.elevation[i] >= 0 and self.elevation[n] < 0) or \
                   (self.elevation[i] < 0 and self.elevation[n] >= 0):
                    is_coast = True
                    break
            
            if is_coast:
                # Convert to lat/lon
                x, y, z = vertex
                lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
                lon = np.arctan2(y, x) * 180 / np.pi
                
                coast_lats.append(lat)
                coast_lons.append(lon)
        
        # Plot coastlines
        ax.scatter(coast_lons, coast_lats, color='black', s=1, 
                 transform=ccrs.PlateCarree(), alpha=0.5)
    
    def _add_special_features(self, ax, feature_size, mode):
        """Add special geological features to 3D visualization"""
        # This would typically come from tectonics simulation
        # For now, we'll just add some placeholder features if they exist
        
        # Check if we have a tectonics simulation attached
        if not hasattr(self, '_tectonics') or self._tectonics is None:
            return
        
        # Add hotspots if available
        if hasattr(self._tectonics, 'hotspots'):
            for hotspot in self._tectonics.hotspots:
                pos = hotspot['position']
                # Scale to planet surface
                scaled_pos = pos * self.radius
                ax.plot([0, scaled_pos[0]], [0, scaled_pos[1]], [0, scaled_pos[2]], 
                      'r--', alpha=0.5, linewidth=1)
                ax.scatter([scaled_pos[0]], [scaled_pos[1]], [scaled_pos[2]], 
                         color='orange', s=feature_size, alpha=0.8, marker='^')
        
        # Add volcanic arcs if available
        if hasattr(self._tectonics, 'volcanic_arcs'):
            for arc in self._tectonics.volcanic_arcs:
                for vertex_idx in arc:
                    vertex = self.grid.vertices[vertex_idx]
                    elevation = self.elevation[vertex_idx]
                    scaled_vertex = vertex * (1 + elevation/self.radius)
                    
                    ax.scatter([scaled_vertex[0]], [scaled_vertex[1]], [scaled_vertex[2]], 
                             color='red', s=feature_size, alpha=0.8, marker='*')
        
        # Add other features as needed
    
    def _add_special_features_2d(self, ax, feature_size, mode):
        """Add special geological features to 2D visualization"""
        # Similar to 3D version but for 2D projections
        
        # Check if we have a tectonics simulation attached
        if not hasattr(self, '_tectonics') or self._tectonics is None:
            return
        
        # Add hotspots if available
        if hasattr(self._tectonics, 'hotspots'):
            for hotspot in self._tectonics.hotspots:
                pos = hotspot['position']
                # Convert to lat/lon
                x, y, z = pos
                lat = np.arcsin(z / np.linalg.norm(pos)) * 180 / np.pi
                lon = np.arctan2(y, x) * 180 / np.pi
                
                ax.scatter([lon], [lat], transform=ccrs.PlateCarree(),
                          color='orange', s=feature_size, alpha=0.8, marker='^')
        
        # Add other features as needed
    
    def _get_elevation_color(self, elevation):
        """Get color for elevation visualization"""
        if elevation < 0:
            # Water - blue shades
            depth_ratio = max(-1, elevation / min(-0.001, np.min(self.elevation)))
            return [0, 0, 0.5 + 0.5 * depth_ratio]
        else:
            # Land - green to brown to white
            rel_height = elevation / max(0.001, np.max(self.elevation))
            if rel_height < 0.2:
                return [0.2, 0.6, 0.2]  # Low land, green
            elif rel_height < 0.5:
                return [0.6, 0.4, 0.2]  # Hills, brown
            else:
                # Gradual transition to white for highest peaks
                mountain_ratio = min(1, (rel_height - 0.5) * 2)
                return [0.6 + 0.4*mountain_ratio, 
                        0.4 + 0.6*mountain_ratio, 
                        0.2 + 0.8*mountain_ratio]
    
    def _get_temperature_color(self, temp):
        """Get color for temperature visualization"""
        if temp < -20:
            return [0, 0, 1]  # Very cold - deep blue
        elif temp < 0:
            # Cold - blue to cyan
            t = (temp + 20) / 20
            return [0, t, 1]
        elif temp < 20:
            # Moderate - cyan to yellow
            t = temp / 20
            return [t, 1, 1-t]
        else:
            # Hot - yellow to red
            t = min(1, (temp - 20) / 20)
            return [1, 1-t, 0]
    
    def _get_precipitation_color(self, precip):
        """Get color for precipitation visualization"""
        if precip < 250:
            return [0.95, 0.95, 0.2]  # Very dry - yellow
        elif precip < 500:
            t = (precip - 250) / 250
            return [0.95-t*0.6, 0.95-t*0.3, 0.2+t*0.3]  # Dry - yellow to tan
        elif precip < 1000:
            t = (precip - 500) / 500
            return [0.35-t*0.25, 0.65+t*0.15, 0.5+t*0.3]  # Moderate - tan to green
        else:
            t = min(1, (precip - 1000) / 1000)
            return [0.1, 0.8-t*0.4, 0.8+t*0.2]  # Wet - green to blue
    
    def _get_age_color(self, age):
        """Get color for crust age visualization"""
        # Normalize age to 0-1 range
        max_age = max(self.age, max(self.crust_age) if self.crust_age is not None else 0)
        if max_age == 0:
            return [0, 0, 0]  # Black for undefined age
            
        normalized_age = min(1, age / max_age)
        
        # Use plasma colormap
        return plt.cm.plasma_r(normalized_age)
    
    def _create_elevation_colormap(self):
        """Create a custom colormap for elevation with land-sea distinction"""
        # Create custom colormap that has a clear distinction at sea level
        # Define colors for specific elevation ranges
        cmap_data = [
            # Deep ocean (dark blue)
            (0.0, [0.0, 0.0, 0.5]),
            # Shallow ocean (light blue)
            (0.49, [0.0, 0.4, 0.8]),
            # Coastline (tan)
            (0.5, [0.7, 0.7, 0.5]),
            # Lowlands (light green)
            (0.55, [0.2, 0.6, 0.2]),
            # Hills (brown)
            (0.7, [0.6, 0.4, 0.2]),
            # Mountains (gray)
            (0.85, [0.5, 0.5, 0.5]),
            # Peaks (white)
            (1.0, [1.0, 1.0, 1.0])
        ]
        
        # Create a custom ListedColormap
        elevation_cmap = colors.LinearSegmentedColormap.from_list(
            "elevation", 
            cmap_data, 
            N=256
        )
        
        return elevation_cmap
    
    def export_heightmap(self, output_path, width=2048, height=1024, 
                        region=None, format='png', bit_depth=16):
        """
        Export a heightmap of the planet. This is a convenience wrapper around ExportManager.
        
        Parameters:
        - output_path: Path to save the heightmap
        - width, height: Dimensions of the output image
        - region: Optional tuple (lat_min, lon_min, lat_max, lon_max) for region extract
        - format: Output format ('png', 'raw', 'tiff')
        - bit_depth: Bit depth for the heightmap (8 or 16)
        
        Returns:
        - Path to the exported heightmap
        """
        from ..utils.export import ExportManager
        export_manager = ExportManager(self)
        
        return export_manager.export_heightmap(
            output_path, width, height, region, format, bit_depth
        )
    
    def save_to_file(self, file_path):
        """
        Save the planet to a file.
        
        Parameters:
        - file_path: Path to save the planet data
        
        Returns:
        - Path to the saved file
        """
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Planet saved to {file_path}")
        return file_path
    
    @classmethod
    def load_from_file(cls, file_path):
        """
        Load a planet from a file.
        
        Parameters:
        - file_path: Path to the saved planet data
        
        Returns:
        - Planet object
        """
        import pickle
        
        # Load using pickle
        with open(file_path, 'rb') as f:
            planet = pickle.load(f)
        
        print(f"Planet loaded from {file_path}")
        return planet
    
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