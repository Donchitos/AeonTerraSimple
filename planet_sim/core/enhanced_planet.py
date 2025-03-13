import numpy as np
from ..utils.sphere_grid import SphericalGrid
import matplotlib.pyplot as plt
import math

class Planet:
    """
    Represents a planet with its physical properties and surface data.
    """
    
    def __init__(self, radius=6371.0, resolution=5):
        """
        Initialize a planet.
        
        Parameters:
        - radius: Planet radius in kilometers
        - resolution: Detail level of the terrain grid
        """
        self.radius = radius
        self.grid = SphericalGrid(radius, resolution)
        
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
    
    def initialize_terrain(self):
        """Initialize a basic terrain with some noise"""
        # Simple random noise for demonstration
        noise_scale = self.radius * 0.02  # 2% of radius as elevation range
        self.elevation = np.random.uniform(-noise_scale, noise_scale, len(self.grid.vertices))
        
        # Ensure some landmass by setting 30% to be above sea level
        threshold = np.percentile(self.elevation, 70)
        self.elevation = np.where(self.elevation > threshold, 
                                  self.elevation + noise_scale*0.5, 
                                  self.elevation)
        
        return self
    
    def visualize(self, save_path=None, mode='elevation'):
        """
        Visualize the planet with matplotlib.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - mode: Visualization mode ('elevation', 'temperature', 'precipitation', 'biome')
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create arrays for plotting
        xs, ys, zs = [], [], []
        colors = []
        
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
                if elevation < 0:
                    # Water - blue
                    colors.append([0, 0, 0.5 + 0.5 * elevation / min(self.elevation)])
                else:
                    # Land - green to brown to white
                    rel_height = elevation / max(0.001, max(self.elevation))
                    if rel_height < 0.2:
                        colors.append([0.2, 0.6, 0.2])  # Low land, green
                    elif rel_height < 0.5:
                        colors.append([0.6, 0.4, 0.2])  # Hills, brown
                    else:
                        colors.append([0.8, 0.8, 0.8])  # Mountains, light gray
            elif mode == 'temperature' and self.temperature is not None:
                temp = self.temperature[i]
                # Temperature color scale from blue (cold) to red (hot)
                if temp < -20:
                    colors.append([0, 0, 1])  # Very cold - deep blue
                elif temp < 0:
                    # Cold - blue to cyan
                    t = (temp + 20) / 20
                    colors.append([0, t, 1])
                elif temp < 20:
                    # Moderate - cyan to yellow
                    t = temp / 20
                    colors.append([t, 1, 1-t])
                else:
                    # Hot - yellow to red
                    t = min(1, (temp - 20) / 20)
                    colors.append([1, 1-t, 0])
            elif mode == 'precipitation' and self.precipitation is not None:
                precip = self.precipitation[i]
                # Precipitation scale from yellow (dry) to dark blue (wet)
                if precip < 250:
                    colors.append([0.95, 0.95, 0.2])  # Very dry - yellow
                elif precip < 500:
                    t = (precip - 250) / 250
                    colors.append([0.95-t*0.6, 0.95-t*0.3, 0.2+t*0.3])  # Dry - yellow to tan
                elif precip < 1000:
                    t = (precip - 500) / 500
                    colors.append([0.35-t*0.25, 0.65+t*0.15, 0.5+t*0.3])  # Moderate - tan to green
                else:
                    t = min(1, (precip - 1000) / 1000)
                    colors.append([0.1, 0.8-t*0.4, 0.8+t*0.2])  # Wet - green to blue
            elif mode == 'biome' and self.biome_ids is not None:
                # Get biome color
                from .biome import BiomeClassifier
                biome_classifier = BiomeClassifier(self)
                colors.append(biome_classifier.get_biome_color(self.biome_ids[i]))
            else:
                # Default coloring
                colors.append([0.7, 0.7, 0.7])  # Gray
        
        # Plot the vertices
        ax.scatter(xs, ys, zs, c=colors, s=10)
        
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
                    ax.plot(river_xs, river_ys, river_zs, color='blue', linewidth=1)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        ax.set_axis_off()
        
        # Add title based on mode
        title = f'Planet Visualization - {mode.capitalize()}'
        plt.title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def visualize_land_water(self, save_path=None, show=False, projection='mercator'):
        """
        Visualize land/water distribution with green/blue coloring.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - show: Whether to display the plot
        - projection: Map projection to use
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure
        fig = plt.figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Setup the projection
        width, height = 1200, 600  # Pixels for projection calculation
        
        # Draw each vertex with land or water color
        for i, vertex in enumerate(self.grid.vertices):
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
            
            # Skip if outside bounds
            if px < 0 or px >= width or py < 0 or py >= height:
                continue
            
            # Plot point with land/water color
            elev = self.elevation[i]
            color = 'green' if elev >= 0 else 'blue'
            ax.scatter(px, py, c=color, s=5, alpha=0.8)
        
        # Add rivers if available
        if hasattr(self, 'rivers') and self.rivers:
            for river in self.rivers:
                if len(river) > 1:
                    river_x = []
                    river_y = []
                    
                    for idx in river:
                        vertex = self.grid.vertices[idx]
                        x, y, z = vertex
                        lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
                        lon = np.arctan2(y, x) * 180 / np.pi
                        
                        if projection == 'mercator':
                            px, py = self.lat_lon_to_mercator(lat, lon, width, height)
                        else:
                            px = (lon + 180) * width / 360
                            py = (90 - lat) * height / 180
                        
                        if 0 <= px < width and 0 <= py < height:
                            river_x.append(px)
                            river_y.append(py)
                    
                    if river_x and river_y:
                        ax.plot(river_x, river_y, 'blue', linewidth=1.5, alpha=0.7)
        
        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis for standard map orientation
        
        # Add title and grid
        plt.title(f'Land and Water Distribution (Age: {self.age:.1f} million years)')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='green', label='Land'),
            Patch(facecolor='blue', edgecolor='blue', label='Water')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            if not show:
                plt.close(fig)
            print(f"Land/water visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_heightmap(self, save_path=None, show=False, projection='mercator'):
        """
        Visualize planet as a grayscale heightmap.
        
        Parameters:
        - save_path: If provided, save the visualization to this file path
        - show: Whether to display the plot
        - projection: Map projection to use
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import numpy as np
        
        # Create figure
        fig = plt.figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Setup the projection
        width, height = 1200, 600  # Pixels for projection calculation
        
        # Find elevation extremes for normalization
        min_elev = min(self.elevation)
        max_elev = max(self.elevation)
        
        # Create a grayscale norm
        norm = colors.Normalize(vmin=min_elev, vmax=max_elev)
        
        # Prepare pixel data
        pixel_data = np.zeros((height, width))
        pixel_count = np.zeros((height, width))
        
        # Draw each vertex with height color
        for i, vertex in enumerate(self.grid.vertices):
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
            
            # Ensure coordinates are within bounds
            px_int = int(px)
            py_int = int(py)
            
            if 0 <= px_int < width and 0 <= py_int < height:
                # Add elevation to pixel grid
                pixel_data[py_int, px_int] += self.elevation[i]
                pixel_count[py_int, px_int] += 1
        
        # Calculate average elevation where multiple vertices map to the same pixel
        mask = pixel_count > 0
        pixel_data[mask] = pixel_data[mask] / pixel_count[mask]
        
        # Display the heightmap
        img = ax.imshow(pixel_data, cmap='gray', norm=norm, extent=[0, width, height, 0])
        
        # Add title and color bar
        plt.title(f'Elevation Heightmap (Age: {self.age:.1f} million years)')
        plt.colorbar(img, ax=ax, label='Elevation (km)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            if not show:
                plt.close(fig)
            print(f"Heightmap visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def lat_lon_to_mercator(self, lat, lon, width, height):
        """
        Convert latitude/longitude to Mercator projection x,y coordinates.
        
        Parameters:
        - lat: Latitude in degrees
        - lon: Longitude in degrees
        - width: Width of the output image in pixels
        - height: Height of the output image in pixels
        
        Returns:
        - x, y coordinates in the Mercator projection
        """
        # Constrain latitude to prevent infinite scaling near poles
        lat = max(min(lat, 85), -85)
        
        # Mercator projection formulas
        x = (lon + 180) * (width / 360)
        
        # Convert latitude to radians
        lat_rad = lat * math.pi / 180
        
        # Mercator y-coordinate formula
        merc_n = math.log(math.tan((math.pi / 4) + (lat_rad / 2)))
        y = (height / 2) - (width * merc_n / (2 * math.pi))
        
        return x, y

    # All other original Planet methods stay the same