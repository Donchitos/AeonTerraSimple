import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math

class PlanetVisualizer:
    """
    Handles visualization of planets and their various properties.
    Consolidates visualization functionality from different classes.
    """
    
    def __init__(self, planet):
        """
        Initialize the visualizer.
        
        Parameters:
        - planet: A Planet object
        """
        self.planet = planet
    
    def visualize_3d(self, save_path=None, mode='elevation', show=False, title=None,
                    custom_colormap=None, figure_size=(12, 10), dpi=100, view_angle=(30, 45)):
        """
        Visualize the planet in 3D with matplotlib.
        
        Parameters:
        - save_path: If provided, save visualization to this file path
        - mode: Visualization mode ('elevation', 'temperature', 'precipitation', 'biome', 'plates')
        - show: Whether to display the plot
        - title: Custom title for the plot
        - custom_colormap: Optional custom colormap
        - figure_size: Size of the figure in inches
        - dpi: DPI for the saved image
        - view_angle: Tuple of (elevation, azimuth) in degrees for 3D view
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create arrays for plotting
        xs, ys, zs = [], [], []
        colors = []
        
        # Map data to colors and position based on mode
        for i, vertex in enumerate(self.planet.grid.vertices):
            # Apply elevation to position
            elevation = self.planet.elevation[i]
            norm = np.linalg.norm(vertex)
            scaled_vertex = vertex * (1 + elevation/self.planet.radius)
            
            xs.append(scaled_vertex[0])
            ys.append(scaled_vertex[1])
            zs.append(scaled_vertex[2])
            
            # Color based on selected mode
            if mode == 'elevation':
                colors.append(self._get_elevation_color(elevation))
            elif mode == 'temperature' and self.planet.temperature is not None:
                colors.append(self._get_temperature_color(self.planet.temperature[i]))
            elif mode == 'precipitation' and self.planet.precipitation is not None:
                colors.append(self._get_precipitation_color(self.planet.precipitation[i]))
            elif mode == 'biome' and self.planet.biome_ids is not None:
                # Get biome color
                from ..core.biome import BiomeClassifier
                biome_classifier = BiomeClassifier(self.planet)
                colors.append(biome_classifier.get_biome_color(self.planet.biome_ids[i]))
            elif mode == 'plates' and hasattr(self.planet, 'plate_ids'):
                # Unique color for each plate
                plate_id = self.planet.plate_ids[i]
                if plate_id >= 0:
                    # Use a hash function to get a consistent color for each plate
                    hue = (plate_id * 0.618033988749895) % 1.0  # Golden ratio to spread colors
                    colors.append(plt.cm.hsv(hue))
                else:
                    colors.append([0.7, 0.7, 0.7])  # Gray for unassigned
            else:
                # Default coloring
                colors.append([0.7, 0.7, 0.7])  # Gray
        
        # Plot the vertices
        ax.scatter(xs, ys, zs, c=colors, s=10, alpha=0.8)
        
        # Plot rivers if available
        if self.planet.rivers and mode in ['elevation', 'biome']:
            for river in self.planet.rivers:
                if len(river) > 1:
                    river_xs = []
                    river_ys = []
                    river_zs = []
                    
                    for idx in river:
                        vertex = self.planet.grid.vertices[idx]
                        elevation = self.planet.elevation[idx]
                        scaled_vertex = vertex * (1 + elevation/self.planet.radius)
                        
                        river_xs.append(scaled_vertex[0])
                        river_ys.append(scaled_vertex[1])
                        river_zs.append(scaled_vertex[2])
                    
                    # Plot the river as a blue line
                    ax.plot(river_xs, river_ys, river_zs, color='blue', linewidth=1, alpha=0.7)
        
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
                'elevation': f'Terrain Elevation (Age: {self.planet.age:.1f} million years)',
                'temperature': 'Surface Temperature (°C)',
                'precipitation': 'Annual Precipitation (mm)',
                'biome': 'Biome Distribution',
                'plates': 'Tectonic Plates'
            }
            plt.title(title_map.get(mode, f'Planet Visualization - {mode.capitalize()}'), fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
            # Update visualization history
            self.planet.last_visualization = {
                'type': '3d',
                'mode': mode,
                'path': save_path,
                'timestamp': self.planet.age
            }
            
            if not show:
                plt.close(fig)
            print(f"3D visualization saved to {save_path}")
            
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_2d(self, save_path=None, mode='elevation', projection='equirectangular',
                     show=False, title=None, figure_size=(12, 6), dpi=100):
        """
        Visualize the planet as a 2D map.
        
        Parameters:
        - save_path: If provided, save visualization to this file path
        - mode: Visualization mode ('elevation', 'temperature', 'precipitation', 'biome', 'plates')
        - projection: Map projection to use ('equirectangular', 'mercator')
        - show: Whether to display the plot
        - title: Custom title for the plot
        - figure_size: Size of the figure in inches
        - dpi: DPI for the saved image
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Set up the map dimensions
        width, height = 1200, 600  # Pixels for visualization
        
        # Create arrays for 2D plotting
        point_x = []
        point_y = []
        point_colors = []
        
        # Project points and get colors based on mode
        for i, vertex in enumerate(self.planet.grid.vertices):
            # Convert 3D coordinates to lat/lon
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
                
            point_x.append(px)
            point_y.append(py)
            
            # Get color based on mode
            if mode == 'elevation':
                point_colors.append(self._get_elevation_color(self.planet.elevation[i]))
            elif mode == 'temperature' and self.planet.temperature is not None:
                point_colors.append(self._get_temperature_color(self.planet.temperature[i]))
            elif mode == 'precipitation' and self.planet.precipitation is not None:
                point_colors.append(self._get_precipitation_color(self.planet.precipitation[i]))
            elif mode == 'biome' and self.planet.biome_ids is not None:
                from ..core.biome import BiomeClassifier
                biome_classifier = BiomeClassifier(self.planet)
                point_colors.append(biome_classifier.get_biome_color(self.planet.biome_ids[i]))
            elif mode == 'plates' and hasattr(self.planet, 'plate_ids'):
                plate_id = self.planet.plate_ids[i]
                if plate_id >= 0:
                    hue = (plate_id * 0.618033988749895) % 1.0  # Golden ratio to spread colors
                    point_colors.append(plt.cm.hsv(hue))
                else:
                    point_colors.append([0.7, 0.7, 0.7])  # Gray for unassigned
            else:
                # Default coloring
                point_colors.append([0.7, 0.7, 0.7])  # Gray
        
        # Plot the points
        scatter = ax.scatter(point_x, point_y, c=point_colors, s=5, alpha=0.8)
        
        # Add rivers if available
        if self.planet.rivers and mode in ['elevation', 'biome']:
            for river in self.planet.rivers:
                if len(river) > 1:
                    river_x = []
                    river_y = []
                    
                    for idx in river:
                        vertex = self.planet.grid.vertices[idx]
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
        
        # Add title
        if title:
            plt.title(title, fontsize=14)
        else:
            title_map = {
                'elevation': f'Terrain Elevation (Age: {self.planet.age:.1f} million years)',
                'temperature': 'Surface Temperature (°C)',
                'precipitation': 'Annual Precipitation (mm)',
                'biome': 'Biome Distribution',
                'plates': 'Tectonic Plates'
            }
            plt.title(title_map.get(mode, f'Planet Visualization - {mode.capitalize()}'), fontsize=14)
        
        # Add a colorbar if not biome mode
        if mode not in ['biome', 'plates']:
            plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
        
        # For biome mode, add a custom legend
        if mode == 'biome' and self.planet.biome_ids is not None:
            from ..core.biome import BiomeClassifier
            biome_classifier = BiomeClassifier(self.planet)
            
            # Get unique biome IDs
            unique_biomes = np.unique(self.planet.biome_ids)
            
            # Create legend patches
            from matplotlib.patches import Patch
            legend_elements = []
            for biome_id in unique_biomes:
                color = biome_classifier.get_biome_color(biome_id)
                legend_elements.append(Patch(facecolor=color, 
                                           label=biome_classifier.get_biome_name(biome_id)))
            
            # Add legend
            ax.legend(handles=legend_elements, loc='lower right', fontsize='small')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
            # Update visualization history
            self.planet.last_visualization = {
                'type': '2d',
                'mode': mode,
                'path': save_path,
                'timestamp': self.planet.age
            }
            
            if not show:
                plt.close(fig)
            print(f"2D visualization saved to {save_path}")
            
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_plates(self, save_path=None, show=False, projection='equirectangular',
                         show_boundaries=True, figure_size=(12, 6), dpi=100):
        """
        Visualize tectonic plates with boundary markers.
        
        Parameters:
        - save_path: If provided, save visualization to this file path
        - show: Whether to display the plot
        - projection: Map projection to use
        - show_boundaries: Whether to highlight plate boundaries
        - figure_size: Size of the figure in inches
        - dpi: DPI for the saved image
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        if not hasattr(self.planet, '_tectonics') or not self.planet._tectonics:
            print("No tectonic simulation attached to this planet.")
            return None
            
        tectonics = self.planet._tectonics
        
        # Use basic 2D visualization with plates mode
        path = self.visualize_2d(
            save_path=save_path, 
            mode='plates', 
            projection=projection,
            show=show, 
            title=f'Tectonic Plates at {self.planet.age:.1f} million years',
            figure_size=figure_size, 
            dpi=dpi
        )
        
        # NOTE: Removed call to _draw_plate_boundaries since it's handled in WorldMapVisualizer
        return path
    
    def visualize_history(self, tectonics, save_path=None, show=False, figure_size=(12, 15), dpi=100):
        """
        Visualize the history of tectonic evolution.
        
        Parameters:
        - tectonics: TectonicSimulation object
        - save_path: If provided, save the visualization to this file path
        - show: Whether to display the plot
        - figure_size: Size of the figure in inches
        - dpi: DPI for the saved image
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        # Check if we have enough history data
        if len(tectonics.history['ages']) < 2:
            print("Not enough history data for visualization.")
            return None
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=figure_size, sharex=True)
        
        # Plot 1: Elevation extremes
        elevations = np.array(tectonics.history['elevations'])
        axes[0].fill_between(tectonics.history['ages'], elevations[:, 0], elevations[:, 1], 
                           color='skyblue', alpha=0.5)
        axes[0].plot(tectonics.history['ages'], elevations[:, 0], 'b-', label='Minimum elevation')
        axes[0].plot(tectonics.history['ages'], elevations[:, 1], 'r-', label='Maximum elevation')
        axes[0].set_ylabel('Elevation (km)')
        axes[0].set_title('Elevation Range Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Number of plates and continental area
        ax2 = axes[1]
        ax2.plot(tectonics.history['ages'], tectonics.history['plate_counts'], 'g-', label='Number of plates')
        ax2.set_ylabel('Number of plates', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(True, alpha=0.3)
        
        # Add continental area on secondary y-axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(tectonics.history['ages'], tectonics.history['continental_area'], 'm-', label='Continental area')
        ax2_twin.set_ylabel('Continental area (%)', color='m')
        ax2_twin.tick_params(axis='y', labelcolor='m')
        
        # Add combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 3: Supercontinent index
        if tectonics.history['supercontinent_index']:
            axes[2].plot(tectonics.history['ages'], tectonics.history['supercontinent_index'], 'k-')
            axes[2].set_ylabel('Supercontinent index')
            axes[2].set_title('Continental Aggregation')
            axes[2].grid(True, alpha=0.3)
            
            # Add horizontal line at supercontinent threshold
            axes[2].axhline(y=1.0 - tectonics.SUPERCONTINENT_THRESHOLD/np.pi, 
                          color='r', linestyle='--', alpha=0.5, 
                          label='Supercontinent threshold')
            axes[2].legend()
        
        # Add events as vertical lines with annotations
        for i, event in enumerate(tectonics.all_events[-10:]):  # Show last 10 events
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
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
            if not show:
                plt.close(fig)
            print(f"Tectonic history visualization saved to {save_path}")
            
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def _get_elevation_color(self, elevation):
        """Get color for elevation visualization"""
        if elevation < 0:
            # Water - blue shades
            depth_ratio = max(-1, elevation / min(-0.001, np.min(self.planet.elevation)))
            return [0, 0, 0.5 + 0.5 * depth_ratio]
        else:
            # Land - green to brown to white
            rel_height = elevation / max(0.001, np.max(self.planet.elevation))
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
    
    def lat_lon_to_mercator(self, lat, lon, width, height):
        """
        Convert latitude/longitude to Mercator projection x,y coordinates.
        
        Parameters:
        - lat: Latitude in degrees
        - lon: Longitude in degrees
        - width: Width of the output image in pixels
        - height: Height of the output image in pixels
        
        Returns:
        - (x, y) coordinates in the Mercator projection
        """
        # Constrain latitude to prevent infinite scaling near poles
        lat = max(min(lat, 85), -85)
        
        # Convert to radians
        lat_rad = math.radians(lat)
        
        # Calculate x-coordinate (same as equirectangular)
        x = (lon + 180) * (width / 360)
        
        # Calculate y-coordinate using Mercator formula
        merc_n = math.log(math.tan((math.pi / 4) + (lat_rad / 2)))
        y = (height / 2) - (width * merc_n / (2 * math.pi))
        
        return x, y

class WorldMapVisualizer(PlanetVisualizer):
    """
    Specialized visualizer for creating detailed 2D world maps.
    Extends PlanetVisualizer with methods for creating different types of world maps.
    """
    
    def __init__(self, planet):
        """
        Initialize the world map visualizer.
        
        Parameters:
        - planet: A Planet object
        """
        super().__init__(planet)
    
def create_tectonic_plate_map(self, save_path=None, width=2048, height=1024, 
                             projection='equirectangular', show_boundaries=True, 
                             show_labels=True, show_velocity=True, show=False,
                             highlight_boundary_types=True):
    """
    Create a detailed tectonic plate map visualization.
    
    Parameters:
    - save_path: Path to save the visualization
    - width: Width of the output image
    - height: Height of the output image
    - projection: Map projection type ('equirectangular' or 'mercator')
    - show_boundaries: Whether to show plate boundaries
    - show_labels: Whether to show plate labels
    - show_velocity: Whether to show plate velocity vectors
    - show: Whether to display the plot
    - highlight_boundary_types: Whether to color boundary types differently
    
    Returns:
    - Path to the saved visualization if save_path is provided
    """
    print("Generating tectonic plate map...")
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create arrays for elevation data (for terrain coloring)
    elevation_grid = np.zeros((height, width), dtype=np.float32)
    
    # Create arrays for plate data
    plate_grid = np.full((height, width), -1, dtype=np.int32)
    
    # Sample the planet's surface at each pixel position
    print("  Sampling plate data...")
    for y in range(height):
        for x in range(width):
            # Convert pixel to lat/lon
            if projection == 'mercator':
                lat = 90 - y * 180 / height
                lon = x * 360 / width - 180
                # Adjust mercator projection
                if abs(lat) > 85:  # Limit latitude for mercator
                    continue
                lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / height))))
            else:
                lat = 90 - y * 180 / height
                lon = x * 360 / width - 180
            
            # Convert lat/lon to 3D point
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            point = np.array([
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad)
            ])
            
            # Find nearest vertex
            distances = np.linalg.norm(self.planet.grid.vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Store plate ID
            plate_grid[y, x] = self.planet.plate_ids[nearest_idx]
            
            # Store elevation for terrain shading
            elevation_grid[y, x] = self.planet.elevation[nearest_idx]
    
    # Define colors for each plate using a consistent method
    print("  Creating plate color mapping...")
    plate_colors = {}
    
    for plate_id in set(self.planet.plate_ids):
        if plate_id >= 0:
            # Use golden ratio to distribute colors evenly around the color wheel
            hue = (plate_id * 0.618033988749895) % 1.0
            
            # Oceanic plates get more blue/green hues, continental more orange/red
            for plate in self.planet._tectonics.plates:
                if plate['id'] == plate_id:
                    if plate['is_oceanic']:
                        # Shift hue towards blue/green
                        hue = (hue * 0.5 + 0.5) % 1.0
                    else:
                        # Shift hue towards orange/red
                        hue = (hue * 0.5) % 1.0
            
            # Create distinct saturation and value based on plate properties
            # Smaller plates get more saturated colors
            for plate in self.planet._tectonics.plates:
                if plate['id'] == plate_id:
                    saturation = min(1.0, 0.5 + 0.5 / (plate['area'] * 10))
                    value = 0.9
                    break
            else:
                # Default values if plate not found
                saturation = 0.7
                value = 0.8
            
            # Convert HSV to RGB
            plate_colors[plate_id] = plt.cm.hsv(hue)
    
    # Render the plate map with subtle terrain shading
    print("  Rendering plates...")
    
    # Create image array
    plate_image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Fill with plate colors and apply subtle terrain shading
    for y in range(height):
        for x in range(width):
            plate_id = plate_grid[y, x]
            
            if plate_id >= 0 and plate_id in plate_colors:
                # Start with plate color
                color = plate_colors[plate_id][:3]  # Extract RGB, discard alpha
                
                # Apply subtle terrain shading
                elevation = elevation_grid[y, x]
                if elevation >= 0:  # Land
                    # More pronounced shading on land
                    shade_factor = 0.8 + 0.2 * min(1.0, elevation / 5.0)
                    color = color * shade_factor
                else:  # Ocean
                    # Subtle darkening for deeper ocean
                    shade_factor = 0.9 + 0.1 * (elevation / min(-0.001, np.min(self.planet.elevation)))
                    color = color * shade_factor
                
                plate_image[y, x] = color
            else:
                # Default gray for unassigned points
                plate_image[y, x] = [0.7, 0.7, 0.7]
    
    # Display the plate image
    ax.imshow(plate_image, origin='upper')
    
    # Draw plate boundaries if requested
    if show_boundaries and hasattr(self.planet, '_tectonics'):
        print("  Drawing plate boundaries...")
        
        # Identify boundary pixels
        boundary_map = np.zeros((height, width), dtype=bool)
        boundary_types = {}  # Maps (y, x) coordinates to boundary type
        
        # Create boundary mask from plate grid
        for y in range(1, height-1):
            for x in range(1, width-1):
                if plate_grid[y, x] >= 0:
                    # Check if any neighbor is a different plate
                    for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                        if 0 <= ny < height and 0 <= nx < width:
                            if plate_grid[ny, nx] >= 0 and plate_grid[ny, nx] != plate_grid[y, x]:
                                boundary_map[y, x] = True
                                
                                # Determine boundary type if original data is available
                                if hasattr(self.planet, '_tectonics') and highlight_boundary_types:
                                    # Get the nearest vertex
                                    # Calculate lat/lon for this pixel
                                    if projection == 'mercator':
                                        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / height))))
                                        lon = -180 + x * 360 / width
                                    else:
                                        lat = 90 - y * 180 / height
                                        lon = -180 + x * 360 / width
                                    
                                    # Convert to 3D point
                                    lat_rad = np.radians(lat)
                                    lon_rad = np.radians(lon)
                                    point = np.array([
                                        np.cos(lat_rad) * np.cos(lon_rad),
                                        np.cos(lat_rad) * np.sin(lon_rad),
                                        np.sin(lat_rad)
                                    ])
                                    
                                    # Find nearest vertex
                                    distances = np.linalg.norm(self.planet.grid.vertices - point, axis=1)
                                    nearest_idx = np.argmin(distances)
                                    
                                    # Find nearest neighbor with different plate
                                    nearest_diff_plate = None
                                    nearest_dist = float('inf')
                                    
                                    neighbors = self.planet.grid.get_vertex_neighbors()[nearest_idx]
                                    for n in neighbors:
                                        if self.planet.plate_ids[n] != plate_grid[y, x] and self.planet.plate_ids[n] >= 0:
                                            # This is a boundary
                                            boundary_types[(y, x)] = self.planet._tectonics._calculate_boundary_type(
                                                nearest_idx, 
                                                self.planet.plate_ids[nearest_idx], 
                                                self.planet.plate_ids[n]
                                            )
                                            break
                                            
                                    if (y, x) not in boundary_types:
                                        # Default if we couldn't determine the type
                                        boundary_types[(y, x)] = "transform"
                                
                                break
        
        # Draw boundaries with appropriate colors
        boundary_y, boundary_x = np.where(boundary_map)
        
        if highlight_boundary_types:
            # Separate boundaries by type
            convergent_x, convergent_y = [], []
            divergent_x, divergent_y = [], []
            transform_x, transform_y = [], []
            
            for y, x in zip(boundary_y, boundary_x):
                boundary_type = boundary_types.get((y, x), "unknown")
                
                if boundary_type == "convergent":
                    convergent_x.append(x)
                    convergent_y.append(y)
                elif boundary_type == "divergent":
                    divergent_x.append(x)
                    divergent_y.append(y)
                elif boundary_type == "transform":
                    transform_x.append(x)
                    transform_y.append(y)
            
            # Plot each boundary type with different color
            if convergent_x:
                ax.scatter(convergent_x, convergent_y, color='red', s=2, alpha=0.8, 
                          label='Convergent')
            if divergent_x:
                ax.scatter(divergent_x, divergent_y, color='blue', s=2, alpha=0.8,
                          label='Divergent')
            if transform_x:
                ax.scatter(transform_x, transform_y, color='purple', s=2, alpha=0.8,
                          label='Transform')
                
            # Add legend
            ax.legend(loc='lower right', fontsize=10)
        else:
            # Plot all boundaries with same color
            ax.scatter(boundary_x, boundary_y, color='black', s=2, alpha=0.8)
    
    # Add plate labels if requested
    if show_labels and hasattr(self.planet, '_tectonics'):
        print("  Adding plate labels...")
        
        for plate in self.planet._tectonics.plates:
            if 'center' in plate:
                center = plate['center']
                # Convert center to lat/lon
                x, y, z = center
                lat = np.arcsin(z / np.linalg.norm(center)) * 180 / np.pi
                lon = np.arctan2(y, x) * 180 / np.pi
                
                if projection == 'mercator':
                    px, py = self.lat_lon_to_mercator(lat, lon, width, height)
                else:
                    px = (lon + 180) * width / 360
                    py = (90 - lat) * height / 180
                
                # Skip if outside view
                if not (0 <= px < width and 0 <= py < height):
                    continue
                
                # Create label with plate info
                plate_type = "Oceanic" if plate.get('is_oceanic', True) else "Continental"
                
                # Calculate velocity in cm/year
                velocity_mag = np.linalg.norm(plate['velocity']) * 100  # Convert to cm/year
                
                label_text = f"Plate {plate['id']}\n({plate_type})\n{velocity_mag:.1f} cm/yr"
                
                # Draw text box
                bbox_props = dict(
                    boxstyle="round,pad=0.3", 
                    fc=tuple(list(plate_colors[plate['id']][:3]) + [0.7]),  # Add transparency
                    ec="black", 
                    lw=1
                )
                ax.text(px, py, label_text, ha='center', va='center', fontsize=10, 
                       bbox=bbox_props)
    
    # Show velocity vectors if requested
    if show_velocity and hasattr(self.planet, '_tectonics'):
        print("  Adding velocity vectors...")
        
        for plate in self.planet._tectonics.plates:
            if 'center' in plate and 'velocity' in plate:
                center = plate['center']
                # Convert center to lat/lon
                x, y, z = center
                lat = np.arcsin(z / np.linalg.norm(center)) * 180 / np.pi
                lon = np.arctan2(y, x) * 180 / np.pi
                
                if projection == 'mercator':
                    px, py = self.lat_lon_to_mercator(lat, lon, width, height)
                else:
                    px = (lon + 180) * width / 360
                    py = (90 - lat) * height / 180
                
                # Skip if outside view
                if not (0 <= px < width and 0 <= py < height):
                    continue
                
                # Calculate velocity direction and magnitude
                velocity = np.array(plate['velocity'])
                vel_mag = np.linalg.norm(velocity)
                
                if vel_mag > 0:
                    # Scale for better visibility
                    vel_scale = 5000 * vel_mag
                    
                    # Project 3D velocity onto 2D surface
                    # This is a simplified projection - for a proper one we'd need
                    # to do a more complex projection based on the coordinate system
                    vel_dir = velocity / vel_mag
                    
                    # Calculate tangential component
                    center_norm = center / np.linalg.norm(center)
                    normal_component = np.dot(vel_dir, center_norm)
                    tangential = vel_dir - normal_component * center_norm
                    
                    # Convert to lat/lon direction
                    # For simplicity, we'll use the fact that:
                    # - x component affects longitude (east-west)
                    # - y component affects longitude (east-west)
                    # - z component affects latitude (north-south)
                    dx = tangential[0] + tangential[1]  # Combined longitude effect
                    dy = -tangential[2]  # Negative because y increases southward in the plot
                    
                    # Normalize and scale
                    dir_mag = np.sqrt(dx*dx + dy*dy)
                    if dir_mag > 0:
                        dx = dx / dir_mag * vel_scale
                        dy = dy / dir_mag * vel_scale
                        
                        # Draw arrow
                        ax.arrow(px, py, dx, dy, head_width=15, head_length=15, 
                               fc=plate_colors[plate['id']], ec='black', 
                               length_includes_head=True, linewidth=2)
    
    # Set axis limits and labels
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add lat/lon grid
    self._add_lat_lon_grid(ax, width, height, projection)
    
    # Add title and legend
    plt.title(f"Tectonic Plates at {self.planet.age:.1f} million years", fontsize=14)
    
    # Add explanation key
    legend_elements = []
    
    if show_boundaries and highlight_boundary_types:
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Convergent Boundary'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Divergent Boundary'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Transform Boundary')
        ])
    
    # Add legend for plate types
    legend_elements.extend([
        plt.Line2D([0], [0], color='w', markerfacecolor='#cc5500', marker='s', markersize=10, label='Continental Plate'),
        plt.Line2D([0], [0], color='w', markerfacecolor='#0066cc', marker='s', markersize=10, label='Oceanic Plate')
    ])
    
    # Position legend in lower left
    plt.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Tectonic plate map saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return save_path if save_path else None
    
def create_realistic_world_map(self, save_path=None, width=2048, height=1024, 
                               projection='equirectangular', show_rivers=True, 
                               show_mountains=True, show=False, lighting_angle=315):
    """
    Create a realistic-looking world map with terrain, rivers, and features.
    
    Parameters:
    - save_path: Path to save the visualization
    - width: Width of the output image
    - height: Height of the output image
    - projection: Map projection type
    - show_rivers: Whether to show rivers
    - show_mountains: Whether to show mountain ranges
    - show: Whether to display the plot
    - lighting_angle: Angle of light source for relief shading (in degrees)
    
    Returns:
    - Path to the saved visualization if save_path is provided
    """
    # Create a higher-resolution map image
    print("Generating realistic world map...")
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create image array for the map
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Create arrays for elevation data
    elevation_grid = np.zeros((height, width), dtype=np.float32)
    
    # Convert lighting angle to radians
    light_rad = np.radians(lighting_angle)
    light_x = np.cos(light_rad)
    light_y = np.sin(light_rad)
    
    # Sample the planet's surface at each pixel position
    print("  Sampling surface data...")
    for y in range(height):
        for x in range(width):
            # Convert pixel to lat/lon
            if projection == 'mercator':
                lat = 90 - y * 180 / height
                lon = x * 360 / width - 180
                # Adjust mercator projection
                if abs(lat) > 85:  # Limit latitude for mercator
                    continue
                lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / height))))
            else:
                lat = 90 - y * 180 / height
                lon = x * 360 / width - 180
            
            # Convert lat/lon to 3D point
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            point = np.array([
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad)
            ])
            
            # Find nearest vertex
            distances = np.linalg.norm(self.planet.grid.vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Get elevation and store it in the grid
            elevation = self.planet.elevation[nearest_idx]
            elevation_grid[y, x] = elevation
            
            # Default color (for fallback)
            color = self._get_elevation_color(elevation)
            
            # Color based on elevation and biome if available
            if hasattr(self.planet, 'biome_ids') and self.planet.biome_ids is not None:
                from ..core.biome import BiomeClassifier
                biome_classifier = BiomeClassifier(self.planet)
                biome_id = self.planet.biome_ids[nearest_idx]
                base_color = biome_classifier.get_biome_color(biome_id)
                
                # Apply different coloring modifications based on biome type
                if biome_id == 0:  # Ocean
                    # Gradient coloring for ocean depth
                    depth_factor = (elevation / min(self.planet.elevation)) if min(self.planet.elevation) < 0 else 0
                    depth_factor = max(0, min(1, depth_factor))
                    # Deeper = darker blue
                    deep_color = np.array([0.0, 0.0, 0.2])
                    shallow_color = np.array([0.0, 0.4, 0.8])
                    color = deep_color + depth_factor * (shallow_color - deep_color)
                    
                    # Add a subtle shoreline effect
                    nearby_land = False
                    if x > 0 and elevation_grid[y, x-1] >= 0: nearby_land = True
                    elif x < width-1 and elevation_grid[y, x+1] >= 0: nearby_land = True
                    elif y > 0 and elevation_grid[y-1, x] >= 0: nearby_land = True
                    elif y < height-1 and elevation_grid[y+1, x] >= 0: nearby_land = True
                    
                    if nearby_land:
                        # Add a slight teal tint for shallow water near shores
                        color += np.array([0.0, 0.2, 0.1])
                        color = np.clip(color, 0, 1)
                
                elif elevation < 0:  # Underwater but not classified as ocean
                    # Use ocean coloring
                    depth_factor = elevation / -10.0  # Normalize to typical ocean depths
                    depth_factor = max(0, min(1, depth_factor))
                    color = np.array([0.0, 0.1, 0.4]) + depth_factor * np.array([0.0, 0.3, 0.4])
                
                else:  # Land with biome
                    # Start with biome base color
                    color = np.array(base_color)
                    
                    # Adjust colors based on elevation for realistic terrain
                    if biome_id in [1, 11]:  # Ice/snow or mountain
                        # Enhance whiteness at higher elevations
                        elev_factor = min(1.0, elevation / 5.0)
                        snow_color = np.array([1.0, 1.0, 1.0])
                        color = color * (1 - elev_factor) + snow_color * elev_factor
                    
                    elif biome_id in [2, 3, 4, 5, 6, 8, 9, 10]:  # Various green biomes
                        # Slight variation in greens based on elevation
                        elev_factor = min(1.0, elevation / 3.0)
                        # Higher elevation = slightly more yellow/brown
                        color[0] = min(1.0, color[0] + elev_factor * 0.2)
                        color[1] = min(1.0, color[1] - elev_factor * 0.1)
                    
                    elif biome_id == 7:  # Desert
                        # Subtle dune effect
                        if (x % 20 < 10 and y % 20 < 10) or (x % 20 >= 10 and y % 20 >= 10):
                            color = color * 0.95  # Slightly darker
            
            # Store the color in the image
            image[y, x] = color
    
    # Apply relief shading using the elevation grid
    print("  Applying relief shading...")
    # Calculate gradients for relief shading (approximating slope)
    dx = np.zeros_like(elevation_grid)
    dy = np.zeros_like(elevation_grid)
    
    # Calculate gradients using central differences
    dx[:, 1:-1] = (elevation_grid[:, 2:] - elevation_grid[:, :-2]) / 2.0
    dy[1:-1, :] = (elevation_grid[2:, :] - elevation_grid[:-2, :]) / 2.0
    
    # Special case for edges
    dx[:, 0] = elevation_grid[:, 1] - elevation_grid[:, 0]
    dx[:, -1] = elevation_grid[:, -1] - elevation_grid[:, -2]
    dy[0, :] = elevation_grid[1, :] - elevation_grid[0, :]
    dy[-1, :] = elevation_grid[-1, :] - elevation_grid[-2, :]
    
    # Compute hillshade
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    
    hillshade = np.cos(slope) * np.cos(np.pi/2 - np.radians(45)) + \
               np.sin(slope) * np.sin(np.pi/2 - np.radians(45)) * \
               np.cos(aspect - light_rad)
    
    # Apply hillshade to image
    for y in range(height):
        for x in range(width):
            if elevation_grid[y, x] >= 0:  # Only apply to land
                # Stronger effect on land
                shading = 0.7 + 0.3 * hillshade[y, x]
            else:
                # Subtle effect in water
                shading = 0.9 + 0.1 * hillshade[y, x]
            
            # Apply shading
            image[y, x] = np.clip(image[y, x] * shading, 0, 1)
    
    # Final touches: Add coastline enhancement
    print("  Enhancing coastlines...")
    coastline_mask = np.zeros((height, width), dtype=bool)
    for y in range(1, height-1):
        for x in range(1, width-1):
            # Check if this is a coastline pixel
            is_land = elevation_grid[y, x] >= 0
            has_water_neighbor = False
            
            # Check 8 surrounding pixels
            for ny in range(y-1, y+2):
                for nx in range(x-1, x+2):
                    if ny == y and nx == x:
                        continue
                    if elevation_grid[ny, nx] < 0:
                        has_water_neighbor = True
                        break
                if has_water_neighbor:
                    break
            
            # Mark coastline pixels
            if is_land and has_water_neighbor:
                coastline_mask[y, x] = True
    
    # Enhance coastlines by darkening slightly
    for y in range(height):
        for x in range(width):
            if coastline_mask[y, x]:
                image[y, x] = image[y, x] * 0.9
    
    # Display the image
    ax.imshow(image, origin='upper')
    
    # Add rivers if requested
    if show_rivers and self.planet.rivers:
        print("  Adding rivers...")
        for river in self.planet.rivers:
            if len(river) > 1:
                river_x = []
                river_y = []
                
                for idx in river:
                    vertex = self.planet.grid.vertices[idx]
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
                    # River width based on position in the river array
                    # More vertices = larger river (approximates flow accumulation)
                    river_size = min(3.0, 0.5 + len(river) / 30.0)
                    
                    # Add subtle glow effect around larger rivers
                    if river_size > 1.5:
                        ax.plot(river_x, river_y, color='royalblue', linewidth=river_size+1.0, alpha=0.2)
                    
                    # Draw the river
                    ax.plot(river_x, river_y, color='royalblue', linewidth=river_size, alpha=0.8)
    
    # Add mountain markers if requested
    if show_mountains:
        print("  Adding mountain markers...")
        mountain_x = []
        mountain_y = []
        mountain_size = []
        
        # Get neighbors for checking if vertices are local maxima
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        # Identify mountain peaks
        for i, elev in enumerate(self.planet.elevation):
            if elev > 1.0:  # Threshold for mountains
                # Check if it's a local maximum
                is_peak = True
                for n in neighbors[i]:
                    if self.planet.elevation[n] > elev:
                        is_peak = False
                        break
                
                if is_peak:
                    # Convert to lat/lon
                    vertex = self.planet.grid.vertices[i]
                    x, y, z = vertex
                    lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
                    lon = np.arctan2(y, x) * 180 / np.pi
                    
                    if projection == 'mercator':
                        px, py = self.lat_lon_to_mercator(lat, lon, width, height)
                    else:
                        px = (lon + 180) * width / 360
                        py = (90 - lat) * height / 180
                    
                    if 0 <= px < width and 0 <= py < height:
                        mountain_x.append(px)
                        mountain_y.append(py)
                        # Size based on elevation
                        mountain_size.append(20 + elev * 80)
        
        if mountain_x and mountain_y:
            # Draw mountains with two markers for better appearance:
            # A dark base and a lighter top
            
            # Draw the dark base first
            ax.scatter(mountain_x, mountain_y, s=mountain_size, marker='^', 
                     color='saddlebrown', edgecolors='black', alpha=0.7)
            
            # Draw a smaller white peak on top
            smaller_size = [s * 0.6 for s in mountain_size]
            ax.scatter(mountain_x, mountain_y, s=smaller_size, marker='^', 
                     color='white', edgecolors='none', alpha=0.7)
    
    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add lat/lon grid
    self._add_lat_lon_grid(ax, width, height, projection)
    
    # Add title
    plt.title(f"World Map at {self.planet.age:.1f} million years", fontsize=14)
    
    # Add scale bar at bottom center
    self._add_scale_bar(ax, width, height)
    
    # Add compass rose in bottom right corner
    self._add_compass_rose(ax, width, height)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Realistic world map saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return save_path if save_path else None

def _add_scale_bar(self, ax, width, height):
    """Add a scale bar to the map"""
    # Calculate scale based on planet radius at the equator
    # For Earth, radius is 6371 km
    km_per_degree = 2 * np.pi * self.planet.radius / 360
    
    # Find a nice round number of km for the scale bar
    map_width_km = km_per_degree * 360  # Full map width in km
    target_length_pixels = width * 0.2  # Aim for 20% of map width
    
    # Calculate how many km would be about 20% of map width
    km_target = map_width_km * 0.2
    
    # Round to a nice number
    scale_lengths = [1000, 2000, 5000, 10000]
    scale_km = min(scale_lengths, key=lambda x: abs(x - km_target))
    
    # Convert to pixels
    scale_pixels = scale_km / map_width_km * width
    
    # Position of scale bar (bottom center)
    bar_y = height * 0.95  # 95% down
    bar_x_start = width/2 - scale_pixels/2
    bar_x_end = width/2 + scale_pixels/2
    
    # Draw the scale bar
    ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 'k-', linewidth=3)
    ax.plot([bar_x_start, bar_x_start], [bar_y-5, bar_y+5], 'k-', linewidth=2)
    ax.plot([bar_x_end, bar_x_end], [bar_y-5, bar_y+5], 'k-', linewidth=2)
    
    # Add label
    if scale_km >= 1000:
        scale_label = f"{scale_km//1000:,d} thousand km"
    else:
        scale_label = f"{scale_km:,d} km"
        
    ax.text(width/2, bar_y - 15, scale_label, ha='center', va='top', fontsize=10)

def _add_compass_rose(self, ax, width, height):
    """Add a compass rose to the map"""
    # Position in bottom right corner
    center_x = width * 0.9
    center_y = height * 0.9
    size = min(width, height) * 0.05
    
    # Draw the compass points
    ax.plot([center_x, center_x], [center_y-size, center_y+size], 'k-', linewidth=2)
    ax.plot([center_x-size, center_x+size], [center_y, center_y], 'k-', linewidth=2)
    
    # Add the letters (N, E, S, W)
    ax.text(center_x, center_y-size-10, 'N', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(center_x+size+10, center_y, 'E', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.text(center_x, center_y+size+10, 'S', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(center_x-size-10, center_y, 'W', ha='right', va='center', fontsize=10, fontweight='bold')
    
    # Add a circle around it
    theta = np.linspace(0, 2*np.pi, 100)
    x = center_x + size * 0.8 * np.cos(theta)
    y = center_y + size * 0.8 * np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=1, alpha=0.5)
    
def create_height_map(self, save_path=None, width=2048, height=1024, 
                         projection='equirectangular', grayscale=True, 
                         color_ramp='terrain', show=False, relief_shading=True,
                         lighting_angle=315, depth_curve='linear'):
    """
    Create a height map suitable for use in game engines or 3D software.
    
    Parameters:
    - save_path: Path to save the visualization
    - width: Width of the output image
    - height: Height of the output image
    - projection: Map projection type
    - grayscale: Whether to output a grayscale heightmap or colored relief
    - color_ramp: Color scheme for non-grayscale output ('terrain', 'hypsometric', 'ocean')
    - show: Whether to display the plot
    - relief_shading: Whether to apply hillshade effect
    - lighting_angle: Angle of light source for relief shading (in degrees)
    - depth_curve: How to curve depth values ('linear', 'logarithmic', 'sqrt')
    
    Returns:
    - Path to the saved visualization if save_path is provided
    """
    print("Generating heightmap...")
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create image arrays
    elevation_grid = np.zeros((height, width), dtype=np.float32)
    
    # Find elevation range for normalization
    min_elev = np.min(self.planet.elevation)
    max_elev = np.max(self.planet.elevation)
    elevation_range = max_elev - min_elev
    
    # Convert lighting angle to radians
    light_rad = np.radians(lighting_angle)
    light_x = np.cos(light_rad)
    light_y = np.sin(light_rad)
    
    # Sample the planet's surface at each pixel position
    print("  Sampling elevation data...")
    for y in range(height):
        for x in range(width):
            # Convert pixel to lat/lon
            if projection == 'mercator':
                lat = 90 - y * 180 / height
                lon = x * 360 / width - 180
                # Adjust mercator projection
                if abs(lat) > 85:  # Limit latitude for mercator
                    continue
                lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / height))))
            else:
                lat = 90 - y * 180 / height
                lon = x * 360 / width - 180
            
            # Convert lat/lon to 3D point
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            point = np.array([
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad)
            ])
            
            # Find nearest vertex
            distances = np.linalg.norm(self.planet.grid.vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Get elevation
            elevation_grid[y, x] = self.planet.elevation[nearest_idx]
    
    # Apply depth curve transformations for better visualization of oceanic features
    if depth_curve == 'logarithmic':
        # Apply log curve to negative values (ocean depths)
        mask = elevation_grid < 0
        if np.any(mask):
            min_neg = np.min(elevation_grid[mask])
            # Normalize negative values to 0-1 range, but inverted (0 is deepest, 1 is at sea level)
            norm_neg = 1.0 - (elevation_grid[mask] - min_neg) / (0 - min_neg)
            # Apply log transform (add small value to avoid log(0))
            log_neg = np.log(norm_neg + 0.01) / np.log(1.01)
            # Convert back to original range
            elevation_grid[mask] = -(log_neg * (0 - min_neg)) + min_neg
    
    elif depth_curve == 'sqrt':
        # Apply square root curve to negative values
        mask = elevation_grid < 0
        if np.any(mask):
            min_neg = np.min(elevation_grid[mask])
            # Normalize negative values to 0-1 range, but inverted
            norm_neg = 1.0 - (elevation_grid[mask] - min_neg) / (0 - min_neg)
            # Apply sqrt transform
            sqrt_neg = np.sqrt(norm_neg)
            # Convert back to original range
            elevation_grid[mask] = -(sqrt_neg * (0 - min_neg)) + min_neg
    
    # Calculate hillshade if needed
    if relief_shading:
        print("  Calculating hillshade...")
        # Calculate gradients using central differences
        dx = np.zeros_like(elevation_grid)
        dy = np.zeros_like(elevation_grid)
        
        dx[:, 1:-1] = (elevation_grid[:, 2:] - elevation_grid[:, :-2]) / 2.0
        dy[1:-1, :] = (elevation_grid[2:, :] - elevation_grid[:-2, :]) / 2.0
        
        # Handle edges
        dx[:, 0] = elevation_grid[:, 1] - elevation_grid[:, 0]
        dx[:, -1] = elevation_grid[:, -1] - elevation_grid[:, -2]
        dy[0, :] = elevation_grid[1, :] - elevation_grid[0, :]
        dy[-1, :] = elevation_grid[-1, :] - elevation_grid[-2, :]
        
        # Compute hillshade
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        
        hillshade = np.cos(slope) * np.cos(np.pi/2 - np.radians(45)) + \
                   np.sin(slope) * np.sin(np.pi/2 - np.radians(45)) * \
                   np.cos(aspect - light_rad)
    
    # Normalize elevation to 0-1 range for visualization
    norm_elevation = (elevation_grid - min_elev) / elevation_range
    
    if grayscale:
        print("  Creating grayscale heightmap...")
        # Use the normalized elevation directly
        if relief_shading:
            # Blend hillshade with elevation
            shaded_elevation = norm_elevation * 0.7 + hillshade * 0.3
            ax.imshow(shaded_elevation, cmap='gray', vmin=0, vmax=1, origin='upper')
        else:
            ax.imshow(norm_elevation, cmap='gray', vmin=0, vmax=1, origin='upper')
        
        # Add colorbar and title
        cbar = plt.colorbar(shrink=0.75)
        cbar.set_label(f"Elevation (normalized: {min_elev:.1f} to {max_elev:.1f} km)")
        plt.title("Heightmap", fontsize=14)
        
    else:
        print(f"  Creating colored {color_ramp} elevation map...")
        # Create a colored elevation map
        if color_ramp == 'terrain':
            # Better custom colormap for terrain (more natural looking)
            cmap_colors = [
                (0.0, [0.0, 0.0, 0.5]),         # Deepest ocean - dark blue
                (0.25, [0.0, 0.3, 0.8]),        # Mid-depth ocean - blue
                (0.45, [0.0, 0.6, 1.0]),        # Shallow ocean - light blue
                (0.5, [0.9, 0.9, 0.2]),         # Coast - yellow
                (0.55, [0.0, 0.6, 0.0]),        # Lowland - green
                (0.7, [0.5, 0.3, 0.0]),         # Hills - brown
                (0.9, [0.7, 0.7, 0.7]),         # Mountains - gray
                (1.0, [1.0, 1.0, 1.0])          # Highest peaks - white
            ]
            
            cmap = LinearSegmentedColormap.from_list("terrain_cmap", cmap_colors)
            
        elif color_ramp == 'hypsometric':
            # Standard hypsometric tints
            cmap_colors = [
                (0.0, [0.0, 0.0, 0.5]),         # Deepest ocean - dark blue
                (0.3, [0.0, 0.3, 0.8]),         # Mid-depth ocean - blue
                (0.5, [0.2, 0.7, 0.7]),         # Shallow ocean - cyan
                (0.501, [0.2, 0.6, 0.0]),       # Coastal plains - green
                (0.6, [0.4, 0.7, 0.0]),         # Low elevations - lime
                (0.7, [0.7, 0.7, 0.0]),         # Mid elevations - yellow
                (0.8, [0.7, 0.5, 0.0]),         # High elevations - orange/tan
                (0.9, [0.6, 0.2, 0.0]),         # Mountains - brown
                (1.0, [1.0, 1.0, 1.0])          # Highest peaks - white
            ]
            
            cmap = LinearSegmentedColormap.from_list("hypsometric_cmap", cmap_colors)
            
        elif color_ramp == 'ocean':
            # Focus on ocean depths
            cmap_colors = [
                (0.0, [0.0, 0.0, 0.2]),         # Deepest ocean - almost black
                (0.1, [0.0, 0.0, 0.4]),         # Very deep ocean - dark blue
                (0.3, [0.0, 0.2, 0.6]),         # Deep ocean - medium blue
                (0.5, [0.0, 0.4, 0.8]),         # Mid-depth ocean - blue
                (0.7, [0.0, 0.6, 0.9]),         # Shallow ocean - light blue
                (0.9, [0.0, 0.8, 1.0]),         # Very shallow - cyan
                (0.95, [0.0, 0.9, 1.0]),        # Shoreline - lighter cyan
                (1.0, [0.9, 0.9, 0.5])          # Land - tan/beige (flat color)
            ]
            
            cmap = LinearSegmentedColormap.from_list("ocean_cmap", cmap_colors)
            
        else:
            # Fallback to matplotlib terrain colormap
            cmap = plt.cm.terrain
        
        # Create a colored image
        if relief_shading:
            # Apply hillshade effect to add terrain detail
            rgb_image = cmap(norm_elevation)[:, :, :3]  # Extract RGB components
            
            # Apply hillshade (brighten or darken based on slope direction)
            for i in range(3):
                rgb_image[:, :, i] = np.clip(rgb_image[:, :, i] * (0.5 + 0.5 * hillshade), 0, 1)
            
            ax.imshow(rgb_image, origin='upper')
        else:
            ax.imshow(norm_elevation, cmap=cmap, origin='upper')
        
        # Add colorbar and title
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(min_elev, max_elev), cmap=cmap), shrink=0.75)
        cbar.set_label("Elevation (km)")
        plt.title(f"Elevation Map ({color_ramp.capitalize()} Color Palette)", fontsize=14)
    
    # Add lat/lon grid
    self._add_lat_lon_grid(ax, width, height, projection)
    
    # Add sea level contour
    sea_level = (0 - min_elev) / elevation_range
    ax.contour(norm_elevation, levels=[sea_level], colors=['#0000FF'], alpha=0.5, linewidths=1)
    
    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add info about min/max elevation
    plt.figtext(0.01, 0.01, f"Min: {min_elev:.1f} km", fontsize=8, color='black')
    plt.figtext(0.01, 0.03, f"Max: {max_elev:.1f} km", fontsize=8, color='black')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Heightmap saved to {save_path}")
        
        # If grayscale, also save a raw version if needed for game engines
        if grayscale and save_path.endswith('.png'):
            raw_path = save_path.replace('.png', '.raw')
            
            # Save as 16-bit raw heightmap
            raw_data = (norm_elevation * 65535).astype(np.uint16)
            
            with open(raw_path, 'wb') as f:
                raw_data.tofile(f)
            
            print(f"Raw 16-bit heightmap saved to {raw_path}")
            
            # Create metadata JSON file
            meta_path = save_path.replace('.png', '.json')
            metadata = {
                "width": width,
                "height": height,
                "min_elevation_km": float(min_elev),
                "max_elevation_km": float(max_elev),
                "projection": projection,
                "bit_depth": 16,
                "format": "RAW"
            }
            
            with open(meta_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            print(f"Heightmap metadata saved to {meta_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return save_path if save_path else None

# Helper function for color map generation
def _create_custom_colormap(self, colors_array):
    """Create a custom colormap from an array of [position, [r, g, b]] values"""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Extract positions and colors
    positions = [p for p, c in colors_array]
    colors = [c for p, c in colors_array]
    
    # Create and return the colormap
    return LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))
    
    def _add_lat_lon_grid(self, ax, width, height, projection):
        """Add latitude and longitude grid lines to the map"""
        # Latitude lines
        for lat in range(-80, 81, 20):
            x_points = []
            y_points = []
            for lon in range(-180, 181, 2):
                if projection == 'mercator':
                    px, py = self.lat_lon_to_mercator(lat, lon, width, height)
                else:
                    px = (lon + 180) * width / 360
                    py = (90 - lat) * height / 180
                
                if 0 <= px < width and 0 <= py < height:
                    x_points.append(px)
                    y_points.append(py)
            
            if x_points and y_points:
                ax.plot(x_points, y_points, 'k-', linewidth=0.5, alpha=0.3)
                # Label at the edges
                if projection != 'mercator' or abs(lat) < 80:
                    ax.text(0, y_points[0], f"{lat}°", fontsize=8, ha='left', va='center', alpha=0.7)
                    ax.text(width, y_points[-1], f"{lat}°", fontsize=8, ha='right', va='center', alpha=0.7)
        
        # Longitude lines
        for lon in range(-180, 181, 30):
            x_points = []
            y_points = []
            for lat in range(-80, 81, 1):
                if projection == 'mercator':
                    px, py = self.lat_lon_to_mercator(lat, lon, width, height)
                else:
                    px = (lon + 180) * width / 360
                    py = (90 - lat) * height / 180
                
                if 0 <= px < width and 0 <= py < height:
                    x_points.append(px)
                    y_points.append(py)
            
            if x_points and y_points:
                ax.plot(x_points, y_points, 'k-', linewidth=0.5, alpha=0.3)
                # Label at the bottom
                ax.text(x_points[-1], height, f"{lon}°", fontsize=8, ha='center', va='top', alpha=0.7)