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
                                 show_labels=False, show=False):
        """
        Create a detailed tectonic plate map visualization.
        
        Parameters:
        - save_path: Path to save the visualization
        - width: Width of the output image
        - height: Height of the output image
        - projection: Map projection type ('equirectangular' or 'mercator')
        - show_boundaries: Whether to show plate boundaries
        - show_labels: Whether to show plate labels
        - show: Whether to display the plot
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create arrays for 2D plotting
        point_x = []
        point_y = []
        point_colors = []
        
        # Project points and get colors based on plate IDs
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
            
            # Get plate ID color
            if hasattr(self.planet, 'plate_ids'):
                plate_id = self.planet.plate_ids[i]
                if plate_id >= 0:
                    hue = (plate_id * 0.618033988749895) % 1.0  # Golden ratio for color distribution
                    point_colors.append(plt.cm.hsv(hue))
                else:
                    point_colors.append([0.7, 0.7, 0.7])  # Gray for unassigned
            else:
                point_colors.append([0.7, 0.7, 0.7])  # Default gray
        
        # Plot the points
        ax.scatter(point_x, point_y, c=point_colors, s=5, alpha=0.8)
        
        # Draw plate boundaries if requested and available
        if show_boundaries and hasattr(self.planet, '_tectonics') and hasattr(self.planet._tectonics, 'plate_boundaries'):
            # Fixed: Handle boolean array properly
            boundary_x = []
            boundary_y = []
            
            # Iterate through all vertices and check if they are boundaries
            for i, is_boundary in enumerate(self.planet._tectonics.plate_boundaries):
                if is_boundary:
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
                        boundary_x.append(px)
                        boundary_y.append(py)
            
            # Plot the boundary points as black dots
            if boundary_x and boundary_y:
                ax.scatter(boundary_x, boundary_y, color='black', s=1.5, alpha=0.8)
        
        # Add labels if requested
        if show_labels and hasattr(self.planet, '_tectonics'):
            plates = self.planet._tectonics.plates
            for plate in plates:
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
                    
                    plate_type = "Oceanic" if plate.get('is_oceanic', True) else "Continental"
                    ax.text(px, py, f"Plate {plate['id']}\n({plate_type})", 
                           ha='center', va='center', fontsize=8, 
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Set axis limits and labels
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invert y-axis
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title and grid
        plt.title(f"Tectonic Plates at {self.planet.age:.1f} million years", fontsize=14)
        
        # Add lat/lon grid
        self._add_lat_lon_grid(ax, width, height, projection)
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            if not show:
                plt.close(fig)
            print(f"Tectonic plate map saved to {save_path}")
            
        return save_path if save_path else None
    
    def create_realistic_world_map(self, save_path=None, width=2048, height=1024, 
                                   projection='equirectangular', show_rivers=True, 
                                   show_mountains=True, show=False):
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
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        # Create a higher-resolution map image
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create image array for the map
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Sample the planet's surface at each pixel position
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
                
                # Get elevation and biome data
                elevation = self.planet.elevation[nearest_idx]
                
                # Color based on elevation and biome if available
                if hasattr(self.planet, 'biome_ids') and self.planet.biome_ids is not None:
                    from ..core.biome import BiomeClassifier
                    biome_classifier = BiomeClassifier(self.planet)
                    color = biome_classifier.get_biome_color(self.planet.biome_ids[nearest_idx])
                    
                    # Adjust color based on elevation for more realistic look
                    if elevation < 0:
                        # Darken water with depth
                        depth_factor = max(0.5, 1 + elevation * 2)
                        color = np.array(color) * depth_factor
                    elif elevation > 0.02:
                        # Add snow to high elevations
                        snow_factor = min(1.0, (elevation - 0.02) * 10)
                        color = color * (1 - snow_factor) + np.array([1, 1, 1]) * snow_factor
                else:
                    # Fallback to elevation coloring
                    color = self._get_elevation_color(elevation)
                
                image[y, x] = color
        
        # Display the image
        ax.imshow(image, origin='upper')
        
        # Add rivers if requested
        if show_rivers and self.planet.rivers:
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
                        # Use river size based on position in the river array
                        river_size = min(3.0, 0.5 + len(river) / 20)
                        ax.plot(river_x, river_y, color='royalblue', linewidth=river_size, alpha=0.8)
        
        # Add mountain markers if requested
        if show_mountains:
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
                            mountain_size.append(20 + elev * 100)
            
            if mountain_x and mountain_y:
                ax.scatter(mountain_x, mountain_y, s=mountain_size, marker='^', 
                         color='saddlebrown', edgecolors='black', alpha=0.7)
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add lat/lon grid
        self._add_lat_lon_grid(ax, width, height, projection)
        
        # Add title
        plt.title(f"World Map at {self.planet.age:.1f} million years", fontsize=14)
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Realistic world map saved to {save_path}")
            
        return save_path if save_path else None
    
    def create_height_map(self, save_path=None, width=2048, height=1024, 
                         projection='equirectangular', grayscale=True, show=False):
        """
        Create a height map suitable for use in game engines or 3D software.
        
        Parameters:
        - save_path: Path to save the visualization
        - width: Width of the output image
        - height: Height of the output image
        - projection: Map projection type
        - grayscale: Whether to output a grayscale heightmap or colored relief
        - show: Whether to display the plot
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        # Create a figure
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create image array
        if grayscale:
            image = np.zeros((height, width), dtype=np.float32)
        else:
            image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Find elevation range for normalization
        min_elev = np.min(self.planet.elevation)
        max_elev = np.max(self.planet.elevation)
        elevation_range = max_elev - min_elev
        
        # Sample the planet's surface at each pixel position
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
                elevation = self.planet.elevation[nearest_idx]
                
                # Normalize elevation to 0-1 range
                normalized_elevation = (elevation - min_elev) / elevation_range
                
                if grayscale:
                    # Store as grayscale
                    image[y, x] = normalized_elevation
                else:
                    # Color based on elevation
                    if elevation < 0:
                        # Blue shades for water
                        depth = normalized_elevation * 0.5  # Scale for better visibility
                        image[y, x] = [0, 0.2 + depth * 0.3, 0.5 + depth * 0.5]
                    else:
                        # Green to brown to white for land
                        land_height = (elevation - 0) / max(0.001, max_elev)
                        if land_height < 0.2:
                            image[y, x] = [0.2, 0.6, 0.2]  # Low land, green
                        elif land_height < 0.5:
                            image[y, x] = [0.6, 0.4, 0.2]  # Hills, brown
                        else:
                            # Gradual transition to white for mountains
                            mountain_ratio = min(1, (land_height - 0.5) * 2)
                            image[y, x] = [0.6 + 0.4*mountain_ratio, 
                                          0.4 + 0.6*mountain_ratio, 
                                          0.2 + 0.8*mountain_ratio]
        
        # Display the image
        if grayscale:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1, origin='upper')
        else:
            ax.imshow(image, origin='upper')
        
        # Add lat/lon grid
        self._add_lat_lon_grid(ax, width, height, projection)
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title and colorbar
        plt.title("Elevation Map", fontsize=14)
        if grayscale:
            plt.colorbar(shrink=0.7, label="Normalized Elevation")
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Height map saved to {save_path}")
            
        return save_path if save_path else None
    
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