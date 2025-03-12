import numpy as np
from ..utils.sphere_grid import SphericalGrid

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