import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.colors as mcolors
import math
import os

class MapVisualizer:
    """
    Advanced visualization module for AeonTerra planet generation system.
    Creates high-quality 2D map visualizations in various styles.
    """
    
    def __init__(self, planet):
        """
        Initialize the map visualizer.
        
        Parameters:
        - planet: A Planet object with elevation, temperature, plates, etc.
        """
        self.planet = planet
        self.default_resolution = (2048, 1024)
        
    def create_map_array(self, width=2048, height=1024, projection='mercator'):
        """
        Create a base map array by projecting 3D vertices onto a 2D grid.
        
        Parameters:
        - width, height: Dimensions of the output map in pixels
        - projection: Map projection type ('mercator', 'equirectangular', etc.)
        
        Returns:
        - grid_indices: For each pixel, the indices of vertices that map to it
        - mask: Boolean array indicating valid pixels
        """
        # Initialize arrays
        grid_indices = [[] for _ in range(height * width)]
        pixel_count = np.zeros((height, width), dtype=np.int32)
        
        # For each vertex, determine its 2D projection
        for i, vertex in enumerate(self.planet.grid.vertices):
            # Convert to lat/lon
            x, y, z = vertex
            norm = np.linalg.norm(vertex)
            lat = math.asin(z / norm) * 180 / math.pi
            lon = math.atan2(y, x) * 180 / math.pi
            
            # Project to 2D
            if projection == 'mercator':
                px, py = self._lat_lon_to_mercator(lat, lon, width, height)
            else:  # equirectangular
                px = (lon + 180) * width / 360
                py = (90 - lat) * height / 180
            
            # Ensure coordinates are within bounds
            px_int = int(min(max(0, px), width - 1))
            py_int = int(min(max(0, py), height - 1))
            
            # Add this vertex to the grid cell
            idx = py_int * width + px_int
            grid_indices[idx].append(i)
            pixel_count[py_int, px_int] += 1
        
        # Create mask for valid pixels
        mask = pixel_count > 0
        
        return grid_indices, mask
    
    def visualize_terrain(self, save_path=None, width=2048, height=1024, 
                         projection='mercator', show=False, satellite_style=False):
        """
        Create a terrain visualization with land and ocean.
        
        Parameters:
        - save_path: Path to save the visualization
        - width, height: Dimensions of the output map
        - projection: Map projection type
        - show: Whether to display the map
        - satellite_style: If True, uses a realistic satellite-like style
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        # Create base map grid and indices
        grid_indices, mask = self.create_map_array(width, height, projection)
        
        # Initialize pixel data arrays
        elevation_data = np.zeros((height, width), dtype=np.float32)
        
        # Additional arrays for satellite style
        temperature_data = None
        precipitation_data = None
        if satellite_style and self.planet.temperature is not None:
            temperature_data = np.zeros((height, width), dtype=np.float32)
        if satellite_style and self.planet.precipitation is not None:
            precipitation_data = np.zeros((height, width), dtype=np.float32)
        
        # Fill arrays with data
        for y in range(height):
            for x in range(width):
                indices = grid_indices[y * width + x]
                if not indices:
                    continue
                
                # Average elevation for this cell
                elevations = [self.planet.elevation[i] for i in indices]
                elevation_data[y, x] = sum(elevations) / len(elevations)
                
                if temperature_data is not None:
                    temps = [self.planet.temperature[i] for i in indices]
                    temperature_data[y, x] = sum(temps) / len(temps)
                
                if precipitation_data is not None:
                    precips = [self.planet.precipitation[i] for i in indices]
                    precipitation_data[y, x] = sum(precips) / len(precips)
        
        # Create figure
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        if satellite_style:
            # Create satellite-style map
            rgb_data = self._create_satellite_style_map(elevation_data, temperature_data, precipitation_data)
            plt.imshow(rgb_data, extent=[0, width, height, 0])
            plt.title(f'Planetary Terrain (Age: {self.planet.age:.1f} million years)')
        else:
            # Create terrain-colored map
            terrain_cmap = self._create_terrain_colormap()
            
            # Find elevation extremes for normalization
            min_elev = np.min(elevation_data[mask])
            max_elev = np.max(elevation_data[mask])
            norm = Normalize(vmin=min_elev, vmax=max_elev)
            
            # Display the map
            plt.imshow(elevation_data, cmap=terrain_cmap, norm=norm, 
                      extent=[0, width, height, 0])
            plt.colorbar(label='Elevation (km)')
            plt.title(f'Elevation Map (Age: {self.planet.age:.1f} million years)')
        
        # Add rivers if available
        if hasattr(self.planet, 'rivers') and self.planet.rivers:
            self._add_rivers_to_map(projection, width, height)
        
        # Finalize the plot
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            if not show:
                plt.close()
            print(f"Terrain visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_plates(self, save_path=None, width=2048, height=1024, 
                        projection='mercator', show=False, show_boundaries=True,
                        show_arrows=True):
        """
        Create a tectonic plate visualization similar to the example.
        
        Parameters:
        - save_path: Path to save the visualization
        - width, height: Dimensions of the output map
        - projection: Map projection type
        - show: Whether to display the map
        - show_boundaries: Whether to highlight plate boundaries
        - show_arrows: Whether to show plate movement arrows
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        # Create base map grid and indices
        grid_indices, mask = self.create_map_array(width, height, projection)
        
        # Initialize pixel data array
        plate_data = np.zeros((height, width), dtype=np.int32)
        boundary_mask = np.zeros((height, width), dtype=bool)
        
        # Generate a color for each plate
        num_plates = max(plate.get('id', 0) for plate in self.planet.plates) + 1
        plate_colors = [plt.cm.hsv((i * 0.618033988749895) % 1.0) for i in range(num_plates)]
        
        # Create RGBA image
        rgb_data = np.zeros((height, width, 4))
        
        # Fill plate data array
        for y in range(height):
            for x in range(width):
                indices = grid_indices[y * width + x]
                if not indices:
                    continue
                
                # Get majority plate for this pixel
                plate_counts = {}
                for i in indices:
                    plate_id = self.planet.plate_ids[i]
                    if plate_id >= 0 and plate_id < num_plates:
                        plate_counts[plate_id] = plate_counts.get(plate_id, 0) + 1
                
                if plate_counts:
                    majority_plate = max(plate_counts.items(), key=lambda x: x[1])[0]
                    plate_data[y, x] = majority_plate
                    
                    # Assign plate color
                    rgb_data[y, x] = plate_colors[majority_plate]
                else:
                    # No valid plate - make it transparent
                    rgb_data[y, x] = [0, 0, 0, 0]
                
                # Check for boundary
                if show_boundaries:
                    is_boundary = False
                    for i in indices:
                        if hasattr(self.planet, 'plate_boundaries') and self.planet.plate_boundaries[i]:
                            is_boundary = True
                            break
                    boundary_mask[y, x] = is_boundary
        
        # Create figure
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Display plate map
        plt.imshow(rgb_data, extent=[0, width, height, 0])
        
        # Add plate boundaries
        if show_boundaries:
            boundary_overlay = np.zeros((height, width, 4))
            boundary_overlay[boundary_mask] = [0, 0, 0, 1]  # Black, fully opaque
            plt.imshow(boundary_overlay, extent=[0, width, height, 0], alpha=0.5)
        
        # Add plate labels and arrows
        if show_arrows:
            for plate in self.planet.plates:
                # Convert center to lat/lon
                center = plate['center']
                x, y, z = center
                lat = math.asin(z / np.linalg.norm(center)) * 180 / math.pi
                lon = math.atan2(y, x) * 180 / math.pi
                
                # Project to pixel coordinates
                if projection == 'mercator':
                    px, py = self._lat_lon_to_mercator(lat, lon, width, height)
                else:
                    px = (lon + 180) * width / 360
                    py = (90 - lat) * height / 180
                
                # Skip if outside the map
                if px < 0 or px >= width or py < 0 or py >= height:
                    continue
                
                # Calculate arrow
                velocity = plate.get('velocity', [0, 0, 0])
                vel_mag = np.linalg.norm(velocity)
                if vel_mag > 1e-6:
                    # Project velocity to 2D (simplified)
                    vel_scaled = velocity / vel_mag * 50  # Scale for visibility
                    
                    # Draw arrow
                    plt.arrow(px, py, vel_scaled[0], -vel_scaled[1],
                             width=2, head_width=10, head_length=15,
                             color=plate_colors[plate['id']], length_includes_head=True)
                
                # Add plate label
                plt.text(px, py, f"{plate['id']}: {plate.get('name', '')}", 
                        fontsize=8, ha='center', va='center',
                        color='white', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5))
        
        plt.title(f'Tectonic Plates (Age: {self.planet.age:.1f} million years)')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            if not show:
                plt.close()
            print(f"Plate visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_heightmap(self, save_path=None, width=2048, height=1024, 
                           projection='mercator', show=False, shaded_relief=True):
        """
        Create a grayscale heightmap visualization.
        
        Parameters:
        - save_path: Path to save the visualization
        - width, height: Dimensions of the output map
        - projection: Map projection type
        - show: Whether to display the map
        - shaded_relief: Whether to add shaded relief effect
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        # Create base map grid and indices
        grid_indices, mask = self.create_map_array(width, height, projection)
        
        # Initialize heightmap data array
        heightmap = np.zeros((height, width), dtype=np.float32)
        
        # Fill heightmap data
        for y in range(height):
            for x in range(width):
                indices = grid_indices[y * width + x]
                if not indices:
                    continue
                
                # Average elevation for this cell
                elevations = [self.planet.elevation[i] for i in indices]
                heightmap[y, x] = sum(elevations) / len(elevations)
        
        # Create figure
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Find elevation extremes for normalization
        min_elev = np.min(heightmap[mask])
        max_elev = np.max(heightmap[mask])
        
        # Add shaded relief if requested
        if shaded_relief:
            # Calculate shaded relief
            dx, dy = np.gradient(heightmap)
            slope = np.pi/2. - np.arctan(np.sqrt(dx*dx + dy*dy))
            aspect = np.arctan2(-dx, dy)
            altitude = np.pi/4.
            azimuth = np.pi/2.
            
            shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
            
            # Combine heightmap with shaded relief
            rgb = np.zeros((height, width, 3))
            for i in range(3):
                rgb[:,:,i] = Normalize(vmin=min_elev, vmax=max_elev)(heightmap)
            rgb = rgb * (0.7 + 0.3 * shaded[:,:,np.newaxis])
            rgb = np.clip(rgb, 0, 1)
            
            # Display grayscale heightmap with shaded relief
            plt.imshow(rgb, cmap='gray', extent=[0, width, height, 0])
        else:
            # Display simple grayscale heightmap
            plt.imshow(heightmap, cmap='gray', vmin=min_elev, vmax=max_elev, 
                      extent=[0, width, height, 0])
        
        plt.colorbar(label='Elevation (km)')
        plt.title(f'Heightmap (Age: {self.planet.age:.1f} million years)')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            if not show:
                plt.close()
            print(f"Heightmap visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_precipitation(self, save_path=None, width=2048, height=1024,
                              projection='mercator', show=False):
        """
        Create a visualization of precipitation patterns.
        
        Parameters:
        - save_path: Path to save the visualization
        - width, height: Dimensions of the output map
        - projection: Map projection type
        - show: Whether to display the map
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        if self.planet.precipitation is None:
            print("No precipitation data available. Run climate simulation first.")
            return None
        
        # Create base map grid and indices
        grid_indices, mask = self.create_map_array(width, height, projection)
        
        # Initialize data array
        precip_data = np.zeros((height, width), dtype=np.float32)
        
        # Fill precipitation data
        for y in range(height):
            for x in range(width):
                indices = grid_indices[y * width + x]
                if not indices:
                    continue
                
                # Average precipitation for this cell
                precips = [self.planet.precipitation[i] for i in indices]
                precip_data[y, x] = sum(precips) / len(precips)
        
        # Create figure
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Setup colormap for precipitation
        precip_cmap = plt.cm.Blues
        
        # Display precipitation map
        im = plt.imshow(precip_data, cmap=precip_cmap, extent=[0, width, height, 0])
        plt.colorbar(im, label='Precipitation (mm/year)')
        plt.title(f'Precipitation Map (Age: {self.planet.age:.1f} million years)')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            if not show:
                plt.close()
            print(f"Precipitation visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_temperature(self, save_path=None, width=2048, height=1024,
                             projection='mercator', show=False):
        """
        Create a visualization of temperature patterns.
        
        Parameters:
        - save_path: Path to save the visualization
        - width, height: Dimensions of the output map
        - projection: Map projection type
        - show: Whether to display the map
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        if self.planet.temperature is None:
            print("No temperature data available. Run climate simulation first.")
            return None
        
        # Create base map grid and indices
        grid_indices, mask = self.create_map_array(width, height, projection)
        
        # Initialize data array
        temp_data = np.zeros((height, width), dtype=np.float32)
        
        # Fill temperature data
        for y in range(height):
            for x in range(width):
                indices = grid_indices[y * width + x]
                if not indices:
                    continue
                
                # Average temperature for this cell
                temps = [self.planet.temperature[i] for i in indices]
                temp_data[y, x] = sum(temps) / len(temps)
        
        # Create figure
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Setup colormap for temperature
        temp_cmap = plt.cm.coolwarm
        
        # Display temperature map
        im = plt.imshow(temp_data, cmap=temp_cmap, extent=[0, width, height, 0])
        plt.colorbar(im, label='Temperature (°C)')
        plt.title(f'Temperature Map (Age: {self.planet.age:.1f} million years)')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            if not show:
                plt.close()
            print(f"Temperature visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def visualize_biomes(self, save_path=None, width=2048, height=1024,
                        projection='mercator', show=False):
        """
        Create a visualization of biome distribution.
        
        Parameters:
        - save_path: Path to save the visualization
        - width, height: Dimensions of the output map
        - projection: Map projection type
        - show: Whether to display the map
        
        Returns:
        - Path to the saved visualization if save_path is provided
        """
        if self.planet.biome_ids is None:
            print("No biome data available. Run biome classification first.")
            return None
        
        # Create base map grid and indices
        grid_indices, mask = self.create_map_array(width, height, projection)
        
        # Import biome classifier for color mapping
        from planet_sim.core.biome import BiomeClassifier
        biome_classifier = BiomeClassifier(self.planet)
        
        # Initialize RGB data array
        rgb_data = np.zeros((height, width, 3), dtype=np.float32)
        
        # Fill biome data
        for y in range(height):
            for x in range(width):
                indices = grid_indices[y * width + x]
                if not indices:
                    continue
                
                # Get most common biome for this cell
                biome_counts = {}
                for i in indices:
                    biome_id = self.planet.biome_ids[i]
                    biome_counts[biome_id] = biome_counts.get(biome_id, 0) + 1
                
                if biome_counts:
                    # Get most common biome
                    majority_biome = max(biome_counts.items(), key=lambda x: x[1])[0]
                    # Get color for this biome
                    rgb_data[y, x] = biome_classifier.get_biome_color(majority_biome)
        
        # Create figure
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Display biome map
        plt.imshow(rgb_data, extent=[0, width, height, 0])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = []
        for i, biome_name in enumerate(biome_classifier.biome_types):
            if i in np.unique(self.planet.biome_ids):
                color = biome_classifier.get_biome_color(i)
                legend_elements.append(Patch(facecolor=color, label=biome_name))
        
        plt.legend(handles=legend_elements, loc='lower right', fontsize='small')
        plt.title(f'Biome Distribution (Age: {self.planet.age:.1f} million years)')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            if not show:
                plt.close()
            print(f"Biome visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return save_path if save_path else None
    
    def _create_terrain_colormap(self):
        """Create a custom colormap for elevation with land-sea distinction"""
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
        
        # Create a custom LinearSegmentedColormap
        return LinearSegmentedColormap.from_list("elevation", cmap_data, N=256)
    
    def _create_satellite_style_map(self, elevation_data, temperature_data=None, precipitation_data=None):
        """
        Create a satellite-style RGB map based on elevation and optional climate data.
        
        Parameters:
        - elevation_data: 2D array of elevation values
        - temperature_data: Optional 2D array of temperature values
        - precipitation_data: Optional 2D array of precipitation values
        
        Returns:
        - rgb_data: 3D array (height, width, 3) with RGB values
        """
        height, width = elevation_data.shape
        rgb_data = np.zeros((height, width, 3))
        
        # Normalize elevation data
        elevation_min = np.min(elevation_data)
        elevation_max = np.max(elevation_data)
        elevation_range = elevation_max - elevation_min
        
        # Create mask for land and water
        land_mask = elevation_data >= 0
        water_mask = elevation_data < 0
        
        # Initialize with default colors
        # Water (blue)
        rgb_data[water_mask] = [0.0, 0.1, 0.5]  # Deep blue
        
        # Land (green)
        rgb_data[land_mask] = [0.1, 0.4, 0.1]  # Default green
        
        # Adjust water depth - deeper water is darker blue
        if np.any(water_mask):
            water_elev = elevation_data[water_mask]
            water_min = np.min(water_elev)
            if water_min < 0:  # Prevent division by zero
                depth_factor = (water_elev - water_min) / (-water_min)
                
                # Shallow water is lighter blue
                rgb_data[water_mask, 0] = 0.0  # No red
                rgb_data[water_mask, 1] = 0.1 + 0.3 * depth_factor  # More green for shallow
                rgb_data[water_mask, 2] = 0.5 + 0.3 * depth_factor  # More blue for shallow
        
        # Adjust land elevation - higher is lighter (snow-capped)
        if np.any(land_mask):
            land_elev = elevation_data[land_mask]
            if elevation_range > 0:
                height_factor = (land_elev - 0) / (elevation_max - 0)
                
                # Base green land with elevation effects
                rgb_data[land_mask, 0] = 0.1 + 0.4 * height_factor  # More red for higher
                rgb_data[land_mask, 1] = 0.4 - 0.1 * height_factor  # Less green for higher
                rgb_data[land_mask, 2] = 0.1 + 0.4 * height_factor  # More blue for higher
                
                # Snow caps on very high mountains
                snow_mask = land_mask & (height_factor > 0.8)
                if np.any(snow_mask):
                    snow_factor = (height_factor[height_factor > 0.8] - 0.8) / 0.2
                    rgb_data[snow_mask] = [
                        0.5 + 0.5 * snow_factor,  # Increase red
                        0.5 + 0.5 * snow_factor,  # Increase green
                        0.5 + 0.5 * snow_factor   # Increase blue
                    ]
        
        # Apply temperature effects if available
        if temperature_data is not None:
            # Adjust land colors based on temperature (deserts, etc.)
            temp_land_mask = land_mask & (elevation_data / elevation_max < 0.7)  # Exclude high mountains
            if np.any(temp_land_mask):
                # Normalize temperature
                temp_min = np.min(temperature_data[temp_land_mask])
                temp_max = np.max(temperature_data[temp_land_mask])
                temp_range = temp_max - temp_min
                
                if temp_range > 0:
                    temp_factor = (temperature_data[temp_land_mask] - temp_min) / temp_range
                    
                    # Hot areas are more yellow/brown (deserts)
                    rgb_data[temp_land_mask, 0] += 0.3 * temp_factor  # More red for hot
                    rgb_data[temp_land_mask, 1] += 0.1 * temp_factor  # Slightly more green for hot
                    rgb_data[temp_land_mask, 2] -= 0.1 * temp_factor  # Less blue for hot
        
        # Apply precipitation effects if available
        if precipitation_data is not None:
            # Adjust land colors based on precipitation (green vs. brown)
            precip_land_mask = land_mask & (elevation_data / elevation_max < 0.7)  # Exclude high mountains
            if np.any(precip_land_mask):
                # Normalize precipitation
                precip_min = np.min(precipitation_data[precip_land_mask])
                precip_max = np.max(precipitation_data[precip_land_mask])
                precip_range = precip_max - precip_min
                
                if precip_range > 0:
                    precip_factor = (precipitation_data[precip_land_mask] - precip_min) / precip_range
                    
                    # Wet areas are greener
                    rgb_data[precip_land_mask, 0] -= 0.1 * precip_factor  # Less red for wet
                    rgb_data[precip_land_mask, 1] += 0.2 * precip_factor  # More green for wet
                    rgb_data[precip_land_mask, 2] -= 0.1 * precip_factor  # Less blue for wet
        
        # Clip values to valid RGB range
        return np.clip(rgb_data, 0, 1)
    
    def _add_rivers_to_map(self, projection, width, height):
        """Add rivers to the current map visualization"""
        if not self.planet.rivers:
            return
            
        for river in self.planet.rivers:
            if len(river) < 2:
                continue
                
            river_x = []
            river_y = []
            
            for idx in river:
                vertex = self.planet.grid.vertices[idx]
                x, y, z = vertex
                lat = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
                lon = np.arctan2(y, x) * 180 / np.pi
                
                if projection == 'mercator':
                    px, py = self._lat_lon_to_mercator(lat, lon, width, height)
                else:
                    px = (lon + 180) * width / 360
                    py = (90 - lat) * height / 180
                
                if 0 <= px < width and 0 <= py < height:
                    river_x.append(px)
                    river_y.append(py)
            
            if river_x and river_y:
                plt.plot(river_x, river_y, color='#99CCFF', linewidth=1, alpha=0.7)
    
    def _lat_lon_to_mercator(self, lat, lon, width, height):
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
    
    def export_heightmap(self, save_path, width=2048, height=1024, 
                        projection='mercator', format='png', bit_depth=16,
                        region=None):
        """
        Export a heightmap as an image file.
        
        Parameters:
        - save_path: Path to save the heightmap
        - width, height: Dimensions of the output image
        - projection: Map projection to use
        - format: Output format ('png', 'raw', 'tiff')
        - bit_depth: Bit depth (8 or 16)
        - region: Optional region to export (lat_min, lon_min, lat_max, lon_max)
        
        Returns:
        - Path to the exported heightmap
        """
        from PIL import Image
        
        # Create base map grid and indices
        grid_indices, mask = self.create_map_array(width, height, projection)
        
        # Initialize heightmap data array
        heightmap = np.zeros((height, width), dtype=np.float32)
        
        # Fill heightmap data
        for y in range(height):
            for x in range(width):
                indices = grid_indices[y * width + x]
                if not indices:
                    continue
                
                # Average elevation for this cell
                elevations = [self.planet.elevation[i] for i in indices]
                heightmap[y, x] = sum(elevations) / len(elevations)
        
        # Find min and max elevation for scaling
        min_elev = np.min(heightmap[mask])
        max_elev = np.max(heightmap[mask])
        
        # Export based on format
        if format == 'png':
            # Scale to 0-255 or 0-65535 based on bit depth
            if bit_depth == 8:
                # 8-bit PNG
                scaled_data = np.zeros((height, width), dtype=np.uint8)
                scaled_data[mask] = ((heightmap[mask] - min_elev) / (max_elev - min_elev) * 255).astype(np.uint8)
                img = Image.fromarray(scaled_data, mode='L')
            else:
                # 16-bit PNG
                scaled_data = np.zeros((height, width), dtype=np.uint16)
                scaled_data[mask] = ((heightmap[mask] - min_elev) / (max_elev - min_elev) * 65535).astype(np.uint16)
                img = Image.fromarray(scaled_data, mode='I;16')
            
            # Save the image
            img.save(save_path)
        
        elif format == 'raw':
            # Scale to 0-255 or 0-65535 based on bit depth
            if bit_depth == 8:
                # 8-bit RAW
                scaled_data = np.zeros((height, width), dtype=np.uint8)
                scaled_data[mask] = ((heightmap[mask] - min_elev) / (max_elev - min_elev) * 255).astype(np.uint8)
            else:
                # 16-bit RAW
                scaled_data = np.zeros((height, width), dtype=np.uint16)
                scaled_data[mask] = ((heightmap[mask] - min_elev) / (max_elev - min_elev) * 65535).astype(np.uint16)
            
            # Save raw data
            scaled_data.tofile(save_path)
        
        elif format == 'tiff':
            # Scale to 0-255 or 0-65535 based on bit depth
            if bit_depth == 8:
                # 8-bit TIFF
                scaled_data = np.zeros((height, width), dtype=np.uint8)
                scaled_data[mask] = ((heightmap[mask] - min_elev) / (max_elev - min_elev) * 255).astype(np.uint8)
                img = Image.fromarray(scaled_data, mode='L')
            else:
                # 16-bit TIFF
                scaled_data = np.zeros((height, width), dtype=np.uint16)
                scaled_data[mask] = ((heightmap[mask] - min_elev) / (max_elev - min_elev) * 65535).astype(np.uint16)
                img = Image.fromarray(scaled_data, mode='I;16')
            
            # Save TIFF
            img.save(save_path, format='TIFF')
        
        print(f"Heightmap exported to {save_path}")
        return save_path