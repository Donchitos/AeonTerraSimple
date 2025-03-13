import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import json
from PIL import Image
import struct
import math

class ExportManager:
    """
    Enhanced class for exporting planet data to various formats.
    Handles heightmaps, climate data, biome maps, and game engine exports.
    """
    
    def __init__(self, planet):
        """
        Initialize the export manager.
        
        Parameters:
        - planet: A Planet object
        """
        self.planet = planet
        self.export_history = []
    
    def export_heightmap(self, output_path, width=2048, height=1024, 
                         region=None, format='png', bit_depth=16,
                         add_metadata=True, scaling_factor=1.0):
        """
        Export a heightmap of the planet's terrain.
        
        Parameters:
        - output_path: Path to save the heightmap
        - width, height: Dimensions of the output image
        - region: Optional tuple (lat_min, lon_min, lat_max, lon_max) for region extract
        - format: Output format ('png', 'raw', 'tiff')
        - bit_depth: Bit depth for the heightmap (8 or 16)
        - add_metadata: Whether to add a JSON metadata file
        - scaling_factor: Vertical exaggeration factor
        
        Returns:
        - Path to the exported heightmap
        """
        # Determine projection type
        if region:
            projection = 'mercator'
            lat_min, lon_min, lat_max, lon_max = region
        else:
            projection = 'equirectangular'
            lat_min, lon_min, lat_max, lon_max = -90, -180, 90, 180
        
        # Create the heightmap grid
        heightmap = self._generate_heightmap_grid(width, height, region, scaling_factor)
        
        # Record min/max values for metadata
        min_elevation = self.planet.elevation.min()
        max_elevation = self.planet.elevation.max()
        
        # Export based on format
        if format == 'raw':
            self._export_raw_heightmap(heightmap, output_path, bit_depth)
        elif format == 'tiff':
            self._export_tiff_heightmap(heightmap, output_path, bit_depth)
        else:  # Default to PNG
            self._export_png_heightmap(heightmap, output_path, bit_depth)
        
        # Add metadata file if requested
        if add_metadata:
            metadata = {
                "width": width,
                "height": height,
                "min_elevation_km": float(min_elevation),
                "max_elevation_km": float(max_elevation),
                "scale_factor": float(scaling_factor),
                "projection": projection,
                "bit_depth": bit_depth,
                "region": {
                    "lat_min": lat_min, 
                    "lon_min": lon_min, 
                    "lat_max": lat_max, 
                    "lon_max": lon_max
                } if region else None
            }
            
            metadata_path = os.path.splitext(output_path)[0] + '.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Record this export
        self.export_history.append({
            "type": "heightmap",
            "path": output_path,
            "dimensions": [width, height],
            "format": format,
            "projection": projection,
            "region": region
        })
        
        print(f"Heightmap exported to {output_path}")
        return output_path
    
    def _generate_heightmap_grid(self, width, height, region=None, scaling_factor=1.0):
        """Generate a heightmap grid from the planet's elevation data"""
        # Create empty grid
        grid = np.zeros((height, width), dtype=np.float32)
        
        # Calculate coordinates for each pixel
        for y in range(height):
            for x in range(width):
                # Convert grid coordinates to lat/lon
                if region:
                    lat_min, lon_min, lat_max, lon_max = region
                    lat = lat_max - (y / height) * (lat_max - lat_min)
                    lon = lon_min + (x / width) * (lon_max - lon_min)
                else:
                    lat = 90 - (y / height) * 180
                    lon = -180 + (x / width) * 360
                
                # Convert lat/lon to 3D coordinates
                lat_rad = math.radians(lat)
                lon_rad = math.radians(lon)
                
                x_coord = math.cos(lat_rad) * math.cos(lon_rad)
                y_coord = math.cos(lat_rad) * math.sin(lon_rad)
                z_coord = math.sin(lat_rad)
                
                # Find closest vertex
                coords = np.array([x_coord, y_coord, z_coord])
                distances = np.sum((self.planet.grid.vertices - coords) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                
                # Get elevation
                elevation = self.planet.elevation[closest_idx]
                
                # Apply scaling
                if scaling_factor != 1.0:
                    elevation *= scaling_factor
                
                grid[y, x] = elevation
        
        return grid
    
    def _export_png_heightmap(self, heightmap, output_path, bit_depth=16):
        """Export heightmap as a PNG image"""
        # Normalize to 0-1 range for image
        min_val = np.min(heightmap)
        max_val = np.max(heightmap)
        normalized = (heightmap - min_val) / (max_val - min_val)
        
        # Create image based on bit depth
        if bit_depth == 16:
            # 16-bit grayscale
            img_data = (normalized * 65535).astype(np.uint16)
            img = Image.fromarray(img_data, mode='I;16')
        else:
            # 8-bit grayscale
            img_data = (normalized * 255).astype(np.uint8)
            img = Image.fromarray(img_data, mode='L')
        
        # Save image
        img.save(output_path)
    
    def _export_raw_heightmap(self, heightmap, output_path, bit_depth=16):
        """Export heightmap as a RAW binary file (common for game engines)"""
        # Normalize to 0-1 range
        min_val = np.min(heightmap)
        max_val = np.max(heightmap)
        normalized = (heightmap - min_val) / (max_val - min_val)
        
        # Prepare binary data based on bit depth
        if bit_depth == 16:
            # 16-bit unsigned integers (0-65535)
            img_data = (normalized * 65535).astype(np.uint16)
            # Make sure we use little-endian format
            img_data = img_data.astype('<u2')
            mode = 'wb'
        else:
            # 8-bit unsigned integers (0-255)
            img_data = (normalized * 255).astype(np.uint8)
            mode = 'wb'
        
        # Write binary data
        with open(output_path, mode) as f:
            img_data.tofile(f)
    
    def _export_tiff_heightmap(self, heightmap, output_path, bit_depth=16):
        """Export heightmap as a TIFF image (for GIS applications)"""
        # Normalize to 0-1 range
        min_val = np.min(heightmap)
        max_val = np.max(heightmap)
        normalized = (heightmap - min_val) / (max_val - min_val)
        
        # Create image based on bit depth
        if bit_depth == 16:
            # 16-bit grayscale
            img_data = (normalized * 65535).astype(np.uint16)
            img = Image.fromarray(img_data, mode='I;16')
        else:
            # 8-bit grayscale
            img_data = (normalized * 255).astype(np.uint8)
            img = Image.fromarray(img_data, mode='L')
        
        # Save image as TIFF
        img.save(output_path, format='TIFF')
    
    def export_climate_map(self, output_path, data_type='temperature', 
                           width=2048, height=1024, region=None, 
                           colormap='viridis', bit_depth=8):
        """
        Export a climate data map (temperature, precipitation, etc).
        
        Parameters:
        - output_path: Path to save the map
        - data_type: Type of data to export ('temperature', 'precipitation')
        - width, height: Dimensions of the output image
        - region: Optional tuple (lat_min, lon_min, lat_max, lon_max) for region extract
        - colormap: Matplotlib colormap to use
        - bit_depth: Bit depth (8 or 16)
        
        Returns:
        - Path to the exported map
        """
        # Check if climate data exists
        if data_type == 'temperature' and self.planet.temperature is None:
            raise ValueError("Temperature data not available. Run climate simulation first.")
        elif data_type == 'precipitation' and self.planet.precipitation is None:
            raise ValueError("Precipitation data not available. Run climate simulation first.")
        
        # Get data based on type
        if data_type == 'temperature':
            data = self.planet.temperature
            title = "Temperature Map (°C)"
            if colormap == 'viridis':  # Override with better map for temperature
                colormap = 'coolwarm'
        elif data_type == 'precipitation':
            data = self.planet.precipitation
            title = "Precipitation Map (mm/year)"
            if colormap == 'viridis':  # Override with better map for precipitation
                colormap = 'Blues'
        
        # Create the data grid
        grid = self._generate_data_grid(data, width, height, region)
        
        # Export as an image
        self._export_data_image(grid, output_path, colormap, title, bit_depth)
        
        # Record this export
        self.export_history.append({
            "type": f"climate_{data_type}",
            "path": output_path,
            "dimensions": [width, height],
            "region": region
        })
        
        print(f"{data_type.capitalize()} map exported to {output_path}")
        return output_path
    
    def export_biome_map(self, output_path, width=2048, height=1024, 
                          region=None, custom_colors=None):
        """
        Export a biome map.
        
        Parameters:
        - output_path: Path to save the map
        - width, height: Dimensions of the output image
        - region: Optional tuple (lat_min, lon_min, lat_max, lon_max) for region extract
        - custom_colors: Optional dict mapping biome IDs to RGB colors
        
        Returns:
        - Path to the exported map
        """
        # Check if biome data exists
        if self.planet.biome_ids is None:
            raise ValueError("Biome data not available. Run biome classification first.")
        
        # Create the biome grid
        grid = self._generate_data_grid(self.planet.biome_ids, width, height, region)
        
        # Create biome classifier to get colors
        from planet_sim.core.biome import BiomeClassifier
        biome_classifier = BiomeClassifier(self.planet)
        
        # Create an RGB image with biome colors
        rgb_img = np.zeros((height, width, 3))
        
        for y in range(height):
            for x in range(width):
                biome_id = int(grid[y, x])
                
                if custom_colors and biome_id in custom_colors:
                    rgb_img[y, x] = custom_colors[biome_id]
                else:
                    rgb_img[y, x] = biome_classifier.get_biome_color(biome_id)
        
        # Save the image
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.imshow(rgb_img)
        plt.title("Biome Map")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Record this export
        self.export_history.append({
            "type": "biome_map",
            "path": output_path,
            "dimensions": [width, height],
            "region": region
        })
        
        print(f"Biome map exported to {output_path}")
        return output_path
    
    def _generate_data_grid(self, data, width, height, region=None):
        """Generate a data grid from any vertex-based planet data"""
        # Create empty grid
        grid = np.zeros((height, width), dtype=np.float32)
        
        # Calculate coordinates for each pixel
        for y in range(height):
            for x in range(width):
                # Convert grid coordinates to lat/lon
                if region:
                    lat_min, lon_min, lat_max, lon_max = region
                    lat = lat_max - (y / height) * (lat_max - lat_min)
                    lon = lon_min + (x / width) * (lon_max - lon_min)
                else:
                    lat = 90 - (y / height) * 180
                    lon = -180 + (x / width) * 360
                
                # Convert lat/lon to 3D coordinates
                lat_rad = math.radians(lat)
                lon_rad = math.radians(lon)
                
                x_coord = math.cos(lat_rad) * math.cos(lon_rad)
                y_coord = math.cos(lat_rad) * math.sin(lon_rad)
                z_coord = math.sin(lat_rad)
                
                # Find closest vertex
                coords = np.array([x_coord, y_coord, z_coord])
                distances = np.sum((self.planet.grid.vertices - coords) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                
                # Get data value
                grid[y, x] = data[closest_idx]
        
        return grid
    
    def _export_data_image(self, grid, output_path, colormap, title, bit_depth=8):
        """Export a data grid as a colored image using the specified colormap"""
        # Create figure
        plt.figure(figsize=(grid.shape[1]/100, grid.shape[0]/100), dpi=100)
        
        # Plot the data with the specified colormap
        img = plt.imshow(grid, cmap=colormap)
        plt.colorbar(img, fraction=0.025, pad=0.04)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close()
    
    def export_for_game_engine(self, output_dir, engine='unreal', region=None, 
                              resolution=4097, include_materials=True):
        """
        Export planet data formatted for specific game engines.
        
        Parameters:
        - output_dir: Directory to save export files
        - engine: Target engine ('unreal' or 'unity')
        - region: Optional tuple (lat_min, lon_min, lat_max, lon_max) for region extract
        - resolution: Resolution of the heightmap (Unreal requires 2^n+1)
        - include_materials: Whether to include material maps
        
        Returns:
        - Path to the export directory
        """
        # Create export directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export heightmap
        if engine == 'unreal':
            # Unreal prefers RAW heightmap
            heightmap_path = os.path.join(output_dir, "heightmap.r16")
            self.export_heightmap(
                heightmap_path, 
                width=resolution, 
                height=resolution, 
                region=region, 
                format='raw', 
                bit_depth=16
            )
            
            # Layer weight maps for different materials
            if include_materials and self.planet.biome_ids is not None:
                self._export_unreal_layer_maps(output_dir, resolution, region)
                
                # Export material setup JSON
                material_json_path = os.path.join(output_dir, "material_setup.json")
                material_setup = self._generate_unreal_material_setup()
                
                with open(material_json_path, 'w') as f:
                    json.dump(material_setup, f, indent=2)
        
        elif engine == 'unity':
            # Unity uses RAW heightmap and splatmaps
            heightmap_path = os.path.join(output_dir, "heightmap.raw")
            self.export_heightmap(
                heightmap_path, 
                width=resolution, 
                height=resolution, 
                region=region, 
                format='raw', 
                bit_depth=16
            )
            
            # Export splatmap if biomes exist
            if include_materials and self.planet.biome_ids is not None:
                self._export_unity_splatmap(output_dir, resolution, region)
                
                # Terrain settings JSON
                terrain_json_path = os.path.join(output_dir, "terrain_settings.json")
                terrain_settings = {
                    "heightmap_resolution": resolution,
                    "width": 10000,
                    "length": 10000,
                    "height": 2000,
                    "base_textures": [
                        "rock_texture",
                        "grass_texture",
                        "sand_texture",
                        "snow_texture"
                    ]
                }
                
                with open(terrain_json_path, 'w') as f:
                    json.dump(terrain_settings, f, indent=2)
        
        # Record this export
        self.export_history.append({
            "type": f"game_engine_{engine}",
            "path": output_dir,
            "resolution": resolution,
            "region": region
        })
        
        print(f"Game engine export for {engine} completed at {output_dir}")
        return output_dir
    
    def _export_unreal_layer_maps(self, output_dir, resolution, region=None):
        """Export material layer weight maps for Unreal Engine"""
        # Define basic layer types
        layers = [
            ("rock", lambda biome, elev: (biome in [0, 1, 11]) or elev > 2),
            ("grass", lambda biome, elev: biome in [2, 3, 4, 5, 6, 8, 9]),
            ("sand", lambda biome, elev: biome == 7 or (biome == 0 and elev > -0.2 and elev < 0.1)),
            ("snow", lambda biome, elev: biome == 1 or (elev > 3)),
            ("mud", lambda biome, elev: biome == 10 or (biome in [5, 9] and elev < 0.5))
        ]
        
        # Generate grid for biomes and elevation
        biome_grid = self._generate_data_grid(self.planet.biome_ids, resolution, resolution, region)
        elev_grid = self._generate_data_grid(self.planet.elevation, resolution, resolution, region)
        
        # Generate and save each layer
        for layer_name, condition in layers:
            layer_grid = np.zeros((resolution, resolution), dtype=np.uint8)
            
            for y in range(resolution):
                for x in range(resolution):
                    biome = int(biome_grid[y, x])
                    elev = float(elev_grid[y, x])
                    
                    if condition(biome, elev):
                        # If condition is met, set weight (0-255)
                        # Additional logic could make this more gradual/blended
                        weight = 255
                        
                        # Feather edges for smooth transitions
                        if layer_name == "snow" and elev < 3.5:
                            # Gradual snow line
                            weight = int(255 * (elev - 3) / 0.5)
                        elif layer_name == "sand" and elev > 0:
                            # Beach transition
                            weight = int(255 * (1 - min(elev*10, 1)))
                        
                        layer_grid[y, x] = max(0, min(255, weight))
            
            # Save as 8-bit grayscale image
            layer_path = os.path.join(output_dir, f"layer_{layer_name}.png")
            img = Image.fromarray(layer_grid, mode='L')
            img.save(layer_path)
            
            print(f"Layer map for {layer_name} exported to {layer_path}")
    
    def _export_unity_splatmap(self, output_dir, resolution, region=None):
        """Export splatmap for Unity terrain system"""
        # Unity splatmap is an RGBA image where each channel represents a texture weight
        # R = first texture, G = second texture, etc.
        
        # Define which biomes map to which texture:
        # R = rock, G = grass, B = sand, A = snow
        texture_mapping = {
            0: [0, 0, 1, 0],  # Ocean = sand
            1: [0, 0, 0, 1],  # Ice Sheet = snow
            2: [0.3, 0.7, 0, 0],  # Tundra = grass + some rock
            3: [0.2, 0.8, 0, 0],  # Taiga = grass + little rock
            4: [0, 1, 0, 0],  # Temperate Forest = grass
            5: [0, 1, 0, 0],  # Temperate Rainforest = grass
            6: [0, 1, 0, 0],  # Temperate Grassland = grass
            7: [0, 0, 1, 0],  # Desert = sand
            8: [0, 0.8, 0.2, 0],  # Savanna = grass + some sand
            9: [0, 1, 0, 0],  # Tropical Seasonal Forest = grass
            10: [0, 1, 0, 0],  # Tropical Rainforest = grass
            11: [1, 0, 0, 0]   # Alpine/Mountain = rock
        }
        
        # Generate biome and elevation grids
        biome_grid = self._generate_data_grid(self.planet.biome_ids, resolution, resolution, region)
        elev_grid = self._generate_data_grid(self.planet.elevation, resolution, resolution, region)
        
        # Create RGBA splatmap
        splatmap = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        
        for y in range(resolution):
            for x in range(resolution):
                biome = int(biome_grid[y, x])
                elev = float(elev_grid[y, x])
                
                # Get base texture weights for this biome
                weights = texture_mapping.get(biome, [0.25, 0.25, 0.25, 0.25])  # Default equal weights
                
                # Adjust weights based on elevation
                if elev > 3:
                    # Add snow at high elevations regardless of biome
                    weights[3] = min(1.0, weights[3] + (elev-3)/2)
                
                if elev < 0 and elev > -0.2:
                    # Beaches
                    weights[2] = 1.0  # Sand
                
                # Normalize weights so they sum to 1
                weight_sum = sum(weights)
                if weight_sum > 0:
                    weights = [w / weight_sum for w in weights]
                
                # Convert to 0-255 range for image
                splatmap[y, x] = [int(w * 255) for w in weights]
        
        # Save as PNG
        splatmap_path = os.path.join(output_dir, "splatmap.png")
        img = Image.fromarray(splatmap.astype(np.uint8), mode='RGBA')
        img.save(splatmap_path)
        
        print(f"Unity splatmap exported to {splatmap_path}")
    
    def _generate_unreal_material_setup(self):
        """Generate material setup JSON for Unreal Engine"""
        return {
            "layers": [
                {
                    "name": "rock",
                    "mask": "layer_rock.png",
                    "albedo": "rock_albedo.png",
                    "normal": "rock_normal.png",
                    "roughness": "rock_roughness.png"
                },
                {
                    "name": "grass",
                    "mask": "layer_grass.png",
                    "albedo": "grass_albedo.png",
                    "normal": "grass_normal.png",
                    "roughness": "grass_roughness.png"
                },
                {
                    "name": "sand",
                    "mask": "layer_sand.png",
                    "albedo": "sand_albedo.png",
                    "normal": "sand_normal.png",
                    "roughness": "sand_roughness.png"
                },
                {
                    "name": "snow",
                    "mask": "layer_snow.png",
                    "albedo": "snow_albedo.png",
                    "normal": "snow_normal.png",
                    "roughness": "snow_roughness.png"
                },
                {
                    "name": "mud",
                    "mask": "layer_mud.png",
                    "albedo": "mud_albedo.png",
                    "normal": "mud_normal.png",
                    "roughness": "mud_roughness.png"
                }
            ],
            "blend_method": "height_based"
        }
    
    def export_all(self, output_dir, width=2048, height=1024, 
                  region=None, include_game_exports=False):
        """
        Export all available planet data formats.
        
        Parameters:
        - output_dir: Directory to save export files
        - width, height: Dimensions of the output images
        - region: Optional tuple (lat_min, lon_min, lat_max, lon_max) for region extract
        - include_game_exports: Whether to include game engine exports
        
        Returns:
        - Dictionary with paths to all exported files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        export_paths = {}
        
        # Export heightmap
        heightmap_path = os.path.join(output_dir, "heightmap.png")
        export_paths["heightmap"] = self.export_heightmap(
            heightmap_path, width=width, height=height, region=region
        )
        
        # Export climate data if available
        if self.planet.temperature is not None:
            temp_path = os.path.join(output_dir, "temperature.png")
            export_paths["temperature"] = self.export_climate_map(
                temp_path, data_type='temperature', width=width, height=height, region=region
            )
        
        if self.planet.precipitation is not None:
            precip_path = os.path.join(output_dir, "precipitation.png")
            export_paths["precipitation"] = self.export_climate_map(
                precip_path, data_type='precipitation', width=width, height=height, region=region
            )
        
        # Export biome map if available
        if self.planet.biome_ids is not None:
            biome_path = os.path.join(output_dir, "biomes.png")
            export_paths["biomes"] = self.export_biome_map(
                biome_path, width=width, height=height, region=region
            )
        
        # Export for game engines if requested
        if include_game_exports:
            unreal_dir = os.path.join(output_dir, "unreal")
            export_paths["unreal"] = self.export_for_game_engine(
                unreal_dir, engine='unreal', region=region
            )
            
            unity_dir = os.path.join(output_dir, "unity")
            export_paths["unity"] = self.export_for_game_engine(
                unity_dir, engine='unity', region=region
            )
        
        # Create a summary JSON
        summary = {
            "export_time": datetime.datetime.now().isoformat(),
            "export_paths": export_paths,
            "planet_data": {
                "resolution": len(self.planet.grid.vertices),
                "elevation_range": [float(self.planet.elevation.min()), float(self.planet.elevation.max())],
                "has_climate": self.planet.temperature is not None,
                "has_biomes": self.planet.biome_ids is not None,
                "region": region
            }
        }
        
        summary_path = os.path.join(output_dir, "export_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"All available data formats exported to {output_dir}")
        return export_paths