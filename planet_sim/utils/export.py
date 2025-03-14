import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import math
import datetime

class ExportManager:
    """
    Handles exporting planet data to various formats.
    Includes heightmaps, climate maps, and game engine exports.
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
        # Generate heightmap
        heightmap = self._generate_heightmap_grid(width, height, region, scaling_factor)
        
        # Export based on format
        if format == 'png':
            self._export_png_heightmap(heightmap, output_path, bit_depth)
        elif format == 'raw':
            self._export_raw_heightmap(heightmap, output_path, bit_depth)
        elif format == 'tiff':
            self._export_tiff_heightmap(heightmap, output_path, bit_depth)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Add metadata file if requested
        if add_metadata:
            min_elevation = float(np.min(self.planet.elevation))
            max_elevation = float(np.max(self.planet.elevation))
            
            if region:
                projection = 'mercator'
                lat_min, lon_min, lat_max, lon_max = region
            else:
                projection = 'equirectangular'
                lat_min, lon_min, lat_max, lon_max = -90, -180, 90, 180
                
            metadata = {
                "width": width,
                "height": height,
                "min_elevation_km": min_elevation,
                "max_elevation_km": max_elevation,
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
            "projection": "mercator" if region else "equirectangular",
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
        Export a climate data map.
        
        Parameters:
        - output_path: Path to save the map
        - data_type: Type of climate data ('temperature' or 'precipitation')
        - width, height: Dimensions of the output image
        - region: Optional tuple (lat_min, lon_min, lat_max, lon_max) for region extract
        - colormap: Matplotlib colormap to use
        - bit_depth: Bit depth for the output image
        
        Returns:
        - Path to the exported map
        """
        # Check if climate data exists
        if data_type == 'temperature' and self.planet.temperature is None:
            raise ValueError("Temperature data not available. Run climate simulation first.")
        elif data_type == 'precipitation' and self.planet.precipitation is None:
            raise ValueError("Precipitation data not available. Run climate simulation first.")
        
        # Get the appropriate data array
        if data_type == 'temperature':
            data = self.planet.temperature
            title = "Temperature Map (°C)"
            cmap = plt.cm.coolwarm if colormap == 'viridis' else plt.get_cmap(colormap)
        else:  # precipitation
            data = self.planet.precipitation
            title = "Precipitation Map (mm/year)"
            cmap = plt.cm.Blues if colormap == 'viridis' else plt.get_cmap(colormap)
        
        # Create the data grid
        grid = self._generate_data_grid(data, width, height, region)
        
        # Export the data as an image
        self._export_data_image(grid, output_path, cmap, title, bit_depth)
        
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
        from ..core.biome import BiomeClassifier
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
                material_setup = {
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
                        }
                    ],
                    "blend_method": "height_based"
                }
                
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
        # Define basic layer types based on biomes and elevation
        # This is a simplified version - a full implementation would be more complex
        if self.planet.biome_ids is None or self.planet.temperature is None:
            print("Warning: Biome or climate data not available for material maps")
            return
        
        # Create necessary data grids
        biome_grid = self._generate_data_grid(self.planet.biome_ids, resolution, resolution, region)
        elev_grid = self._generate_data_grid(self.planet.elevation, resolution, resolution, region)
        
        # Define mapping of biomes to materials
        biome_to_material = {
            0: "water",     # Ocean
            1: "snow",      # Ice Sheet
            2: "grass",     # Tundra
            3: "grass",     # Taiga
            4: "grass",     # Temperate Forest
            5: "grass",     # Temperate Rainforest
            6: "grass",     # Temperate Grassland
            7: "sand",      # Desert
            8: "grass",     # Savanna
            9: "grass",     # Tropical Seasonal Forest
            10: "grass",    # Tropical Rainforest
            11: "rock"      # Alpine/Mountain
        }
        
        # Initialize layer maps
        rock_layer = np.zeros((resolution, resolution), dtype=np.uint8)
        grass_layer = np.zeros((resolution, resolution), dtype=np.uint8)
        sand_layer = np.zeros((resolution, resolution), dtype=np.uint8)
        snow_layer = np.zeros((resolution, resolution), dtype=np.uint8)
        
        # Fill layer maps based on biome and elevation
        for y in range(resolution):
            for x in range(resolution):
                biome = int(biome_grid[y, x])
                elev = float(elev_grid[y, x])
                
                # Default material from biome
                primary_material = biome_to_material.get(biome, "rock")
                
                # Set primary material weight
                if primary_material == "rock":
                    rock_layer[y, x] = 255
                elif primary_material == "grass":
                    grass_layer[y, x] = 255
                elif primary_material == "sand":
                    sand_layer[y, x] = 255
                elif primary_material == "snow":
                    snow_layer[y, x] = 255
                
                # Additional rules based on elevation
                if elev > 3.0:  # High mountains get snow regardless of biome
                    snow_factor = min(1.0, (elev - 3.0) / 2.0)
                    snow_layer[y, x] = int(255 * snow_factor)
                
                if elev < 0.1 and elev >= 0:  # Beaches get sand
                    sand_factor = 1.0 - min(1.0, elev / 0.1)
                    sand_layer[y, x] = int(255 * sand_factor)
                
                if elev > 1.0 and elev < 3.0:  # Mid-high elevations get more rock
                    rock_factor = min(1.0, (elev - 1.0) / 2.0)
                    rock_layer[y, x] = int(255 * rock_factor)
        
        # Save layer maps
        def save_layer(layer, name):
            path = os.path.join(output_dir, f"layer_{name}.png")
            img = Image.fromarray(layer, mode='L')
            img.save(path)
            return path
        
        save_layer(rock_layer, "rock")
        save_layer(grass_layer, "grass")
        save_layer(sand_layer, "sand")
        save_layer(snow_layer, "snow")
        
        print(f"Exported material layer maps to {output_dir}")
    
    def _export_unity_splatmap(self, output_dir, resolution, region=None):
        """Export splatmap for Unity terrain system"""
        # Unity splatmap is an RGBA image where each channel represents a texture weight
        # R = first texture, G = second texture, etc.
        
        # Create necessary data grids
        biome_grid = self._generate_data_grid(self.planet.biome_ids, resolution, resolution, region)
        elev_grid = self._generate_data_grid(self.planet.elevation, resolution, resolution, region)
        
        # Create RGBA splatmap
        splatmap = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        
        # Fill splatmap based on biome and elevation
        # Channel mapping: R = rock, G = grass, B = sand, A = snow
        for y in range(resolution):
            for x in range(resolution):
                biome = int(biome_grid[y, x])
                elev = float(elev_grid[y, x])
                
                # Initialize weights
                weights = [0, 0, 0, 0]  # rock, grass, sand, snow
                
                # Set weights based on biome
                if biome == 0:  # Ocean
                    weights = [0, 0, 1, 0]  # Sand
                elif biome == 1:  # Ice Sheet
                    weights = [0, 0, 0, 1]  # Snow
                elif biome in [2, 3, 4, 5, 6, 8, 9, 10]:  # Various forests and grasslands
                    weights = [0, 1, 0, 0]  # Grass
                elif biome == 7:  # Desert
                    weights = [0, 0, 1, 0]  # Sand
                elif biome == 11:  # Alpine
                    weights = [1, 0, 0, 0]  # Rock
                
                # Adjust weights based on elevation
                if elev > 3.0:  # High mountains get snow
                    snow_factor = min(1.0, (elev - 3.0) / 2.0)
                    weights[3] = snow_factor
                
                if elev < 0.1 and elev >= 0:  # Beaches get sand
                    sand_factor = 1.0 - min(1.0, elev / 0.1)
                    weights[2] = sand_factor
                
                if elev > 1.0 and elev < 3.0:  # Mid-high elevations get more rock
                    rock_factor = min(1.0, (elev - 1.0) / 2.0)
                    weights[0] = rock_factor
                
                # Normalize weights so they sum to 1
                weight_sum = sum(weights)
                if weight_sum > 0:
                    weights = [w / weight_sum for w in weights]
                
                # Set splatmap values (0-255)
                splatmap[y, x] = [int(w * 255) for w in weights]
        
        # Save splatmap
        splatmap_path = os.path.join(output_dir, "splatmap.png")
        img = Image.fromarray(splatmap.astype(np.uint8), mode='RGBA')
        img.save(splatmap_path)
        
        print(f"Exported Unity splatmap to {splatmap_path}")
        return splatmap_path

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