# AeonTerra Export Formats Specification

This document specifies the file formats and data structures used for exporting data from the AeonTerra planet generation system to external applications like game engines.

## Overview

AeonTerra provides several export formats for different purposes:

1. **Heightmap Images** - For terrain elevation data
2. **Data Maps** - For climate, biome, and other property data
3. **Region Extracts** - For focused export of specific areas
4. **Game Engine Packages** - For direct integration with game engines

## Heightmap Format

### Standard Heightmap

Heightmaps are exported as 16-bit grayscale PNG files by default.

**Format Details:**
- **Bit Depth**: 16-bit (65,536 height levels)
- **Value Range**: 0 (lowest point) to 65,535 (highest point)
- **Projection**: Equirectangular (for full planets), Mercator (for regions)
- **File Extension**: `.png`

**Examples:**
- `planet_heightmap.png` - Full planet heightmap
- `region_heightmap.png` - Specific region heightmap

### RAW Heightmap

For maximum precision, heightmaps can also be exported as RAW 16-bit files.

**Format Details:**
- **Bit Depth**: 16-bit unsigned integer
- **Endianness**: Little-endian
- **Header**: None (headerless)
- **Data Order**: Row by row, from top to bottom
- **File Extension**: `.r16` or `.raw`

**Example Usage in Unreal Engine:**
1. Import the RAW file into the Landscape tool
2. Specify the same width and height used during export
3. Set component size based on desired scale

### Height Scaling

Height values are scaled from actual elevation to the 0-65535 range:
- Minimum elevation (typically ocean floors) maps to value 0
- Maximum elevation (highest mountains) maps to value 65535
- Linear scaling is applied between these points

**Metadata File:**
Each heightmap includes a corresponding `.json` metadata file with:
```json
{
  "width": 4096,
  "height": 2048,
  "min_elevation_km": -11.034,
  "max_elevation_km": 8.848,
  "scale_factor": 3307.31,
  "projection": "equirectangular"
}
```

## Climate and Biome Data Maps

### Temperature Map

**Format Details:**
- **Bit Depth**: 16-bit grayscale or 8-bit RGB
- **Value Range**: 0 (coldest) to 65,535 (hottest)
- **Color Mapping**: Blue (cold) to Red (hot)
- **File Extension**: `.png`

### Precipitation Map

**Format Details:**
- **Bit Depth**: 16-bit grayscale or 8-bit RGB
- **Value Range**: 0 (no precipitation) to 65,535 (maximum precipitation)
- **Color Mapping**: Yellow (dry) to Blue (wet)
- **File Extension**: `.png`

### Biome Map

**Format Details:**
- **Format**: 8-bit indexed color PNG
- **Values**: 0-255, each representing a specific biome type
- **Palette**: Custom color palette matching biome colors in visualization
- **File Extension**: `.png`

### Combination Maps

For game engine use, several properties can be combined into RGBA channels:
- **R**: Height
- **G**: Moisture/precipitation
- **B**: Temperature
- **A**: Biome index or terrain roughness

## Region Extraction Format

### Region Definition

Regions are defined by latitude/longitude boundaries or by center point and radius:

```json
{
  "region_type": "bounds",
  "lat_min": 30.0,
  "lat_max": 45.0,
  "lon_min": -120.0,
  "lon_max": -100.0,
  "resolution": 2048
}
```

Or:

```json
{
  "region_type": "radius",
  "center_lat": 37.5,
  "center_lon": -110.0,
  "radius_km": 500,
  "resolution": 2048
}
```

### Projection Methods

For region exports, different projections are available:

1. **Equirectangular** - Simple lat/lon projection (default)
2. **Mercator** - Preserves angles, distorts area
3. **Lambert Conformal Conic** - Good for mid-latitude regions
4. **Stereographic** - For polar regions

Each projection includes appropriate metadata for georeferencing.

## Game Engine Integration Formats

### Unreal Engine Format

For direct use in Unreal Engine, AeonTerra exports:

1. **Heightmap** - 16-bit RAW file (`.r16`) or PNG
2. **Layer Maps** - Set of 8-bit grayscale images for terrain layers
   - `layer_rock.png`
   - `layer_grass.png`
   - `layer_sand.png`
   - etc.
3. **Material Setup** - JSON file describing material structure:

```json
{
  "layers": [
    {
      "name": "rock",
      "mask": "layer_rock.png",
      "albedo": "rock_albedo.png",
      "normal": "rock_normal.png",
      "roughness": "rock_roughness.png"
    },
    ...
  ],
  "blend_method": "height_based"
}
```

### Unity Format

For Unity terrain system:

1. **Heightmap** - 16-bit RAW file (`.raw`)
2. **Splatmap** - RGBA PNG with terrain texture weights
3. **Terrain Data** - JSON file with terrain settings:

```json
{
  "heightmap_resolution": 2049,
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
```

## File Naming Convention

AeonTerra follows this naming convention for export files:

```
[planet_name]_[data_type]_[region_identifier]_[resolution].[extension]
```

Examples:
- `terraworld_height_full_4096.png`
- `terraworld_biome_NorthAmerica_2048.png`
- `terraworld_temperature_equator_1024.png`

## Example Export Commands

### Basic Heightmap Export

```python
from planet_sim.utils.export import export_heightmap

# Export full planet heightmap
export_heightmap(planet, "planet_full_8192.png", width=8192, height=4096)
```

### Region Export

```python
from planet_sim.utils.export import export_region

# Export specific region
export_region(
    planet,
    "mountain_region",
    lat_bounds=(30, 45),
    lon_bounds=(-120, -100),
    resolution=2048,
    formats=["heightmap", "biome", "temperature", "precipitation"]
)
```

### Game Engine Export

```python
from planet_sim.utils.export import export_for_unreal

# Export for Unreal Engine
export_for_unreal(
    planet,
    "unreal_terrain",
    region_center=(37.5, -110.0),
    region_radius_km=500,
    heightmap_resolution=4097,  # Unreal requires 2^n+1
    include_layers=True
)
```

## Technical Notes

1. All heightmaps use a vertical scale factor to exaggerate terrain features for better visualization
2. Biome maps include a 1-pixel border overlap to prevent rendering artifacts at texture edges
3. For game engines, consider using multiple tiles for very large terrains to avoid memory limitations
4. When encoding multiple data layers in different channels, be sure to document the exact mapping