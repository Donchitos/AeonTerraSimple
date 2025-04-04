#include "aeonterracpp/heightmap/heightmap_generator.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace aeonterracpp {

//-----------------------------------------------------------------------------
// HeightmapData implementation
//-----------------------------------------------------------------------------

HeightmapData::HeightmapData(int width, int height, const BoundingBox& region)
    : width(width),
      height(height),
      data(width * height, 0.0f),
      minValue(std::numeric_limits<float>::max()),
      maxValue(-std::numeric_limits<float>::max()),
      region(region) {
}

float HeightmapData::get(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        throw std::out_of_range("Heightmap coordinates out of range");
    }
    return data[y * width + x];
}

void HeightmapData::set(int x, int y, float value) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        throw std::out_of_range("Heightmap coordinates out of range");
    }
    data[y * width + x] = value;
    
    // Update min/max values
    minValue = std::min(minValue, value);
    maxValue = std::max(maxValue, value);
}

void HeightmapData::normalize() {
    if (maxValue == minValue) {
        // Avoid division by zero
        std::fill(data.begin(), data.end(), 0.5f);
        minValue = 0.5f;
        maxValue = 0.5f;
        return;
    }
    
    float range = maxValue - minValue;
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (data[i] - minValue) / range;
    }
    
    minValue = 0.0f;
    maxValue = 1.0f;
}

void HeightmapData::blend(const HeightmapData& other, float weight) {
    if (width != other.width || height != other.height) {
        throw std::invalid_argument("Cannot blend heightmaps of different sizes");
    }
    
    // Blend each value
    float invWeight = 1.0f - weight;
    minValue = std::numeric_limits<float>::max();
    maxValue = -std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = data[i] * invWeight + other.data[i] * weight;
        minValue = std::min(minValue, data[i]);
        maxValue = std::max(maxValue, data[i]);
    }
}

//-----------------------------------------------------------------------------
// HeightmapGenerator implementation
//-----------------------------------------------------------------------------

HeightmapGenerator::HeightmapGenerator(int maxResolution)
    : maxResolution(maxResolution) {
}

HeightmapGenerator::~HeightmapGenerator() = default;

HeightmapData HeightmapGenerator::generateHeightmap(const Planet& planet, const HeightmapConfig& config) {
    // Verify resolution limits
    if (config.width > maxResolution || config.height > maxResolution) {
        throw std::invalid_argument("Requested heightmap resolution exceeds maximum supported resolution");
    }
    
    // Use tiled processing for large heightmaps if requested
    if (config.useTiling && 
        (config.width > config.tileSize || config.height > config.tileSize)) {
        return generateHeightmapTiled(planet, config);
    } else {
        return generateHeightmapDirect(planet, config);
    }
}

HeightmapData HeightmapGenerator::generateHeightmapDirect(const Planet& planet, const HeightmapConfig& config) {
    // Create heightmap data structure
    HeightmapData heightmap(config.width, config.height, config.region);
    
    // Process the region
    processRegion(planet, heightmap, config.region);
    
    // Enhance detail if requested
    if (config.enhanceDetail) {
        enhanceDetail(heightmap, config.roughness, config.octaves, config.detailScale);
    }
    
    // Normalize if requested
    if (config.normalizeValues) {
        heightmap.normalize();
    }
    
    return heightmap;
}

HeightmapData HeightmapGenerator::generateHeightmapTiled(const Planet& planet, const HeightmapConfig& config) {
    // Create the final heightmap
    HeightmapData finalHeightmap(config.width, config.height, config.region);
    
    // Calculate tile dimensions
    int tileSize = config.tileSize;
    int overlap = config.tileOverlap;
    
    int tilesX = (config.width + tileSize - 1) / tileSize;
    int tilesY = (config.height + tileSize - 1) / tileSize;
    
    // Latitude and longitude ranges
    float latRange = config.region.latMax - config.region.latMin;
    float lonRange = config.region.lonMax - config.region.lonMin;
    
    // Process each tile
    for (int tileY = 0; tileY < tilesY; ++tileY) {
        for (int tileX = 0; tileX < tilesX; ++tileX) {
            // Tile coordinates in pixels
            int tileStartX = tileX * tileSize - (tileX > 0 ? overlap : 0);
            int tileStartY = tileY * tileSize - (tileY > 0 ? overlap : 0);
            int tileEndX = std::min((tileX + 1) * tileSize + (tileX < tilesX - 1 ? overlap : 0), config.width);
            int tileEndY = std::min((tileY + 1) * tileSize + (tileY < tilesY - 1 ? overlap : 0), config.height);
            
            int tileWidth = tileEndX - tileStartX;
            int tileHeight = tileEndY - tileStartY;
            
            // Tile coordinates in lat/lon
            float tileLatMin = config.region.latMin + latRange * static_cast<float>(tileStartY) / config.height;
            float tileLatMax = config.region.latMin + latRange * static_cast<float>(tileEndY) / config.height;
            float tileLonMin = config.region.lonMin + lonRange * static_cast<float>(tileStartX) / config.width;
            float tileLonMax = config.region.lonMin + lonRange * static_cast<float>(tileEndX) / config.width;
            
            BoundingBox tileRegion(tileLatMin, tileLonMin, tileLatMax, tileLonMax);
            
            // Create tile heightmap
            HeightmapData tileHeightmap(tileWidth, tileHeight, tileRegion);
            
            // Process tile
            processRegion(planet, tileHeightmap, tileRegion);
            
            // Enhance detail if requested
            if (config.enhanceDetail) {
                enhanceDetail(tileHeightmap, config.roughness, config.octaves, config.detailScale);
            }
            
            // Copy tile data to final heightmap
            for (int y = 0; y < tileHeight; ++y) {
                for (int x = 0; x < tileWidth; ++x) {
                    // Coordinates in final heightmap
                    int finalX = tileStartX + x;
                    int finalY = tileStartY + y;
                    
                    // Skip if outside final heightmap
                    if (finalX < 0 || finalX >= config.width || finalY < 0 || finalY >= config.height) {
                        continue;
                    }
                    
                    // Apply blending for overlap regions
                    if ((tileX > 0 && x < overlap) || 
                        (tileX < tilesX - 1 && x >= tileWidth - overlap) ||
                        (tileY > 0 && y < overlap) ||
                        (tileY < tilesY - 1 && y >= tileHeight - overlap)) {
                        
                        // Calculate blend weight based on distance to edge
                        float blendWeight = 1.0f;
                        
                        if (tileX > 0 && x < overlap) {
                            blendWeight = std::min(blendWeight, static_cast<float>(x) / overlap);
                        }
                        if (tileX < tilesX - 1 && x >= tileWidth - overlap) {
                            blendWeight = std::min(blendWeight, static_cast<float>(tileWidth - 1 - x) / overlap);
                        }
                        if (tileY > 0 && y < overlap) {
                            blendWeight = std::min(blendWeight, static_cast<float>(y) / overlap);
                        }
                        if (tileY < tilesY - 1 && y >= tileHeight - overlap) {
                            blendWeight = std::min(blendWeight, static_cast<float>(tileHeight - 1 - y) / overlap);
                        }
                        
                        // Only set if this is a better value than existing
                        float tileValue = tileHeightmap.get(x, y);
                        finalHeightmap.set(finalX, finalY, tileValue);
                    } else {
                        // No blending needed for non-overlap regions
                        finalHeightmap.set(finalX, finalY, tileHeightmap.get(x, y));
                    }
                }
            }
        }
    }
    
    // Normalize if requested
    if (config.normalizeValues) {
        finalHeightmap.normalize();
    }
    
    return finalHeightmap;
}

void HeightmapGenerator::processRegion(const Planet& planet, HeightmapData& heightmap, const BoundingBox& region) {
    const auto& grid = planet.getGrid();
    
    // Get all vertices within the region
    std::vector<int> regionVertices = grid.getVerticesInRegion(region);
    
    // If no vertices in region, return empty heightmap
    if (regionVertices.empty()) {
        std::cerr << "Warning: No vertices found in region " 
                  << region.latMin << "," << region.lonMin << " to "
                  << region.latMax << "," << region.lonMax << std::endl;
        return;
    }
    
    // Process each pixel in the heightmap
    for (int y = 0; y < heightmap.height; ++y) {
        for (int x = 0; x < heightmap.width; ++x) {
            // Convert pixel coordinates to lat/lon
            float lat = region.latMin + (region.latMax - region.latMin) * static_cast<float>(y) / heightmap.height;
            float lon = region.lonMin + (region.lonMax - region.lonMin) * static_cast<float>(x) / heightmap.width;
            
            // Convert lat/lon to 3D position
            Vec3f position = grid.latLonToCartesian(lat, lon);
            
            // Find closest vertices and interpolate elevation
            float totalWeight = 0.0f;
            float weightedSum = 0.0f;
            
            // Use inverse distance weighting for interpolation
            for (int vertexIdx : regionVertices) {
                Vec3f vertexPos = grid.getVertex(vertexIdx);
                
                // Calculate distance
                float distance = (position - vertexPos).norm();
                
                // Skip if too far
                const float maxDistance = 0.2f * grid.getRadius();
                if (distance > maxDistance) {
                    continue;
                }
                
                // Calculate weight (inverse distance squared)
                float weight = 1.0f / (distance * distance + 0.0001f);
                
                // Get elevation at this vertex
                float elevation = planet.getElevation(vertexIdx);
                
                // Add to weighted sum
                weightedSum += elevation * weight;
                totalWeight += weight;
            }
            
            // Set the interpolated height
            if (totalWeight > 0.0f) {
                heightmap.set(x, y, weightedSum / totalWeight);
            } else {
                // No vertices close enough, use zero elevation
                heightmap.set(x, y, 0.0f);
            }
        }
    }
}

void HeightmapGenerator::enhanceDetail(HeightmapData& heightmap, float roughness, int octaves, float scale) {
    // Create a new heightmap for the detail
    HeightmapData detailMap(heightmap.width, heightmap.height, heightmap.region);
    
    // Generate noise for each pixel
    for (int y = 0; y < heightmap.height; ++y) {
        for (int x = 0; x < heightmap.width; ++x) {
            // Get normalized coordinates (0-1)
            float nx = static_cast<float>(x) / heightmap.width;
            float ny = static_cast<float>(y) / heightmap.height;
            
            // Generate fractal noise
            float noise = generateFractalNoise(nx * 10.0f, ny * 10.0f, octaves, roughness, 42);
            
            // Normalize noise to -1 to 1 range
            noise = noise * 2.0f - 1.0f;
            
            // Set detail value
            detailMap.set(x, y, noise);
        }
    }
    
    // Blend detail with original heightmap
    for (size_t i = 0; i < heightmap.data.size(); ++i) {
        // Get local height range (use nearby average as reference)
        float baseHeight = heightmap.data[i];
        float detail = detailMap.data[i];
        
        // Scale detail by local height and configured scale factor
        float detailAmount = scale * (baseHeight - heightmap.minValue) / 
                           (heightmap.maxValue - heightmap.minValue + 0.0001f);
        
        // Apply detail
        heightmap.data[i] += detail * detailAmount;
    }
    
    // Update min/max values
    heightmap.minValue = *std::min_element(heightmap.data.begin(), heightmap.data.end());
    heightmap.maxValue = *std::max_element(heightmap.data.begin(), heightmap.data.end());
}

float HeightmapGenerator::generateNoise(float x, float y, int seed) const {
    // Simple hash-based noise function
    int xi = static_cast<int>(std::floor(x));
    int yi = static_cast<int>(std::floor(y));
    float xf = x - xi;
    float yf = y - yi;
    
    // Hash coordinates and seed for pseudo-random values
    const auto hash = [seed](int x, int y) {
        int h = seed + x * 374761393 + y * 668265263;
        h = (h ^ (h >> 13)) * 1274126177;
        return static_cast<float>(h & 0x7FFFFFFF) / 0x7FFFFFFF;
    };
    
    // Get random values at grid corners
    float v00 = hash(xi, yi);
    float v10 = hash(xi + 1, yi);
    float v01 = hash(xi, yi + 1);
    float v11 = hash(xi + 1, yi + 1);
    
    // Smooth interpolation function
    const auto smoothstep = [](float t) {
        return t * t * (3.0f - 2.0f * t);
    };
    
    // Interpolate using smoothstep
    float sx = smoothstep(xf);
    float sy = smoothstep(yf);
    
    float i0 = v00 + sx * (v10 - v00);
    float i1 = v01 + sx * (v11 - v01);
    
    return i0 + sy * (i1 - i0);
}

float HeightmapGenerator::generateFractalNoise(float x, float y, int octaves, float roughness, int seed) const {
    float total = 0.0f;
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float normalizer = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        total += generateNoise(x * frequency, y * frequency, seed + i) * amplitude;
        normalizer += amplitude;
        
        amplitude *= roughness;
        frequency *= 2.0f;
    }
    
    return total / normalizer;
}

void HeightmapGenerator::exportPNG(const HeightmapData& heightmap, const std::string& filename, bool normalizeValues) {
    // This is a placeholder for PNG export
    // In a real implementation, you would use a library like stb_image_write.h, libpng, etc.
    
    std::cout << "PNG export is not implemented in this version" << std::endl;
    std::cout << "Exporting raw data to " << filename << ".raw instead" << std::endl;
    
    // Export as raw data instead
    exportRAW(heightmap, filename + ".raw", normalizeValues);
}

void HeightmapGenerator::exportTIFF(const HeightmapData& heightmap, const std::string& filename, int bitDepth) {
    // This is a placeholder for TIFF export
    // In a real implementation, you would use a library like libtiff
    
    std::cout << "TIFF export is not implemented in this version" << std::endl;
    std::cout << "Exporting raw data to " << filename << ".raw instead" << std::endl;
    
    // Export as raw data instead
    exportRAW(heightmap, filename + ".raw", true);
}

void HeightmapGenerator::exportRAW(const HeightmapData& heightmap, const std::string& filename, bool normalizeValues) {
    // Create a copy to avoid modifying the original
    HeightmapData exportData = heightmap;
    
    // Normalize if requested
    if (normalizeValues && (exportData.minValue != 0.0f || exportData.maxValue != 1.0f)) {
        exportData.normalize();
    }
    
    // Write header and data to file
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write header (width, height)
    file.write(reinterpret_cast<const char*>(&exportData.width), sizeof(exportData.width));
    file.write(reinterpret_cast<const char*>(&exportData.height), sizeof(exportData.height));
    
    // Write min/max values
    file.write(reinterpret_cast<const char*>(&exportData.minValue), sizeof(exportData.minValue));
    file.write(reinterpret_cast<const char*>(&exportData.maxValue), sizeof(exportData.maxValue));
    
    // Write region
    file.write(reinterpret_cast<const char*>(&exportData.region.latMin), sizeof(exportData.region.latMin));
    file.write(reinterpret_cast<const char*>(&exportData.region.lonMin), sizeof(exportData.region.lonMin));
    file.write(reinterpret_cast<const char*>(&exportData.region.latMax), sizeof(exportData.region.latMax));
    file.write(reinterpret_cast<const char*>(&exportData.region.lonMax), sizeof(exportData.region.lonMax));
    
    // Write height data
    file.write(reinterpret_cast<const char*>(exportData.data.data()), 
               exportData.data.size() * sizeof(float));
    
    std::cout << "Exported heightmap (" << exportData.width << "x" << exportData.height 
              << ") to " << filename << std::endl;
}

//-----------------------------------------------------------------------------
// HeightmapConfigFactory implementation
//-----------------------------------------------------------------------------

HeightmapConfig HeightmapConfigFactory::createGlobalConfig(int width, int height) {
    HeightmapConfig config;
    config.width = width;
    config.height = height;
    config.region = BoundingBox(-90.0f, -180.0f, 90.0f, 180.0f);
    
    // Use tiling for large heightmaps
    if (width > 4096 || height > 4096) {
        config.useTiling = true;
        config.tileSize = 2048;
        config.tileOverlap = 64;
    }
    
    return config;
}

HeightmapConfig HeightmapConfigFactory::createRegionalConfig(float centerLat, float centerLon, 
                                                           float width, float height,
                                                           int resolution) {
    HeightmapConfig config;
    
    // Calculate lat/lon bounds
    float latMin = centerLat - height / 2.0f;
    float latMax = centerLat + height / 2.0f;
    float lonMin = centerLon - width / 2.0f;
    float lonMax = centerLon + width / 2.0f;
    
    // Clamp to valid ranges
    latMin = std::max(-90.0f, latMin);
    latMax = std::min(90.0f, latMax);
    
    // Normalize longitude to -180 to 180
    while (lonMin < -180.0f) lonMin += 360.0f;
    while (lonMin > 180.0f) lonMin -= 360.0f;
    while (lonMax < -180.0f) lonMax += 360.0f;
    while (lonMax > 180.0f) lonMax -= 360.0f;
    
    // Handle cases where region crosses the date line
    if (lonMin > lonMax) {
        // Split into two regions or wrap around
        // For simplicity, we'll wrap around here
        lonMax += 360.0f;
    }
    
    config.region = BoundingBox(latMin, lonMin, latMax, lonMax);
    
    // Calculate pixel dimensions
    config.width = static_cast<int>(std::ceil(width * resolution));
    config.height = static_cast<int>(std::ceil(height * resolution));
    
    // Enable detail enhancement for regional maps
    config.enhanceDetail = true;
    config.roughness = 0.6f;
    config.octaves = 6;
    config.detailScale = 0.3f;
    
    return config;
}

HeightmapConfig HeightmapConfigFactory::createDetailedTerrainConfig(const BoundingBox& region, int resolution) {
    HeightmapConfig config;
    config.region = region;
    
    // Calculate width and height in degrees
    float widthDegrees = region.lonMax - region.lonMin;
    float heightDegrees = region.latMax - region.latMin;
    
    // Normalize width for date line crossing
    if (widthDegrees < 0.0f) widthDegrees += 360.0f;
    
    // Calculate pixel dimensions
    config.width = static_cast<int>(std::ceil(widthDegrees * resolution));
    config.height = static_cast<int>(std::ceil(heightDegrees * resolution));
    
    // Max resolution check
    const int maxDimension = 16384;
    if (config.width > maxDimension || config.height > maxDimension) {
        float scale = std::min(
            static_cast<float>(maxDimension) / config.width,
            static_cast<float>(maxDimension) / config.height
        );
        config.width = static_cast<int>(config.width * scale);
        config.height = static_cast<int>(config.height * scale);
    }
    
    // Enable detail enhancement with high detail
    config.enhanceDetail = true;
    config.roughness = 0.7f;
    config.octaves = 8;
    config.detailScale = 0.5f;
    
    // Enable tiling for large heightmaps
    if (config.width > 4096 || config.height > 4096) {
        config.useTiling = true;
        config.tileSize = 2048;
        config.tileOverlap = 64;
    }
    
    return config;
}

} // namespace aeonterracpp