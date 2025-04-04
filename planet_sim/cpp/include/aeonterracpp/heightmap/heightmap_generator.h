#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/vector_types.h"
#include <memory>
#include <string>
#include <vector>

namespace aeonterracpp {

/**
 * @brief Data structure to hold heightmap information
 */
struct HeightmapData {
    int width;                  // Width in pixels
    int height;                 // Height in pixels
    std::vector<float> data;    // Raw height data (row-major order)
    float minValue;             // Minimum height value
    float maxValue;             // Maximum height value
    BoundingBox region;         // Region covered by the heightmap
    
    // Constructor
    HeightmapData(int width, int height, const BoundingBox& region);
    
    // Get height at specific coordinates
    float get(int x, int y) const;
    
    // Set height at specific coordinates
    void set(int x, int y, float value);
    
    // Normalize values to 0-1 range
    void normalize();
    
    // Blend with another heightmap
    void blend(const HeightmapData& other, float weight);
};

/**
 * @brief Configuration for heightmap generation
 */
struct HeightmapConfig {
    // Input parameters
    BoundingBox region;         // Region to extract
    int width = 1024;           // Width in pixels
    int height = 1024;          // Height in pixels
    
    // Detail parameters
    bool enhanceDetail = false; // Whether to enhance detail
    float roughness = 0.5f;     // Roughness for detail enhancement
    int octaves = 4;            // Number of octaves for detail enhancement
    float detailScale = 0.2f;   // Scale of detail relative to base terrain
    
    // Export parameters
    bool normalizeValues = true;// Normalize values to 0-1 range
    int bitDepth = 16;          // Bit depth for export (8, 16, or 32)
    
    // Tiling parameters
    bool useTiling = false;     // Use tiled processing for large heightmaps
    int tileSize = 1024;        // Size of each tile
    int tileOverlap = 64;       // Overlap between tiles
};

/**
 * @brief Generator for height maps from planet data
 */
class HeightmapGenerator {
public:
    /**
     * @brief Constructor
     * 
     * @param maxResolution Maximum supported resolution
     */
    HeightmapGenerator(int maxResolution = 16384);
    
    /**
     * @brief Destructor
     */
    ~HeightmapGenerator();
    
    /**
     * @brief Generate a heightmap from planet data
     * 
     * @param planet Source planet
     * @param config Configuration parameters
     * @return HeightmapData Generated heightmap
     */
    HeightmapData generateHeightmap(const Planet& planet, const HeightmapConfig& config);
    
    /**
     * @brief Enhance detail in a heightmap using fractal noise
     * 
     * @param heightmap Heightmap to enhance
     * @param roughness Roughness parameter (0-1)
     * @param octaves Number of octaves
     * @param scale Scale of detail relative to base terrain
     */
    void enhanceDetail(HeightmapData& heightmap, float roughness, int octaves, float scale);
    
    /**
     * @brief Export heightmap as a PNG image
     * 
     * @param heightmap Heightmap to export
     * @param filename Output filename
     * @param normalizeValues Whether to normalize values to 0-1 range
     */
    void exportPNG(const HeightmapData& heightmap, const std::string& filename, bool normalizeValues = true);
    
    /**
     * @brief Export heightmap as a TIFF image
     * 
     * @param heightmap Heightmap to export
     * @param filename Output filename
     * @param bitDepth Bit depth (8, 16, or 32)
     */
    void exportTIFF(const HeightmapData& heightmap, const std::string& filename, int bitDepth = 16);
    
    /**
     * @brief Export heightmap as raw binary data
     * 
     * @param heightmap Heightmap to export
     * @param filename Output filename
     * @param normalizeValues Whether to normalize values to 0-1 range
     */
    void exportRAW(const HeightmapData& heightmap, const std::string& filename, bool normalizeValues = true);
    
private:
    // Maximum heightmap resolution
    int maxResolution;
    
    // Internal methods
    HeightmapData generateHeightmapDirect(const Planet& planet, const HeightmapConfig& config);
    HeightmapData generateHeightmapTiled(const Planet& planet, const HeightmapConfig& config);
    void processRegion(const Planet& planet, HeightmapData& heightmap, const BoundingBox& region);
    
    // Noise generation for detail enhancement
    float generateNoise(float x, float y, int seed) const;
    float generateFractalNoise(float x, float y, int octaves, float roughness, int seed) const;
};

/**
 * @brief Factory for creating commonly used heightmap configurations
 */
class HeightmapConfigFactory {
public:
    /**
     * @brief Create a global equirectangular projection config
     * 
     * @param width Width in pixels
     * @param height Height in pixels
     * @return HeightmapConfig Configuration
     */
    static HeightmapConfig createGlobalConfig(int width = 4096, int height = 2048);
    
    /**
     * @brief Create a regional equirectangular projection config
     * 
     * @param centerLat Center latitude in degrees
     * @param centerLon Center longitude in degrees
     * @param width Width in degrees
     * @param height Height in degrees
     * @param resolution Pixels per degree
     * @return HeightmapConfig Configuration
     */
    static HeightmapConfig createRegionalConfig(float centerLat, float centerLon, 
                                               float width, float height,
                                               int resolution = 16);
    
    /**
     * @brief Create a high-detail terrain config
     * 
     * @param region Region to extract
     * @param resolution Resolution in pixels
     * @return HeightmapConfig Configuration
     */
    static HeightmapConfig createDetailedTerrainConfig(const BoundingBox& region, int resolution = 2048);
};

} // namespace aeonterracpp