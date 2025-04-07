#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/tectonics.h"
#include "aeonterracpp/core/vector_types.h"
#include <memory>
#include <random>
#include <vector>
#include <functional>

namespace aeonterracpp {

/**
 * @brief Parameters for heightmap generation
 */
struct HeightmapParams {
    // Resolution
    int width = 1024;
    int height = 512;
    
    // Basic parameters
    float oceanLevel = 0.0f;             // Sea level in km (0.0 means actual sea level)
    float maxElevation = 10.0f;          // Maximum elevation in km
    float minElevation = -10.0f;         // Minimum elevation in km (ocean depth)
    float continentalShelfDepth = -0.2f; // Depth of continental shelves in km
    
    // Noise parameters
    float noiseSeed = 0;                 // Seed for noise generation (0 for random)
    float noiseScale = 1.0f;             // Scale of noise features
    float noiseStrength = 0.5f;          // Strength of noise influence
    int noiseOctaves = 6;                // Number of noise octaves
    float noisePersistence = 0.5f;       // Persistence for noise octaves
    float noiseLacunarity = 2.0f;        // Lacunarity for noise octaves
    
    // Mountain parameters
    float mountainRoughness = 0.7f;      // Roughness of mountain terrain
    float plateauFlatness = 0.3f;        // Flatness of plateau regions
    
    // Erosion parameters
    int erosionIterations = 5;           // Number of erosion simulation iterations
    float erosionStrength = 0.3f;        // Strength of erosion effects
    float riverErosionFactor = 0.4f;     // How much rivers carve the landscape
    
    // Tectonic influence
    float tectonicInfluence = 0.7f;      // How much tectonic boundaries influence terrain
    float faultDisplacement = 0.2f;      // Displacement along tectonic faults
    
    // Climate influence
    float climateInfluence = 0.5f;       // How much climate affects terrain formation
    float precipitationErosion = 0.4f;   // Erosion strength due to precipitation
};

/**
 * @brief Class for generating heightmaps from planetary data
 */
class HeightmapGenerator {
public:
    /**
     * @brief Construct a new Heightmap Generator
     * 
     * @param planet Reference to the planet
     * @param tectonics Optional reference to a tectonic simulation (if null, tectonic features won't be used)
     */
    HeightmapGenerator(const Planet& planet, const TectonicSimulation* tectonics = nullptr);
    
    /**
     * @brief Set generation parameters
     * 
     * @param params The parameters to use
     */
    void setParameters(const HeightmapParams& params);
    
    /**
     * @brief Get current parameters
     * 
     * @return const HeightmapParams& The current parameters
     */
    const HeightmapParams& getParameters() const;
    
    /**
     * @brief Generate a heightmap
     * 
     * @param equirectangular If true, generates an equirectangular projection (default)
     *                        If false, generates equal-area projection
     * @return std::vector<float> The generated heightmap, row-major order
     */
    std::vector<float> generate(bool equirectangular = true);
    
    /**
     * @brief Generate a normal map from current heightmap data
     * 
     * @return std::vector<Vec3f> Normal vectors for each point, row-major order
     */
    std::vector<Vec3f> generateNormalMap();
    
    /**
     * @brief Generate a slope map from current heightmap data
     * 
     * @return std::vector<float> Slope values (0-1), row-major order
     */
    std::vector<float> generateSlopeMap();
    
    /**
     * @brief Save heightmap to file
     * 
     * @param filename The file path
     * @param format The format (e.g., "png", "raw", "tiff")
     * @return bool Success or failure
     */
    bool saveToFile(const std::string& filename, const std::string& format = "png");
    
    /**
     * @brief Get the width of the generated heightmap
     */
    int getWidth() const;
    
    /**
     * @brief Get the height of the generated heightmap
     */
    int getHeight() const;
    
    /**
     * @brief Get the raw heightmap data
     */
    const std::vector<float>& getData() const;
    
    /**
     * @brief Set a custom terrain modifier function
     * 
     * @param modifier Function that takes (lat, lon, currentHeight) and returns modified height
     */
    void setTerrainModifier(std::function<float(float, float, float)> modifier);
    
private:
    const Planet& planet;
    const TectonicSimulation* tectonics;
    HeightmapParams params;
    std::vector<float> heightmapData;
    int width;
    int height;
    std::mt19937 rng;
    std::function<float(float, float, float)> terrainModifier;
    
    // Helper methods
    float samplePlanetElevation(float lat, float lon) const;
    float generateNoise(float x, float y) const;
    void applyTerrainFeatures();
    void applyErosion();
    void applyTectonicFeatures();
    void applyClimateInfluence();
    void normalizeTerrain();
};

} // namespace aeonterracpp