#pragma once

#include "aeonterracpp/core/climate_system.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <array>

namespace aeonterracpp {

// Biome types based on climate and elevation
enum class BiomeType {
    // Forests
    TropicalRainforest,
    TemperateRainforest,
    TropicalSeasonalForest,
    TemperateDeciduousForest,
    ConiferousForest,
    
    // Grasslands
    Savanna,
    Grassland,
    Shrubland,
    
    // Deserts
    HotDesert,
    ColdDesert,
    
    // Tundra and Ice
    Tundra,
    IceSheet,
    
    // Wetlands
    Swamp,
    Marsh,
    
    // Mountains
    Mountain,
    HighMountain,
    
    // Special types
    VolcanicRegion,
    OceanicIsland,
    
    // Ocean types
    Ocean,
    CoastalWaters,
    DeepOcean,
    OceanTrench,
    
    // Count (must be last)
    Count
};

struct BiomeProperties {
    std::string name;
    float baseErosionRate;        // Base rate of erosion
    float vegetationDensity;      // 0-1 scale
    float soilDepth;              // Typical soil depth in meters
    float albedo;                 // Surface reflectivity
    std::string description;      // Human-readable description
    
    // Constructor with default values
    BiomeProperties(const std::string& _name = "Unknown", 
                    float _erosionRate = 0.5f,
                    float _vegDensity = 0.5f,
                    float _soilDepth = 1.0f,
                    float _albedo = 0.3f,
                    const std::string& _desc = "");
};

class BiomeSystem {
public:
    /**
     * @brief Construct a new Biome System
     * 
     * @param planet Reference to the planet
     * @param climate Reference to the climate system
     */
    BiomeSystem(Planet& planet, const ClimateSystem& climate);
    
    /**
     * @brief Initialize biome system
     */
    void initialize();
    
    /**
     * @brief Update biomes based on current climate
     * 
     * @param timeStep Time increment in million years
     */
    void update(double timeStep);
    
    /**
     * @brief Get biome type at a specific position
     * 
     * @param position Position on the planet surface
     * @return BiomeType The biome type at this position
     */
    BiomeType getBiomeType(const Vec3f& position) const;
    
    /**
     * @brief Get biome properties at a specific position
     * 
     * @param position Position on the planet surface
     * @return const BiomeProperties& Properties of the biome
     */
    const BiomeProperties& getBiomeProperties(const Vec3f& position) const;
    
    /**
     * @brief Get vegetation density at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Vegetation density (0-1)
     */
    float getVegetationDensity(const Vec3f& position) const;
    
    /**
     * @brief Classify biomes based on climate and elevation
     */
    void classifyBiomes();
    
    /**
     * @brief Calculate soil properties based on climate and biome
     */
    void calculateSoilProperties();
    
    /**
     * @brief Calculate vegetation cover based on climate and biome
     */
    void calculateVegetationCover();
    
    /**
     * @brief Get biome data
     * 
     * @return const std::vector<BiomeType>& Biome types at each vertex
     */
    const std::vector<BiomeType>& getBiomeData() const;
    
    /**
     * @brief Get vegetation data
     * 
     * @return const std::vector<float>& Vegetation density at each vertex
     */
    const std::vector<float>& getVegetationData() const;
    
    /**
     * @brief Get soil depth data
     * 
     * @return const std::vector<float>& Soil depth at each vertex
     */
    const std::vector<float>& getSoilDepthData() const;
    
    /**
     * @brief Get biome name from type
     * 
     * @param type Biome type
     * @return std::string Name of the biome
     */
    std::string getBiomeName(BiomeType type) const;
    
    /**
     * @brief Get biome color for visualization
     * 
     * @param type Biome type
     * @return std::array<float, 3> RGB color values (0-1)
     */
    std::array<float, 3> getBiomeColor(BiomeType type) const;
    
private:
    Planet& planet;
    const ClimateSystem& climate;
    
    // Biome data
    std::vector<BiomeType> biomes;
    std::vector<float> vegetation;
    std::vector<float> soilDepth;
    
    // Biome properties lookup
    std::array<BiomeProperties, static_cast<size_t>(BiomeType::Count)> biomeProperties;
    
    // Biome colors for visualization (RGB)
    std::array<std::array<float, 3>, static_cast<size_t>(BiomeType::Count)> biomeColors;
    
    // Initialize biome properties
    void initializeBiomeProperties();
    
    // Initialize biome colors
    void initializeBiomeColors();
    
    // Helper method to determine biome from climate data
    BiomeType classifyBiomeFromClimate(float temperature, float precipitation, 
                                      float elevation, int koeppenZone) const;
};

} // namespace aeonterracpp