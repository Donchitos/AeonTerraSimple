#include "aeonterracpp/core/biome_climate_feedback.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace aeonterracpp {

BiomeClimateFeedback::BiomeClimateFeedback(Planet& planet, 
                                         ClimateSystem& climate,
                                         BiomeSystem& biomes)
    : planet(planet), climate(climate), biomes(biomes) {
    // Initialize with default parameters
    initialize();
}

void BiomeClimateFeedback::initialize() {
    // Set default feedback parameters
    params.albedoFeedbackStrength = 1.0f;
    params.evapotranspirationStrength = 1.0f;
    params.roughnessLengthEffect = 1.0f;
    params.vegetationGrowthRate = 1.0f;
    params.vegetationDiebackRate = 1.0f;
    params.carbonSequestrationRate = 1.0f;
    params.localWarmingEffect = 1.0f;
    
    // Initialize data buffers with the size of the planet's vertex count
    size_t vertexCount = planet.getVertexCount();
    albedoModifier.resize(vertexCount, 0.0f);
    evapotranspirationRate.resize(vertexCount, 0.0f);
    roughnessLength.resize(vertexCount, 0.0f);
    sequesteredCarbon.resize(vertexCount, 0.0f);
    
    // Initialize history tracking
    albedoHistory.clear();
    temperatureHistory.clear();
    vegetationHistory.clear();
    
    std::cout << "Biome-climate feedback system initialized with " 
              << vertexCount << " vertices." << std::endl;
}

void BiomeClimateFeedback::setParameters(const BiomeClimateFeedbackParams& params) {
    this->params = params;
}

const BiomeClimateFeedbackParams& BiomeClimateFeedback::getParameters() const {
    return params;
}

void BiomeClimateFeedback::update(double timeStep, float dayOfYear) {
    // Save current state for history
    recordCurrentState();
    
    // Calculate current biome effects on climate
    calculateAlbedoEffects();
    calculateEvapotranspirationEffects();
    calculateRoughnessEffects();
    calculateCarbonSequestration(timeStep);
    
    // Apply the feedback effects to the climate system
    applyFeedbackToClimate();
    
    // Track changes in biomes due to climate changes
    trackBiomeChanges(timeStep);
    
    std::cout << "Updated biome-climate feedbacks for " << timeStep << " million years." << std::endl;
}

void BiomeClimateFeedback::calculateAlbedoEffects() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < albedoModifier.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells (ocean albedo is handled directly by climate system)
        if (planet.getElevation(vertexIdx) < 0.0f) {
            albedoModifier[i] = 0.0f;
            continue;
        }
        
        // Get position and biome data
        Vec3f pos = grid.getVertex(vertexIdx);
        BiomeType biomeType = biomes.getBiomeType(pos);
        const BiomeProperties& props = biomes.getBiomeProperties(pos);
        float vegetationDensity = biomes.getVegetationDensity(pos);
        
        // Get base albedo from biome properties
        float baseAlbedo = props.albedo;
        
        // Adjust for seasonal effects (leaf drop in deciduous forests, snow cover, etc.)
        // In a real implementation, this would be more sophisticated and use climate data
        float seasonalFactor = 1.0f;
        
        // Modify albedo based on vegetation density
        // Dense vegetation typically has lower albedo than bare soil
        float vegetationAlbedo = baseAlbedo * (0.8f + 0.2f * (1.0f - vegetationDensity));
        
        // Calculate albedo modifier relative to default climate system values
        // Climate system typically uses around 0.3 for average land albedo
        float modifierValue = vegetationAlbedo - 0.3f;
        
        // Apply feedback strength parameter
        albedoModifier[i] = modifierValue * params.albedoFeedbackStrength;
    }
}

void BiomeClimateFeedback::calculateEvapotranspirationEffects() {
    const SphericalGrid& grid = planet.getGrid();
    const HydrologicalSystem& hydrology = climate.getHydrology();
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    
    for (size_t i = 0; i < evapotranspirationRate.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            evapotranspirationRate[i] = 0.0f;
            continue;
        }
        
        // Get position and biome data
        Vec3f pos = grid.getVertex(vertexIdx);
        BiomeType biomeType = biomes.getBiomeType(pos);
        float vegetationDensity = biomes.getVegetationDensity(pos);
        
        // Get climate data
        float precipitation = hydrology.getPrecipitation(pos);
        float temperature = atmosphere.getTemperature(pos);
        
        // Base evapotranspiration rate depends on vegetation type and density
        float baseRate = 0.0f;
        
        switch (biomeType) {
            case BiomeType::TropicalRainforest:
                baseRate = 1500.0f; // mm/year
                break;
            case BiomeType::TemperateRainforest:
                baseRate = 1000.0f;
                break;
            case BiomeType::TropicalSeasonalForest:
                baseRate = 1200.0f;
                break;
            case BiomeType::TemperateDeciduousForest:
                baseRate = 800.0f;
                break;
            case BiomeType::ConiferousForest:
                baseRate = 600.0f;
                break;
            case BiomeType::Savanna:
                baseRate = 700.0f;
                break;
            case BiomeType::Grassland:
                baseRate = 500.0f;
                break;
            case BiomeType::Shrubland:
                baseRate = 400.0f;
                break;
            case BiomeType::HotDesert:
            case BiomeType::ColdDesert:
                baseRate = 100.0f;
                break;
            case BiomeType::Tundra:
                baseRate = 200.0f;
                break;
            case BiomeType::IceSheet:
                baseRate = 50.0f;
                break;
            default:
                baseRate = 500.0f; // Default value
        }
        
        // Scale by vegetation density
        baseRate *= vegetationDensity;
        
        // Adjust for available moisture (limited by precipitation)
        // Plants can't transpire more water than is available
        baseRate = std::min(baseRate, precipitation * 0.8f);
        
        // Adjust for temperature (reduced at very low temperatures)
        if (temperature < 5.0f) {
            baseRate *= std::max(0.1f, temperature / 5.0f);
        }
        
        // Apply feedback strength parameter
        evapotranspirationRate[i] = baseRate * params.evapotranspirationStrength;
    }
}

void BiomeClimateFeedback::calculateRoughnessEffects() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < roughnessLength.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            roughnessLength[i] = 0.0001f; // Default ocean roughness (smooth)
            continue;
        }
        
        // Get position and biome data
        Vec3f pos = grid.getVertex(vertexIdx);
        BiomeType biomeType = biomes.getBiomeType(pos);
        float vegetationDensity = biomes.getVegetationDensity(pos);
        
        // Base roughness length depends on vegetation type (meters)
        float baseRoughness = 0.0f;
        
        switch (biomeType) {
            case BiomeType::TropicalRainforest:
            case BiomeType::TemperateRainforest:
                baseRoughness = 2.0f; // Tall, dense forest
                break;
            case BiomeType::TropicalSeasonalForest:
            case BiomeType::TemperateDeciduousForest:
            case BiomeType::ConiferousForest:
                baseRoughness = 1.5f; // Forest
                break;
            case BiomeType::Savanna:
                baseRoughness = 0.5f; // Scattered trees
                break;
            case BiomeType::Grassland:
                baseRoughness = 0.1f; // Short grass
                break;
            case BiomeType::Shrubland:
                baseRoughness = 0.3f; // Bushes
                break;
            case BiomeType::HotDesert:
            case BiomeType::ColdDesert:
                baseRoughness = 0.01f; // Bare ground
                break;
            case BiomeType::Tundra:
                baseRoughness = 0.05f; // Low vegetation
                break;
            case BiomeType::IceSheet:
                baseRoughness = 0.001f; // Smooth ice
                break;
            default:
                baseRoughness = 0.1f; // Default value
        }
        
        // Scale by vegetation density
        baseRoughness *= 0.2f + 0.8f * vegetationDensity;
        
        // Apply feedback strength parameter
        roughnessLength[i] = baseRoughness * params.roughnessLengthEffect;
    }
}

void BiomeClimateFeedback::calculateCarbonSequestration(double timeStep) {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < sequesteredCarbon.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells (ocean carbon sequestration is handled separately)
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get position and biome data
        Vec3f pos = grid.getVertex(vertexIdx);
        BiomeType biomeType = biomes.getBiomeType(pos);
        float vegetationDensity = biomes.getVegetationDensity(pos);
        
        // Base carbon sequestration rate depends on vegetation type (tons/hectare/year)
        float baseSequestrationRate = 0.0f;
        
        switch (biomeType) {
            case BiomeType::TropicalRainforest:
                baseSequestrationRate = 5.0f;
                break;
            case BiomeType::TemperateRainforest:
                baseSequestrationRate = 4.0f;
                break;
            case BiomeType::TropicalSeasonalForest:
                baseSequestrationRate = 3.5f;
                break;
            case BiomeType::TemperateDeciduousForest:
                baseSequestrationRate = 3.0f;
                break;
            case BiomeType::ConiferousForest:
                baseSequestrationRate = 2.5f;
                break;
            case BiomeType::Savanna:
                baseSequestrationRate = 1.5f;
                break;
            case BiomeType::Grassland:
                baseSequestrationRate = 1.0f;
                break;
            case BiomeType::Shrubland:
                baseSequestrationRate = 0.8f;
                break;
            case BiomeType::HotDesert:
            case BiomeType::ColdDesert:
                baseSequestrationRate = 0.1f;
                break;
            case BiomeType::Tundra:
                baseSequestrationRate = 0.3f;
                break;
            case BiomeType::IceSheet:
                baseSequestrationRate = 0.0f;
                break;
            default:
                baseSequestrationRate = 0.5f; // Default value
        }
        
        // Scale by vegetation density
        baseSequestrationRate *= vegetationDensity;
        
        // Convert time step from million years to years
        float yearlySequestration = baseSequestrationRate * params.carbonSequestrationRate;
        float totalSequestration = yearlySequestration * static_cast<float>(timeStep * 1e6);
        
        // Add to accumulated sequestered carbon
        sequesteredCarbon[i] += totalSequestration;
    }
}

void BiomeClimateFeedback::applyFeedbackToClimate() {
    // In a real implementation, this would directly modify climate system parameters
    // For now, we'll just print the feedback effects
    
    // Calculate average feedback effects across the planet
    float avgAlbedoModifier = 0.0f;
    float avgEvapotranspiration = 0.0f;
    float avgRoughness = 0.0f;
    float totalSequestered = 0.0f;
    int landCount = 0;
    
    for (size_t i = 0; i < albedoModifier.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only count land cells
        if (planet.getElevation(vertexIdx) >= 0.0f) {
            avgAlbedoModifier += albedoModifier[i];
            avgEvapotranspiration += evapotranspirationRate[i];
            avgRoughness += roughnessLength[i];
            totalSequestered += sequesteredCarbon[i];
            landCount++;
        }
    }
    
    if (landCount > 0) {
        avgAlbedoModifier /= landCount;
        avgEvapotranspiration /= landCount;
        avgRoughness /= landCount;
    }
    
    std::cout << "Applied biome-climate feedbacks:" << std::endl;
    std::cout << "  Average albedo modifier: " << avgAlbedoModifier << std::endl;
    std::cout << "  Average evapotranspiration: " << avgEvapotranspiration << " mm/year" << std::endl;
    std::cout << "  Average roughness length: " << avgRoughness << " m" << std::endl;
    std::cout << "  Total carbon sequestered: " << totalSequestered << " tons" << std::endl;
    
    // Here we would apply these effects to the climate system
    // This would include modifying temperature patterns, precipitation,
    // wind patterns, and other climate variables
}

void BiomeClimateFeedback::trackBiomeChanges(double timeStep) {
    // In a real implementation, this would analyze climate changes and predict biome responses
    // For now, we'll just track a few key metrics
    
    // Count biomes by type
    std::vector<int> biomeCounts(static_cast<size_t>(BiomeType::Count), 0);
    
    for (size_t i = 0; i < albedoModifier.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get position and biome data
        Vec3f pos = grid.getVertex(vertexIdx);
        BiomeType biomeType = biomes.getBiomeType(pos);
        
        // Count this biome type
        biomeCounts[static_cast<size_t>(biomeType)]++;
    }
    
    // Print biome distribution
    std::cout << "Current biome distribution:" << std::endl;
    for (size_t i = 0; i < biomeCounts.size(); ++i) {
        if (biomeCounts[i] > 0) {
            BiomeType type = static_cast<BiomeType>(i);
            std::string name = biomes.getBiomeName(type);
            std::cout << "  " << name << ": " << biomeCounts[i] << " cells" << std::endl;
        }
    }
}

void BiomeClimateFeedback::recordCurrentState() {
    // Record current state for history tracking
    const SphericalGrid& grid = planet.getGrid();
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    
    // Skip if we don't have enough data yet
    if (albedoModifier.empty() || roughnessLength.empty()) {
        return;
    }
    
    // Calculate average values to store in history
    float avgAlbedo = 0.0f;
    float avgTemperature = 0.0f;
    float avgVegetation = 0.0f;
    int landCount = 0;
    
    for (size_t i = 0; i < albedoModifier.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only count land cells
        if (planet.getElevation(vertexIdx) >= 0.0f) {
            // Get position and data
            Vec3f pos = grid.getVertex(vertexIdx);
            
            // Albedo (convert modifier to absolute value)
            avgAlbedo += 0.3f + albedoModifier[i];
            
            // Temperature
            avgTemperature += atmosphere.getTemperature(pos);
            
            // Vegetation density
            avgVegetation += biomes.getVegetationDensity(pos);
            
            landCount++;
        }
    }
    
    if (landCount > 0) {
        avgAlbedo /= landCount;
        avgTemperature /= landCount;
        avgVegetation /= landCount;
    }
    
    // Store in history
    albedoHistory.push_back(avgAlbedo);
    temperatureHistory.push_back(avgTemperature);
    vegetationHistory.push_back(avgVegetation);
}

const std::vector<float>& BiomeClimateFeedback::getAlbedoHistory() const {
    return albedoHistory;
}

const std::vector<float>& BiomeClimateFeedback::getTemperatureHistory() const {
    return temperatureHistory;
}

const std::vector<float>& BiomeClimateFeedback::getVegetationHistory() const {
    return vegetationHistory;
}

float BiomeClimateFeedback::getAlbedoModifier(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(albedoModifier.size())) {
        return albedoModifier[vertexIdx];
    }
    
    return 0.0f;
}

float BiomeClimateFeedback::getEvapotranspirationRate(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(evapotranspirationRate.size())) {
        return evapotranspirationRate[vertexIdx];
    }
    
    return 0.0f;
}

float BiomeClimateFeedback::getRoughnessLength(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(roughnessLength.size())) {
        return roughnessLength[vertexIdx];
    }
    
    return 0.0f;
}

float BiomeClimateFeedback::getSequesteredCarbon(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(sequesteredCarbon.size())) {
        return sequesteredCarbon[vertexIdx];
    }
    
    return 0.0f;
}

} // namespace aeonterracpp