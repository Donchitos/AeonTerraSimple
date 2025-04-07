#include "aeonterracpp/core/biome_system.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace aeonterracpp {

// BiomeProperties constructor
BiomeProperties::BiomeProperties(const std::string& _name, 
                               float _erosionRate,
                               float _vegDensity,
                               float _soilDepth,
                               float _albedo,
                               const std::string& _desc)
    : name(_name),
      baseErosionRate(_erosionRate),
      vegetationDensity(_vegDensity),
      soilDepth(_soilDepth),
      albedo(_albedo),
      description(_desc) {
}

BiomeSystem::BiomeSystem(Planet& planet, const ClimateSystem& climate)
    : planet(planet), climate(climate) {
    // Initialize data vectors with the size of the planet's vertex count
    size_t vertexCount = planet.getVertexCount();
    biomes.resize(vertexCount, BiomeType::Ocean);
    vegetation.resize(vertexCount, 0.0f);
    soilDepth.resize(vertexCount, 0.0f);
    
    // Initialize biome properties
    initializeBiomeProperties();
    
    // Initialize biome colors
    initializeBiomeColors();
}

void BiomeSystem::initialize() {
    std::cout << "Initializing biome system..." << std::endl;
    
    // Classify biomes based on climate
    classifyBiomes();
    
    // Calculate soil properties
    calculateSoilProperties();
    
    // Calculate vegetation cover
    calculateVegetationCover();
    
    std::cout << "Biome system initialized successfully." << std::endl;
}

void BiomeSystem::update(double timeStep) {
    // Reclassify biomes based on updated climate
    classifyBiomes();
    
    // Update soil properties
    calculateSoilProperties();
    
    // Update vegetation cover with gradual changes
    // Vegetation doesn't immediately match climate - it takes time to grow or die off
    for (size_t i = 0; i < vegetation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get target vegetation density for this biome
        BiomeType biomeType = biomes[i];
        float targetDensity = biomeProperties[static_cast<size_t>(biomeType)].vegetationDensity;
        
        // Adjust current vegetation towards target with a time lag
        // Slower for growing, faster for dying off
        float currentDensity = vegetation[i];
        float growthRate = (targetDensity > currentDensity) ? 
                           0.2f * static_cast<float>(timeStep) : 
                           0.4f * static_cast<float>(timeStep);
        
        vegetation[i] = currentDensity + (targetDensity - currentDensity) * growthRate;
    }
}

BiomeType BiomeSystem::getBiomeType(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(biomes.size())) {
        return biomes[vertexIdx];
    }
    
    return BiomeType::Ocean; // Default return
}

const BiomeProperties& BiomeSystem::getBiomeProperties(const Vec3f& position) const {
    BiomeType type = getBiomeType(position);
    return biomeProperties[static_cast<size_t>(type)];
}

float BiomeSystem::getVegetationDensity(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(vegetation.size())) {
        return vegetation[vertexIdx];
    }
    
    return 0.0f; // Default return
}

void BiomeSystem::classifyBiomes() {
    const SphericalGrid& grid = planet.getGrid();
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    const HydrologicalSystem& hydrology = climate.getHydrology();
    
    for (size_t i = 0; i < biomes.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Get elevation
        float elevation = planet.getElevation(vertexIdx);
        
        // Check if this is ocean
        if (elevation < 0.0f) {
            // Classify ocean types based on depth
            if (elevation < -8.0f) {
                biomes[i] = BiomeType::OceanTrench;
            } else if (elevation < -4.0f) {
                biomes[i] = BiomeType::DeepOcean;
            } else if (elevation < -0.1f) {
                biomes[i] = BiomeType::Ocean;
            } else {
                biomes[i] = BiomeType::CoastalWaters;
            }
            continue;
        }
        
        // Get climate data
        float temperature = atmosphere.getTemperature(pos);
        float precipitation = hydrology.getPrecipitation(pos);
        
        // Get Köppen climate zone (simplified for this example)
        // In a real implementation, this would come from climate.getClimateZone(pos)
        int koeppenZone = 0;
        if (temperature > 18.0f) {
            koeppenZone = 0; // Tropical (A)
        } else if (precipitation < 500.0f) {
            koeppenZone = 1; // Arid (B)
        } else if (temperature > 0.0f) {
            koeppenZone = 2; // Temperate (C)
        } else if (temperature > -10.0f) {
            koeppenZone = 3; // Continental (D)
        } else {
            koeppenZone = 4; // Polar (E)
        }
        
        // Classify biome based on climate and elevation
        biomes[i] = classifyBiomeFromClimate(temperature, precipitation, elevation, koeppenZone);
    }
}

BiomeType BiomeSystem::classifyBiomeFromClimate(float temperature, float precipitation, 
                                               float elevation, int koeppenZone) const {
    // Check for high mountains first
    if (elevation > 3.0f) {
        return BiomeType::HighMountain;
    } else if (elevation > 1.5f) {
        return BiomeType::Mountain;
    }
    
    // Classify based on temperature and precipitation
    if (temperature > 18.0f) {
        // Tropical climates
        if (precipitation > 2000.0f) {
            return BiomeType::TropicalRainforest;
        } else if (precipitation > 1000.0f) {
            return BiomeType::TropicalSeasonalForest;
        } else if (precipitation > 500.0f) {
            return BiomeType::Savanna;
        } else {
            return BiomeType::HotDesert;
        }
    } else if (temperature > 5.0f) {
        // Temperate climates
        if (precipitation > 2000.0f) {
            return BiomeType::TemperateRainforest;
        } else if (precipitation > 1000.0f) {
            return BiomeType::TemperateDeciduousForest;
        } else if (precipitation > 500.0f) {
            return BiomeType::Grassland;
        } else if (precipitation > 250.0f) {
            return BiomeType::Shrubland;
        } else {
            return BiomeType::ColdDesert;
        }
    } else if (temperature > -5.0f) {
        // Boreal/Subarctic climates
        if (precipitation > 600.0f) {
            return BiomeType::ConiferousForest;
        } else if (precipitation > 200.0f) {
            return BiomeType::Tundra;
        } else {
            return BiomeType::ColdDesert;
        }
    } else {
        // Polar climates
        return BiomeType::IceSheet;
    }
}

void BiomeSystem::calculateSoilProperties() {
    for (size_t i = 0; i < soilDepth.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            soilDepth[i] = 0.0f;
            continue;
        }
        
        // Get biome properties
        const BiomeProperties& props = biomeProperties[static_cast<size_t>(biomes[i])];
        
        // Base soil depth from biome
        soilDepth[i] = props.soilDepth;
        
        // Adjust for slope - steeper slopes have thinner soil
        const SphericalGrid& grid = planet.getGrid();
        Vec3f pos = grid.getVertex(vertexIdx);
        float maxSlope = 0.0f;
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            Vec3f neighborPos = grid.getVertex(neighborIdx);
            float elevationDiff = std::abs(planet.getElevation(vertexIdx) - planet.getElevation(neighborIdx));
            float distance = (neighborPos - pos).norm();
            float slope = elevationDiff / distance;
            maxSlope = std::max(maxSlope, slope);
        }
        
        // Reduce soil depth on steep slopes
        if (maxSlope > 0.1f) {
            soilDepth[i] *= std::max(0.1f, 1.0f - maxSlope * 5.0f);
        }
    }
}

void BiomeSystem::calculateVegetationCover() {
    const SphericalGrid& grid = planet.getGrid();
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    const HydrologicalSystem& hydrology = climate.getHydrology();
    
    for (size_t i = 0; i < vegetation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            vegetation[i] = 0.0f;
            continue;
        }
        
        // Get biome properties
        const BiomeProperties& props = biomeProperties[static_cast<size_t>(biomes[i])];
        
        // Start with base vegetation density for this biome
        float vegDensity = props.vegetationDensity;
        
        // Adjust based on local climate conditions
        Vec3f pos = grid.getVertex(vertexIdx);
        float temperature = atmosphere.getTemperature(pos);
        float precipitation = hydrology.getPrecipitation(pos);
        
        // Reduce vegetation in extreme temperatures
        if (temperature < -10.0f || temperature > 35.0f) {
            float tempFactor = 1.0f - std::min(1.0f, (std::abs(temperature) - 10.0f) / 25.0f);
            vegDensity *= tempFactor;
        }
        
        // Reduce vegetation in very dry areas
        if (precipitation < 250.0f) {
            float precipFactor = precipitation / 250.0f;
            vegDensity *= precipFactor;
        }
        
        // Adjust for slope - very steep slopes have less vegetation
        float maxSlope = 0.0f;
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            Vec3f neighborPos = grid.getVertex(neighborIdx);
            float elevationDiff = std::abs(planet.getElevation(vertexIdx) - planet.getElevation(neighborIdx));
            float distance = (neighborPos - pos).norm();
            float slope = elevationDiff / distance;
            maxSlope = std::max(maxSlope, slope);
        }
        
        // Reduce vegetation on steep slopes
        if (maxSlope > 0.2f) {
            vegDensity *= std::max(0.2f, 1.0f - (maxSlope - 0.2f) * 2.0f);
        }
        
        vegetation[i] = vegDensity;
    }
}

const std::vector<BiomeType>& BiomeSystem::getBiomeData() const {
    return biomes;
}

const std::vector<float>& BiomeSystem::getVegetationData() const {
    return vegetation;
}

const std::vector<float>& BiomeSystem::getSoilDepthData() const {
    return soilDepth;
}

std::string BiomeSystem::getBiomeName(BiomeType type) const {
    return biomeProperties[static_cast<size_t>(type)].name;
}

std::array<float, 3> BiomeSystem::getBiomeColor(BiomeType type) const {
    return biomeColors[static_cast<size_t>(type)];
}

void BiomeSystem::initializeBiomeProperties() {
    size_t idx = 0;
    
    // Forests
    biomeProperties[idx++] = BiomeProperties("Tropical Rainforest", 0.3f, 0.95f, 2.0f, 0.12f, 
                                            "Dense evergreen forest with high biodiversity");
    
    biomeProperties[idx++] = BiomeProperties("Temperate Rainforest", 0.35f, 0.9f, 1.8f, 0.14f, 
                                            "Coniferous or broadleaf forest with high rainfall");
    
    biomeProperties[idx++] = BiomeProperties("Tropical Seasonal Forest", 0.4f, 0.8f, 1.5f, 0.15f, 
                                            "Forest with wet and dry seasons");
    
    biomeProperties[idx++] = BiomeProperties("Temperate Deciduous Forest", 0.45f, 0.85f, 1.2f, 0.18f, 
                                            "Forest that sheds leaves seasonally");
    
    biomeProperties[idx++] = BiomeProperties("Coniferous Forest", 0.4f, 0.8f, 0.8f, 0.15f, 
                                            "Forest dominated by cone-bearing trees");
    
    // Grasslands
    biomeProperties[idx++] = BiomeProperties("Savanna", 0.6f, 0.5f, 1.0f, 0.2f, 
                                            "Grassy plains with scattered trees");
    
    biomeProperties[idx++] = BiomeProperties("Grassland", 0.7f, 0.6f, 1.2f, 0.25f, 
                                            "Open grassland with few trees");
    
    biomeProperties[idx++] = BiomeProperties("Shrubland", 0.65f, 0.4f, 0.7f, 0.25f, 
                                            "Area dominated by shrubs");
    
    // Deserts
    biomeProperties[idx++] = BiomeProperties("Hot Desert", 0.9f, 0.05f, 0.2f, 0.4f, 
                                            "Arid region with little vegetation");
    
    biomeProperties[idx++] = BiomeProperties("Cold Desert", 0.8f, 0.1f, 0.3f, 0.35f, 
                                            "Arid region with cold temperatures");
    
    // Tundra and Ice
    biomeProperties[idx++] = BiomeProperties("Tundra", 0.5f, 0.3f, 0.4f, 0.35f, 
                                            "Cold treeless region with permafrost");
    
    biomeProperties[idx++] = BiomeProperties("Ice Sheet", 0.1f, 0.0f, 0.0f, 0.8f, 
                                            "Permanent ice cover");
    
    // Wetlands
    biomeProperties[idx++] = BiomeProperties("Swamp", 0.2f, 0.9f, 3.0f, 0.1f, 
                                            "Forested wetland");
    
    biomeProperties[idx++] = BiomeProperties("Marsh", 0.25f, 0.8f, 2.5f, 0.11f, 
                                            "Wetland dominated by herbaceous plants");
    
    // Mountains
    biomeProperties[idx++] = BiomeProperties("Mountain", 0.8f, 0.4f, 0.5f, 0.3f, 
                                            "Elevated terrain with various vegetation zones");
    
    biomeProperties[idx++] = BiomeProperties("High Mountain", 0.95f, 0.1f, 0.1f, 0.6f, 
                                            "High elevation terrain with sparse vegetation");
    
    // Special types
    biomeProperties[idx++] = BiomeProperties("Volcanic Region", 0.7f, 0.2f, 0.5f, 0.25f, 
                                            "Area with active or recent volcanic activity");
    
    biomeProperties[idx++] = BiomeProperties("Oceanic Island", 0.5f, 0.7f, 0.8f, 0.2f, 
                                            "Island with unique ecosystem");
    
    // Ocean types
    biomeProperties[idx++] = BiomeProperties("Ocean", 0.0f, 0.0f, 0.0f, 0.06f, 
                                            "Open ocean");
    
    biomeProperties[idx++] = BiomeProperties("Coastal Waters", 0.0f, 0.1f, 0.0f, 0.1f, 
                                            "Shallow coastal waters");
    
    biomeProperties[idx++] = BiomeProperties("Deep Ocean", 0.0f, 0.0f, 0.0f, 0.05f, 
                                            "Deep ocean waters");
    
    biomeProperties[idx++] = BiomeProperties("Ocean Trench", 0.0f, 0.0f, 0.0f, 0.04f, 
                                            "Very deep ocean trench");
}

void BiomeSystem::initializeBiomeColors() {
    size_t idx = 0;
    
    // Forests (greens)
    biomeColors[idx++] = {0.0f, 0.6f, 0.0f};      // Tropical Rainforest - dark green
    biomeColors[idx++] = {0.0f, 0.5f, 0.2f};      // Temperate Rainforest - dark blue-green
    biomeColors[idx++] = {0.2f, 0.7f, 0.2f};      // Tropical Seasonal Forest - medium green
    biomeColors[idx++] = {0.4f, 0.7f, 0.1f};      // Temperate Deciduous Forest - yellowish green
    biomeColors[idx++] = {0.0f, 0.4f, 0.2f};      // Coniferous Forest - dark blue-green
    
    // Grasslands (yellows and light greens)
    biomeColors[idx++] = {0.8f, 0.7f, 0.2f};      // Savanna - tan/yellow
    biomeColors[idx++] = {0.8f, 0.8f, 0.4f};      // Grassland - light yellow-green
    biomeColors[idx++] = {0.7f, 0.7f, 0.3f};      // Shrubland - olive green
    
    // Deserts (tans and browns)
    biomeColors[idx++] = {0.9f, 0.8f, 0.5f};      // Hot Desert - tan
    biomeColors[idx++] = {0.8f, 0.7f, 0.6f};      // Cold Desert - light brown
    
    // Tundra and Ice (whites and light blues)
    biomeColors[idx++] = {0.8f, 0.8f, 0.7f};      // Tundra - light tan/gray
    biomeColors[idx++] = {0.95f, 0.95f, 0.95f};   // Ice Sheet - white
    
    // Wetlands (blue-greens)
    biomeColors[idx++] = {0.2f, 0.5f, 0.4f};      // Swamp - dark blue-green
    biomeColors[idx++] = {0.4f, 0.6f, 0.5f};      // Marsh - medium blue-green
    
    // Mountains (grays and browns)
    biomeColors[idx++] = {0.6f, 0.6f, 0.6f};      // Mountain - gray
    biomeColors[idx++] = {0.8f, 0.8f, 0.8f};      // High Mountain - light gray
    
    // Special types
    biomeColors[idx++] = {0.6f, 0.3f, 0.3f};      // Volcanic Region - reddish brown
    biomeColors[idx++] = {0.3f, 0.7f, 0.3f};      // Oceanic Island - green
    
    // Ocean types (blues)
    biomeColors[idx++] = {0.2f, 0.2f, 0.8f};      // Ocean - medium blue
    biomeColors[idx++] = {0.3f, 0.3f, 0.9f};      // Coastal Waters - light blue
    biomeColors[idx++] = {0.1f, 0.1f, 0.6f};      // Deep Ocean - dark blue
    biomeColors[idx++] = {0.0f, 0.0f, 0.4f};      // Ocean Trench - very dark blue
}

} // namespace aeonterracpp