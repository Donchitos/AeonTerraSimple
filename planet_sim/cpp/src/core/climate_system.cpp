#include "aeonterracpp/core/climate_system.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace aeonterracpp {

ClimateSystem::ClimateSystem(Planet& planet)
    : planet(planet), currentDayOfYear(182.0f), rng(std::random_device()()) {
    // Initialize climate parameters with Earth-like defaults
    params = ClimateParams();
    
    // Initialize climate zones
    climateZones.resize(planet.getVertexCount(), 0);
}

void ClimateSystem::setParameters(const ClimateParams& params) {
    this->params = params;
}

const ClimateParams& ClimateSystem::getParameters() const {
    return params;
}

void ClimateSystem::initialize() {
    std::cout << "Initializing climate system..." << std::endl;
    
    // Create component systems
    atmosphere = std::make_unique<AtmosphericCirculation>(planet, params);
    atmosphere->initialize();
    
    ocean = std::make_unique<OceanicCirculation>(planet, *atmosphere, params);
    ocean->initialize();
    
    hydrology = std::make_unique<HydrologicalSystem>(planet, *atmosphere, *ocean, params);
    hydrology->initialize();
    
    // Initialize climate zones
    initializeClimateZones();
    
    std::cout << "Climate system initialized successfully." << std::endl;
}

void ClimateSystem::simulate(double timeStep, float dayOfYear) {
    // Update day of year
    currentDayOfYear = dayOfYear;
    
    // Simulate atmospheric circulation
    atmosphere->simulate(timeStep, dayOfYear);
    
    // Simulate oceanic circulation
    ocean->simulate(timeStep, dayOfYear);
    
    // Simulate hydrological system
    hydrology->simulate(timeStep, dayOfYear);
    
    // Update climate zones
    calculateClimateZones();
    
    // Simulate weathering and erosion based on climate
    simulateWeathering(timeStep);
}

const AtmosphericCirculation& ClimateSystem::getAtmosphere() const {
    return *atmosphere;
}

const OceanicCirculation& ClimateSystem::getOcean() const {
    return *ocean;
}

const HydrologicalSystem& ClimateSystem::getHydrology() const {
    return *hydrology;
}

void ClimateSystem::initializeClimateZones() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < climateZones.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Simple initial classification based on latitude
        // 0: Tropical (0-23.5°)
        // 1: Subtropical (23.5-35°)
        // 2: Temperate (35-55°)
        // 3: Subpolar (55-66.5°)
        // 4: Polar (66.5-90°)
        
        if (absLat < 23.5f) {
            climateZones[i] = 0; // Tropical
        } else if (absLat < 35.0f) {
            climateZones[i] = 1; // Subtropical
        } else if (absLat < 55.0f) {
            climateZones[i] = 2; // Temperate
        } else if (absLat < 66.5f) {
            climateZones[i] = 3; // Subpolar
        } else {
            climateZones[i] = 4; // Polar
        }
    }
}

void ClimateSystem::calculateClimateZones() {
    // Implement Köppen climate classification
    // A: Tropical
    // B: Arid
    // C: Temperate
    // D: Continental
    // E: Polar
    
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<float>& temps = atmosphere->getTemperatureData();
    const std::vector<float>& precips = hydrology->getPrecipitationData();
    
    // Climate zones encoded as:
    // 0-4: Major classes (A=0, B=1, C=2, D=3, E=4)
    // 5-14: A subclasses (Af=5, Am=6, Aw=7)
    // 15-24: B subclasses (BWh=15, BWk=16, BSh=17, BSk=18)
    // 25-34: C subclasses (Cfa=25, Cfb=26, Cfc=27, Csa=28, Csb=29, Csc=30, Cwa=31, Cwb=32, Cwc=33)
    // 35-44: D subclasses (Dfa=35, Dfb=36, Dfc=37, Dfd=38, Dsa=39, Dsb=40, Dsc=41, Dsd=42, Dwa=43, Dwb=44, Dwc=45, Dwd=46)
    // 45-49: E subclasses (ET=47, EF=48)
    
    for (size_t i = 0; i < climateZones.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Skip ocean cells (simplification)
        if (planet.getElevation(vertexIdx) < 0.0f) {
            climateZones[i] = 50; // Ocean
            continue;
        }
        
        // Get temperature and precipitation
        float annualTemp = 15.0f; // Default
        if (vertexIdx < static_cast<int>(temps.size())) {
            annualTemp = temps[vertexIdx];
        }
        
        float annualPrecip = 1000.0f; // Default
        if (vertexIdx < static_cast<int>(precips.size())) {
            annualPrecip = precips[vertexIdx];
        }
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Major classification
        
        // Polar climates (E): Average temperature of warmest month < 10 °C
        if (annualTemp + 10.0f < 10.0f) { // Simplification: annual temp + 10 as warmest month
            if (annualTemp + 10.0f > 0.0f) {
                climateZones[i] = 47; // ET: Tundra
            } else {
                climateZones[i] = 48; // EF: Ice Cap
            }
        }
        // Continental/Microthermal climates (D): Coldest month < 0 °C, warmest month > 10 °C
        else if (annualTemp - 10.0f < 0.0f && annualTemp + 10.0f > 10.0f) {
            if (annualPrecip > 800.0f) {
                climateZones[i] = 36; // Dfb: Humid continental
            } else {
                climateZones[i] = 40; // Dsb: Dry continental
            }
        }
        // Temperate/Mesothermal climates (C): Coldest month > 0 °C and < 18 °C, warmest month > 10 °C
        else if (annualTemp - 10.0f > 0.0f && annualTemp - 10.0f < 18.0f && annualTemp + 10.0f > 10.0f) {
            if (annualPrecip > 800.0f) {
                climateZones[i] = 26; // Cfb: Marine/Oceanic
            } else {
                climateZones[i] = 29; // Csb: Mediterranean
            }
        }
        // Arid climates (B): Annual precipitation < potential evapotranspiration
        else if (annualPrecip < 500.0f + 20.0f * annualTemp) {
            if (annualPrecip < 250.0f + 10.0f * annualTemp) {
                if (annualTemp > 18.0f) {
                    climateZones[i] = 15; // BWh: Hot desert
                } else {
                    climateZones[i] = 16; // BWk: Cold desert
                }
            } else {
                if (annualTemp > 18.0f) {
                    climateZones[i] = 17; // BSh: Hot steppe
                } else {
                    climateZones[i] = 18; // BSk: Cold steppe
                }
            }
        }
        // Tropical climates (A): Coldest month > 18 °C
        else if (annualTemp - 10.0f > 18.0f) {
            if (annualPrecip > 2000.0f) {
                climateZones[i] = 5; // Af: Tropical rainforest
            } else if (annualPrecip > 1000.0f) {
                climateZones[i] = 6; // Am: Tropical monsoon
            } else {
                climateZones[i] = 7; // Aw: Tropical savanna
            }
        }
        // Default
        else {
            climateZones[i] = 26; // Cfb: Marine/Oceanic (fallback)
        }
    }
}

void ClimateSystem::simulateWeathering(double timeStep) {
    // Weathering simulation based on climate
    // Temperature, precipitation, and biome affect weathering rates
    
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<float>& temps = atmosphere->getTemperatureData();
    const std::vector<float>& precips = hydrology->getPrecipitationData();
    
    for (size_t i = 0; i < grid.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get temperature and precipitation
        float temperature = 15.0f; // Default
        if (vertexIdx < static_cast<int>(temps.size())) {
            temperature = temps[vertexIdx];
        }
        
        float precipitation = 1000.0f; // Default
        if (vertexIdx < static_cast<int>(precips.size())) {
            precipitation = precips[vertexIdx];
        }
        
        // Calculate chemical weathering rate
        // Chemical weathering increases with temperature and precipitation
        float chemicalWeathering = 0.0f;
        if (temperature > 0.0f) {
            // Simplified Arrhenius equation
            chemicalWeathering = 0.1f * std::exp(temperature / 25.0f) * (precipitation / 1000.0f);
        }
        
        // Calculate physical weathering rate
        // Physical weathering is stronger at high elevations and where freeze-thaw occurs
        float physicalWeathering = 0.0f;
        float elevation = planet.getElevation(vertexIdx);
        
        if (elevation > 0.0f) {
            physicalWeathering = 0.05f * elevation;
            
            // Freeze-thaw is most active near 0°C
            if (temperature > -5.0f && temperature < 5.0f) {
                physicalWeathering *= 2.0f;
            }
        }
        
        // Combined weathering rate
        float weatheringRate = (chemicalWeathering + physicalWeathering) * timeStep * 0.01f;
        
        // Apply weathering effects (this would normally modify elevation or surface properties)
        // Since we can't modify the const Planet reference directly, this is a placeholder
        // In a real implementation, we'd track these changes and apply them later
        
        // Weathering increases sediment production, which affects river loads, etc.
        // This would be connected to the erosion and sedimentation models
    }
}

bool ClimateSystem::saveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write climate parameters
    file.write(reinterpret_cast<const char*>(&params), sizeof(params));
    
    // Write climate zones
    size_t zonesSize = climateZones.size();
    file.write(reinterpret_cast<const char*>(&zonesSize), sizeof(zonesSize));
    file.write(reinterpret_cast<const char*>(climateZones.data()), zonesSize * sizeof(int));
    
    return file.good();
}

bool ClimateSystem::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }
    
    // Read climate parameters
    file.read(reinterpret_cast<char*>(&params), sizeof(params));
    
    // Read climate zones
    size_t zonesSize;
    file.read(reinterpret_cast<char*>(&zonesSize), sizeof(zonesSize));
    
    climateZones.resize(zonesSize);
    file.read(reinterpret_cast<char*>(climateZones.data()), zonesSize * sizeof(int));
    
    // Reinitialize component systems with loaded parameters
    atmosphere = std::make_unique<AtmosphericCirculation>(planet, params);
    atmosphere->initialize();
    
    ocean = std::make_unique<OceanicCirculation>(planet, *atmosphere, params);
    ocean->initialize();
    
    hydrology = std::make_unique<HydrologicalSystem>(planet, *atmosphere, *ocean, params);
    hydrology->initialize();
    
    return file.good();
}

} // namespace aeonterracpp