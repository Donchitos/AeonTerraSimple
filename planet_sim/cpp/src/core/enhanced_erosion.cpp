#include "aeonterracpp/core/enhanced_erosion.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace aeonterracpp {

EnhancedErosion::EnhancedErosion(Planet& planet, 
                               const ClimateSystem& climate,
                               const BiomeSystem& biomes)
    : planet(planet), climate(climate), biomes(biomes), 
      grid(planet.getGrid()), rng(std::random_device()()) {
    
    // Initialize with default parameters
    initialize();
}

void EnhancedErosion::initialize(const ErosionParams& params) {
    this->params = params;
    
    // Initialize data buffers
    initializeBuffers();
    
    // Classify vertices for different erosion processes
    classifyVertices();
    
    // Initialize flow information from hydrology
    updateFlowInfo();
    
    std::cout << "Enhanced erosion system initialized with " 
              << planet.getVertexCount() << " vertices." << std::endl;
}

void EnhancedErosion::setParameters(const ErosionParams& params) {
    this->params = params;
}

const ErosionParams& EnhancedErosion::getParameters() const {
    return params;
}

void EnhancedErosion::simulateStep(double timeStep) {
    // Scale time step by the time scale factor
    double scaledTimeStep = timeStep * params.timeScaleFactor;
    
    // Update flow information from hydrology
    updateFlowInfo();
    
    // Recalculate vertex classifications
    classifyVertices();
    
    // Reset erosion rates
    for (auto& rates : processErosionRates) {
        std::fill(rates.begin(), rates.end(), 0.0f);
    }
    std::fill(totalErosionRate.begin(), totalErosionRate.end(), 0.0f);
    
    // Simulate each erosion process
    simulateFluvialErosion(scaledTimeStep);
    simulateCoastalErosion(scaledTimeStep);
    simulateGlacialErosion(scaledTimeStep);
    simulateAeolianErosion(scaledTimeStep);
    simulateChemicalWeathering(scaledTimeStep);
    simulateThermalWeathering(scaledTimeStep);
    simulateMassMovements(scaledTimeStep);
    simulateBioticErosion(scaledTimeStep);
    calculateTectonicErosion(scaledTimeStep);
    
    // Apply a minimum background erosion rate
    applyBackgroundErosion(scaledTimeStep);
    
    // Calculate total erosion rate
    for (size_t i = 0; i < totalErosionRate.size(); ++i) {
        totalErosionRate[i] = 0.0f;
        
        // Sum up all process-specific rates
        for (size_t process = 0; process < static_cast<size_t>(ErosionProcessType::Count); ++process) {
            totalErosionRate[i] += processErosionRates[process][i];
        }
        
        // Apply global erosion rate multiplier
        totalErosionRate[i] *= params.globalErosionRate;
        
        // Convert from meters/Myr to current time step
        float erosionThisStep = totalErosionRate[i] * static_cast<float>(scaledTimeStep);
        
        // Add to accumulated erosion
        accumulatedErosion[i] += erosionThisStep;
        
        // Add eroded material to the sediment buffer
        sedimentBuffer[i] += erosionThisStep;
    }
    
    // Transport sediment downstream
    transportSediment();
    
    // Apply the erosion and deposition to the planet's elevation
    applyErosionToSurface();
    
    std::cout << "Simulated erosion for " << scaledTimeStep << " million years." << std::endl;
}

float EnhancedErosion::getErosionRate(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = grid.findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(totalErosionRate.size())) {
        return totalErosionRate[vertexIdx];
    }
    
    return 0.0f;
}

float EnhancedErosion::getProcessErosionRate(const Vec3f& position, ErosionProcessType processType) const {
    // Find closest vertex to the given position
    int vertexIdx = grid.findClosestVertex(position);
    
    if (vertexIdx >= 0 && 
        vertexIdx < static_cast<int>(processErosionRates[static_cast<size_t>(processType)].size())) {
        return processErosionRates[static_cast<size_t>(processType)][vertexIdx];
    }
    
    return 0.0f;
}

float EnhancedErosion::getAccumulatedErosion(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = grid.findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(accumulatedErosion.size())) {
        return accumulatedErosion[vertexIdx];
    }
    
    return 0.0f;
}

float EnhancedErosion::getAccumulatedDeposition(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = grid.findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(accumulatedDeposition.size())) {
        return accumulatedDeposition[vertexIdx];
    }
    
    return 0.0f;
}

void EnhancedErosion::simulateFluvialErosion(double timeStep) {
    // Fluvial erosion is based on:
    // 1. Water flow (from hydrology)
    // 2. Slope
    // 3. Surface erodibility (from biomes)
    
    std::vector<float> slopes(planet.getVertexCount(), 0.0f);
    calculateSlopes(slopes);
    
    const HydrologicalSystem& hydrology = climate.getHydrology();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get river flow at this vertex
        float flow = flowBuffer[i];
        
        // Only erode if flow exceeds threshold
        if (flow < params.riverFlowThreshold) {
            continue;
        }
        
        // Get slope
        float slope = slopes[i];
        
        // Get biome properties at this vertex
        Vec3f pos = grid.getVertex(vertexIdx);
        BiomeType biomeType = biomes.getBiomeType(pos);
        const BiomeProperties& biomeProps = biomes.getBiomeProperties(pos);
        
        // Calculate erodibility based on biome
        float erodibility = biomeProps.baseErosionRate;
        
        // Apply vegetation protection factor (more vegetation = less erosion)
        float vegetationDensity = biomes.getVegetationDensity(pos);
        float vegetationProtection = 1.0f - vegetationDensity * 0.8f;  // 0.2-1.0 range
        
        // Calculate fluvial erosion rate using the classic stream power law:
        // E = K * A^m * S^n
        // where:
        // E = erosion rate
        // K = erodibility coefficient
        // A = drainage area (proportional to flow)
        // S = slope
        // m, n = empirical exponents
        
        const float m = 0.5f;  // Typical values for m range from 0.3 to 0.7
        const float n = 1.0f;  // Typical values for n range from 0.7 to 1.7
        
        float streamPower = std::pow(flow * 0.01f, m) * std::pow(slope, n);
        float erosionRate = erodibility * vegetationProtection * streamPower;
        
        // Apply fluvial erosion strength parameter
        erosionRate *= params.fluvialErosionStrength;
        
        // Store the calculated erosion rate (in m/Myr)
        processErosionRates[static_cast<size_t>(ErosionProcessType::Fluvial)][i] = erosionRate;
    }
}

void EnhancedErosion::simulateCoastalErosion(double timeStep) {
    // Coastal erosion is based on:
    // 1. Wave energy (simplified model)
    // 2. Coast exposure
    // 3. Shore material erodibility
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process coastal cells
        if (!isCoastal[i]) {
            continue;
        }
        
        // Get biome properties
        Vec3f pos = grid.getVertex(vertexIdx);
        const BiomeProperties& biomeProps = biomes.getBiomeProperties(pos);
        
        // Calculate erodibility based on biome
        float erodibility = biomeProps.baseErosionRate;
        
        // Calculate exposure to waves
        // For a full implementation, this would consider fetch distance and prevailing winds
        // Here we use a simplified approach based on neighbors
        float exposure = 0.0f;
        int oceanNeighbors = 0;
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            if (planet.getElevation(neighborIdx) < 0.0f) {
                oceanNeighbors++;
                
                // Calculate depth gradient (steeper = more wave energy)
                float depthDiff = std::abs(planet.getElevation(neighborIdx) - planet.getElevation(vertexIdx));
                exposure += depthDiff;
            }
        }
        
        if (oceanNeighbors > 0) {
            exposure /= oceanNeighbors;
        }
        
        // Calculate coastal erosion rate
        float erosionRate = erodibility * exposure * params.coastalWaveEnergy;
        
        // Apply coastal erosion strength parameter
        erosionRate *= params.coastalErosionStrength;
        
        // Store the calculated erosion rate (in m/Myr)
        processErosionRates[static_cast<size_t>(ErosionProcessType::Coastal)][i] = erosionRate;
    }
}

void EnhancedErosion::simulateGlacialErosion(double timeStep) {
    // Glacial erosion is based on:
    // 1. Temperature (from climate)
    // 2. Slope and elevation
    // 3. Substrate erodibility
    
    std::vector<float> slopes(planet.getVertexCount(), 0.0f);
    calculateSlopes(slopes);
    
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get position and biome properties
        Vec3f pos = grid.getVertex(vertexIdx);
        float temperature = atmosphere.getTemperature(pos);
        
        // Check if this is a glaciated area
        if (temperature > params.glacialTemperatureThreshold) {
            continue;
        }
        
        // Get biome properties
        const BiomeProperties& biomeProps = biomes.getBiomeProperties(pos);
        float erodibility = biomeProps.baseErosionRate;
        
        // Calculate ice thickness (simplified)
        // In a full implementation, this would come from a glacier simulation
        float elevation = planet.getElevation(vertexIdx);
        float iceThickness = std::max(0.0f, (params.glacialTemperatureThreshold - temperature) * 0.1f);
        
        // Get slope
        float slope = slopes[i];
        
        // Calculate glacial erosion rate
        // Simplified form of: E = Kg * Vi * Pi
        // where Kg = erodibility, Vi = ice velocity, Pi = effective ice pressure
        
        float effectiveIcePressure = iceThickness * 9.81f * 917.0f; // Ice density = 917 kg/m³
        float iceVelocity = slope * iceThickness * 0.1f; // Simplified velocity calculation
        
        float erosionRate = erodibility * iceVelocity * (effectiveIcePressure / 1e6f);
        
        // Apply glacial erosion strength parameter
        erosionRate *= params.glacialErosionStrength;
        
        // Store the calculated erosion rate (in m/Myr)
        processErosionRates[static_cast<size_t>(ErosionProcessType::Glacial)][i] = erosionRate;
        
        // Mark as glaciated for visualization
        isGlaciated[i] = true;
    }
}

void EnhancedErosion::simulateAeolianErosion(double timeStep) {
    // Aeolian (wind) erosion is based on:
    // 1. Wind speed and direction (from climate)
    // 2. Surface erodibility
    // 3. Vegetation cover
    
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean and glaciated cells
        if (planet.getElevation(vertexIdx) < 0.0f || isGlaciated[i]) {
            continue;
        }
        
        // Get position and wind data
        Vec3f pos = grid.getVertex(vertexIdx);
        Vec3f windVector = atmosphere.getWindVector(pos);
        float windSpeed = windVector.norm();
        
        // Only erode if wind exceeds threshold
        if (windSpeed < params.windErosionThreshold) {
            continue;
        }
        
        // Get biome properties
        const BiomeProperties& biomeProps = biomes.getBiomeProperties(pos);
        float erodibility = biomeProps.baseErosionRate;
        
        // Apply vegetation protection factor
        float vegetationDensity = biomes.getVegetationDensity(pos);
        float vegetationProtection = 1.0f - vegetationDensity * 0.9f;  // 0.1-1.0 range
        
        // Calculate aeolian erosion rate
        // This is a simplified model: E = Kw * W^2 * (1-V)
        // where Kw = wind erodibility, W = wind speed, V = vegetation cover
        
        float erosionRate = erodibility * std::pow(windSpeed / 10.0f, 2.0f) * vegetationProtection;
        
        // Apply aeolian erosion strength parameter
        erosionRate *= params.aeolianErosionStrength;
        
        // Store the calculated erosion rate (in m/Myr)
        processErosionRates[static_cast<size_t>(ErosionProcessType::Aeolian)][i] = erosionRate;
    }
}

void EnhancedErosion::simulateChemicalWeathering(double timeStep) {
    // Chemical weathering is based on:
    // 1. Temperature (from climate)
    // 2. Precipitation (from hydrology)
    // 3. Rock type and erodibility
    
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    const HydrologicalSystem& hydrology = climate.getHydrology();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get position and climate data
        Vec3f pos = grid.getVertex(vertexIdx);
        float temperature = atmosphere.getTemperature(pos);
        float precipitation = hydrology.getPrecipitation(pos);
        
        // Get biome properties
        const BiomeProperties& biomeProps = biomes.getBiomeProperties(pos);
        float erodibility = biomeProps.baseErosionRate;
        
        // Chemical weathering increases with temperature (Arrhenius relationship)
        // and precipitation
        
        float temperatureFactor = 0.0f;
        if (temperature > 0.0f) {
            // Simplified Arrhenius equation for temperature dependence
            temperatureFactor = std::exp((temperature - 10.0f) / 10.0f);
        }
        
        float precipitationFactor = precipitation / 1000.0f; // Normalize to 1.0 at 1000mm/year
        
        float erosionRate = erodibility * temperatureFactor * precipitationFactor * 0.5f;
        
        // Apply chemical weathering strength parameter
        erosionRate *= params.chemicalWeatheringStrength;
        
        // Store the calculated erosion rate (in m/Myr)
        processErosionRates[static_cast<size_t>(ErosionProcessType::Chemical)][i] = erosionRate;
    }
}

void EnhancedErosion::simulateThermalWeathering(double timeStep) {
    // Thermal weathering is based on:
    // 1. Temperature (from climate)
    // 2. Temperature variation
    // 3. Rock erodibility
    
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get position and temperature
        Vec3f pos = grid.getVertex(vertexIdx);
        float temperature = atmosphere.getTemperature(pos);
        
        // Get biome properties
        const BiomeProperties& biomeProps = biomes.getBiomeProperties(pos);
        float erodibility = biomeProps.baseErosionRate;
        
        // Thermal weathering is most effective near freezing point
        // where freeze-thaw cycles occur
        float freezeThawFactor = std::exp(-std::pow((temperature - 0.0f) / 5.0f, 2.0f));
        
        float erosionRate = erodibility * freezeThawFactor * 0.7f;
        
        // Apply thermal weathering strength parameter
        erosionRate *= params.thermalWeatheringStrength;
        
        // Store the calculated erosion rate (in m/Myr)
        processErosionRates[static_cast<size_t>(ErosionProcessType::Thermal)][i] = erosionRate;
    }
}

void EnhancedErosion::simulateMassMovements(double timeStep) {
    // Mass movements (landslides, etc.) are based on:
    // 1. Slope
    // 2. Soil saturation (from hydrology)
    // 3. Vegetation cover (from biomes)
    
    std::vector<float> slopes(planet.getVertexCount(), 0.0f);
    calculateSlopes(slopes);
    
    const HydrologicalSystem& hydrology = climate.getHydrology();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get slope
        float slope = slopes[i];
        
        // Only process cells with significant slope
        if (slope < params.criticalSlope * 0.5f) {
            continue;
        }
        
        // Get position and biome data
        Vec3f pos = grid.getVertex(vertexIdx);
        float vegetationDensity = biomes.getVegetationDensity(pos);
        
        // Get precipitation as proxy for soil saturation
        float precipitation = hydrology.getPrecipitation(pos);
        float soilSaturation = std::min(1.0f, precipitation / 2000.0f);
        
        // Calculate mass movement probability
        // Higher when:
        // - Slope is steeper
        // - Soil is more saturated
        // - Less vegetation to hold soil
        
        float vegetationProtection = 1.0f - vegetationDensity * 0.8f;
        float stabilityFactor = slope / params.criticalSlope;
        
        // Calculate probability of mass movement
        float movementProbability = stabilityFactor * soilSaturation * vegetationProtection;
        
        // Only erode if probability exceeds threshold or slope exceeds critical value
        if (movementProbability > 0.7f || slope > params.criticalSlope) {
            // Calculate average height difference to downslope neighbors
            float avgHeightDiff = 0.0f;
            int downNeighbors = 0;
            
            const auto& neighbors = grid.getNeighbors(vertexIdx);
            for (int neighborIdx : neighbors) {
                float heightDiff = planet.getElevation(vertexIdx) - planet.getElevation(neighborIdx);
                if (heightDiff > 0.0f) {
                    avgHeightDiff += heightDiff;
                    downNeighbors++;
                }
            }
            
            if (downNeighbors > 0) {
                avgHeightDiff /= downNeighbors;
            }
            
            // Calculate erosion rate based on slope stability
            float erosionRate = avgHeightDiff * stabilityFactor * 2.0f;
            
            // Apply mass movement strength parameter
            erosionRate *= params.massMovementStrength;
            
            // Store the calculated erosion rate (in m/Myr)
            processErosionRates[static_cast<size_t>(ErosionProcessType::Gravity)][i] = erosionRate;
        }
    }
}

void EnhancedErosion::simulateBioticErosion(double timeStep) {
    // Biotic erosion is based on:
    // 1. Vegetation density (from biomes)
    // 2. Biome type
    // 3. Soil conditions
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get position and biome data
        Vec3f pos = grid.getVertex(vertexIdx);
        BiomeType biomeType = biomes.getBiomeType(pos);
        float vegetationDensity = biomes.getVegetationDensity(pos);
        
        // Biotic erosion includes both protective and erosive effects
        
        // In some biomes, vegetation can increase erosion through root action
        // In others, it mostly protects against erosion
        
        float bioticFactor = 0.0f;
        
        // Determine biotic erosion factor based on biome type
        switch (biomeType) {
            case BiomeType::TropicalRainforest:
            case BiomeType::TemperateRainforest:
                // High rainfall, high vegetation = moderate root action
                bioticFactor = vegetationDensity * 0.4f;
                break;
            
            case BiomeType::TropicalSeasonalForest:
            case BiomeType::TemperateDeciduousForest:
                // Seasonal forests = strong root action
                bioticFactor = vegetationDensity * 0.6f;
                break;
            
            case BiomeType::Grassland:
            case BiomeType::Savanna:
                // Grasses have dense root networks = high root action
                bioticFactor = vegetationDensity * 0.7f;
                break;
            
            case BiomeType::HotDesert:
            case BiomeType::ColdDesert:
                // Low vegetation = low biological activity
                bioticFactor = vegetationDensity * 0.2f;
                break;
            
            default:
                // Default moderate action
                bioticFactor = vegetationDensity * 0.5f;
        }
        
        // Calculate biotic erosion rate
        float erosionRate = bioticFactor * 0.3f;
        
        // Apply biotic erosion strength parameter
        erosionRate *= params.bioticErosionStrength;
        
        // Store the calculated erosion rate (in m/Myr)
        processErosionRates[static_cast<size_t>(ErosionProcessType::Biotic)][i] = erosionRate;
    }
}

void EnhancedErosion::calculateTectonicErosion(double timeStep) {
    // Tectonic erosion is based on:
    // 1. Proximity to plate boundaries
    // 2. Tectonic activity (uplift/subsidence rates)
    // 3. Terrain characteristics
    
    // Ideally this would come from a tectonic simulator
    // Here we use a simplified approach based on plate ID changes
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Check if this is near a plate boundary
        bool nearBoundary = false;
        int plateId = planet.getPlateId(vertexIdx);
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            if (planet.getPlateId(neighborIdx) != plateId) {
                nearBoundary = true;
                break;
            }
        }
        
        if (nearBoundary) {
            // Higher erosion at plate boundaries
            float erosionRate = 1.0f;
            
            // Store the calculated erosion rate (in m/Myr)
            processErosionRates[static_cast<size_t>(ErosionProcessType::Tectonic)][i] = erosionRate;
        }
    }
}

void EnhancedErosion::applyErosionToSurface() {
    // Apply the accumulated erosion and deposition to the surface
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Get current elevation
        float elevation = planet.getElevation(vertexIdx);
        
        // Get net erosion/deposition for this cell
        // We use the buffer since we just moved sediment
        float netErosion = erosionBuffer[i];
        
        // Apply to elevation, but avoid creating huge holes
        float newElevation = elevation - netErosion;
        
        // Set the new elevation
        planet.setElevation(vertexIdx, newElevation);
    }
    
    // Reset the erosion buffer
    std::fill(erosionBuffer.begin(), erosionBuffer.end(), 0.0f);
}

void EnhancedErosion::applyBackgroundErosion(double timeStep) {
    // Apply a minimum background erosion to all land cells
    // This represents processes not explicitly modeled
    
    const float backgroundRate = 0.01f; // m/Myr
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Add background erosion to all processes
        for (size_t process = 0; process < static_cast<size_t>(ErosionProcessType::Count); ++process) {
            if (processErosionRates[process][i] < backgroundRate) {
                processErosionRates[process][i] = backgroundRate;
            }
        }
    }
}

void EnhancedErosion::initializeBuffers() {
    size_t vertexCount = planet.getVertexCount();
    
    // Initialize erosion rate vectors for each process
    processErosionRates.resize(static_cast<size_t>(ErosionProcessType::Count));
    for (auto& rates : processErosionRates) {
        rates.resize(vertexCount, 0.0f);
    }
    
    // Initialize total erosion rate
    totalErosionRate.resize(vertexCount, 0.0f);
    
    // Initialize accumulated erosion and deposition
    accumulatedErosion.resize(vertexCount, 0.0f);
    accumulatedDeposition.resize(vertexCount, 0.0f);
    
    // Initialize temporary buffers
    erosionBuffer.resize(vertexCount, 0.0f);
    sedimentBuffer.resize(vertexCount, 0.0f);
    sedimentCapacity.resize(vertexCount, 0.0f);
    
    // Initialize vertex classification flags
    isGlaciated.resize(vertexCount, false);
    isCoastal.resize(vertexCount, false);
    
    // Initialize flow information
    flowBuffer.resize(vertexCount, 0.0f);
    flowDirectionBuffer.resize(vertexCount, -1);
    upstreamAreaBuffer.resize(vertexCount, 0.0f);
}

void EnhancedErosion::updateFlowInfo() {
    // Get flow information from hydrological system
    const HydrologicalSystem& hydrology = climate.getHydrology();
    const std::vector<float>& riverFlow = hydrology.getRiverFlowData();
    
    // Copy flow data to our buffer
    if (riverFlow.size() == flowBuffer.size()) {
        flowBuffer = riverFlow;
    }
    
    // Ideally, we would also get flow directions and upstream areas
    // For now, we'll use a simple approximation
    
    // Calculate upstream catchment areas
    std::fill(upstreamAreaBuffer.begin(), upstreamAreaBuffer.end(), 0.0f);
    
    // Calculate simple flow directions based on steepest descent
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        
        // Find steepest downward neighbor
        int steepestIdx = -1;
        float maxDescent = 0.0f;
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            float neighborElevation = planet.getElevation(neighborIdx);
            float descent = elevation - neighborElevation;
            
            if (descent > maxDescent) {
                maxDescent = descent;
                steepestIdx = neighborIdx;
            }
        }
        
        // Set flow direction to steepest neighbor, or -1 if no downward path
        flowDirectionBuffer[i] = steepestIdx;
        
        // Each cell has at least its own area
        upstreamAreaBuffer[i] = 1.0f;
    }
    
    // Accumulate areas downstream
    // We need to start from highest points and move downstream
    // So we sort vertices by elevation
    
    struct HeightVertex {
        int index;
        float elevation;
        
        bool operator<(const HeightVertex& other) const {
            return elevation > other.elevation; // Higher elevations first
        }
    };
    
    std::vector<HeightVertex> sortedVertices;
    sortedVertices.reserve(planet.getVertexCount());
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        sortedVertices.push_back({vertexIdx, planet.getElevation(vertexIdx)});
    }
    
    std::sort(sortedVertices.begin(), sortedVertices.end());
    
    // Propagate areas downstream
    for (const auto& vertex : sortedVertices) {
        int vertexIdx = vertex.index;
        int downstreamIdx = flowDirectionBuffer[vertexIdx];
        
        if (downstreamIdx >= 0 && downstreamIdx < static_cast<int>(upstreamAreaBuffer.size())) {
            upstreamAreaBuffer[downstreamIdx] += upstreamAreaBuffer[vertexIdx];
        }
    }
}

void EnhancedErosion::classifyVertices() {
    // Reset classification flags
    std::fill(isGlaciated.begin(), isGlaciated.end(), false);
    std::fill(isCoastal.begin(), isCoastal.end(), false);
    
    // Classify coastal cells (land adjacent to water)
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Check if any neighbor is ocean
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            if (planet.getElevation(neighborIdx) < 0.0f) {
                isCoastal[i] = true;
                break;
            }
        }
    }
    
    // Glaciated cells are classified during glacial erosion
}

void EnhancedErosion::calculateSlopes(std::vector<float>& slopes) {
    // Calculate the maximum slope for each vertex
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        float maxSlope = 0.0f;
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            float neighborElevation = planet.getElevation(neighborIdx);
            Vec3f neighborPos = grid.getVertex(neighborIdx);
            
            // Calculate the slope between the two points
            float elevationDiff = std::abs(elevation - neighborElevation);
            float distance = (neighborPos - pos).norm();
            
            float slope = elevationDiff / distance;
            maxSlope = std::max(maxSlope, slope);
        }
        
        slopes[i] = maxSlope;
    }
}

void EnhancedErosion::calculateSedimentCapacity() {
    // Calculate sediment carrying capacity for each cell
    // This is used for sediment transport
    
    std::vector<float> slopes(planet.getVertexCount(), 0.0f);
    calculateSlopes(slopes);
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            sedimentCapacity[i] = 0.0f;
            continue;
        }
        
        // Get slope and flow
        float slope = slopes[i];
        float flow = flowBuffer[i];
        
        // Calculate sediment capacity based on slope and flow
        // C = Ks * Q * S
        // where Ks = sediment capacity factor, Q = flow, S = slope
        
        sedimentCapacity[i] = params.sedimentCapacityFactor * flow * slope * 0.01f;
    }
}

void EnhancedErosion::transportSediment() {
    // Calculate sediment capacity for each cell
    calculateSedimentCapacity();
    
    // Create a buffer for updated sediment
    std::vector<float> newSediment(sedimentBuffer.size(), 0.0f);
    
    // Clear the erosion buffer (this will track the net erosion/deposition)
    std::fill(erosionBuffer.begin(), erosionBuffer.end(), 0.0f);
    
    // Process vertices from highest to lowest elevation
    struct HeightVertex {
        int index;
        float elevation;
        
        bool operator<(const HeightVertex& other) const {
            return elevation > other.elevation; // Higher elevations first
        }
    };
    
    std::vector<HeightVertex> sortedVertices;
    sortedVertices.reserve(planet.getVertexCount());
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        sortedVertices.push_back({vertexIdx, planet.getElevation(vertexIdx)});
    }
    
    std::sort(sortedVertices.begin(), sortedVertices.end());
    
    // Transport sediment downstream
    for (const auto& vertex : sortedVertices) {
        int vertexIdx = vertex.index;
        float currentSediment = sedimentBuffer[vertexIdx];
        
        // Get downstream vertex
        int downstreamIdx = flowDirectionBuffer[vertexIdx];
        
        // Skip if no downstream path
        if (downstreamIdx < 0 || downstreamIdx >= static_cast<int>(sedimentBuffer.size())) {
            // Sediment remains here (sink point)
            newSediment[vertexIdx] += currentSediment;
            
            // Record as deposition
            erosionBuffer[vertexIdx] -= currentSediment;
            accumulatedDeposition[vertexIdx] += currentSediment;
            
            continue;
        }
        
        // Check if we're draining to ocean
        if (planet.getElevation(downstreamIdx) < 0.0f) {
            // Sediment goes to ocean floor
            newSediment[downstreamIdx] += currentSediment;
            
            // Record as deposition in ocean
            erosionBuffer[downstreamIdx] -= currentSediment;
            accumulatedDeposition[downstreamIdx] += currentSediment;
            
            continue;
        }
        
        // Get sediment capacity
        float capacity = sedimentCapacity[vertexIdx];
        
        if (currentSediment > capacity) {
            // Carrying too much sediment, deposit the excess
            float excessSediment = currentSediment - capacity;
            
            // Deposit excess here
            newSediment[vertexIdx] += excessSediment;
            
            // Record as deposition
            erosionBuffer[vertexIdx] -= excessSediment;
            accumulatedDeposition[vertexIdx] += excessSediment;
            
            // Pass the rest downstream
            newSediment[downstreamIdx] += capacity;
        } else {
            // Capacity to carry more, try to erode more
            float additionalErosion = std::min(capacity - currentSediment, 0.01f);
            
            // Record as erosion
            erosionBuffer[vertexIdx] += additionalErosion;
            
            // Pass all sediment downstream
            newSediment[downstreamIdx] += currentSediment + additionalErosion;
        }
    }
    
    // Update sediment buffer
    sedimentBuffer = newSediment;
}

void EnhancedErosion::updateBiomeEffects(BiomeSystem& biomes) {
    // This would provide feedback to update biome properties based on erosion
    // For example:
    // - Heavy erosion could reduce soil depth
    // - Deposition could create fertile floodplains
    // - Erosion patterns could influence vegetation distribution
    
    // This is a placeholder for the actual implementation
    std::cout << "Updating biome effects from erosion..." << std::endl;
}

void EnhancedErosion::updateHydrologyEffects(ClimateSystem& climate) {
    // This would provide feedback to update hydrology based on erosion
    // For example:
    // - Erosion could create new river channels
    // - Deposition could dam valleys to form lakes
    // - Changed topography affects water flow patterns
    
    // This is a placeholder for the actual implementation
    std::cout << "Updating hydrology effects from erosion..." << std::endl;
}

const std::vector<float>& EnhancedErosion::getErosionData() const {
    return totalErosionRate;
}

const std::vector<float>& EnhancedErosion::getProcessErosionData(ErosionProcessType processType) const {
    return processErosionRates[static_cast<size_t>(processType)];
}

const std::vector<float>& EnhancedErosion::getSedimentData() const {
    return sedimentBuffer;
}

} // namespace aeonterracpp