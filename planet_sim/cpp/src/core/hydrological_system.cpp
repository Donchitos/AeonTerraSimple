#include "aeonterracpp/core/climate_system.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_set>

namespace aeonterracpp {

HydrologicalSystem::HydrologicalSystem(const Planet& planet,
                                      const AtmosphericCirculation& atmosphere,
                                      const OceanicCirculation& ocean,
                                      const ClimateParams& params)
    : planet(planet), atmosphere(atmosphere), ocean(ocean), params(params) {
    // Initialize data vectors with the size of the planet's vertex count
    size_t vertexCount = planet.getVertexCount();
    precipitation.resize(vertexCount, 1000.0f); // Default precipitation in mm/year
    evaporation.resize(vertexCount, 1000.0f);   // Default evaporation in mm/year
    soilMoisture.resize(vertexCount, 50.0f);    // Default soil moisture in % capacity
    riverFlow.resize(vertexCount, 0.0f);        // Default river flow in m³/s
    flowDirection.resize(vertexCount, -1);      // Default: no outflow (sink)
    drainage.resize(vertexCount, 0.0f);         // Default drainage area in km²
    isRiver.resize(vertexCount, false);         // Default: not a river
    isLake.resize(vertexCount, false);          // Default: not a lake
    atmosphericMoisture.resize(vertexCount, 0.0f); // Default atmospheric moisture content
}

void HydrologicalSystem::initialize() {
    std::cout << "Initializing hydrological system..." << std::endl;
    
    // Initialize moisture transport based on atmosphere
    calculateMoistureTransport();
    
    // Calculate precipitation patterns
    calculateConvectionalPrecipitation();
    calculateOrographicPrecipitation();
    calculateFrontalPrecipitation();
    
    // Calculate evaporation rates
    calculateEvaporation();
    
    // Initialize river networks
    calculateFlowDirections();
    simulateDetailedRiverNetworks();
    
    // Calculate lake formation
    calculateLakeFormation();
    
    std::cout << "Hydrological system initialized successfully." << std::endl;
}

void HydrologicalSystem::simulate(double timeStep, float dayOfYear) {
    // Update moisture transport based on current wind patterns
    calculateMoistureTransport();
    
    // Calculate precipitation patterns
    calculateConvectionalPrecipitation();
    calculateOrographicPrecipitation();
    calculateFrontalPrecipitation();
    
    // Calculate evaporation rates
    calculateEvaporation();
    
    // Update river networks
    calculateFlowDirections();
    simulateDetailedRiverNetworks();
    
    // Update lake formation
    calculateLakeFormation();
    
    // Calculate delta formation at river mouths
    calculateDeltaFormation();
    
    // Apply seasonal variations based on day of year
    applySeasonalVariations(dayOfYear);
}

float HydrologicalSystem::getPrecipitation(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(precipitation.size())) {
        return precipitation[vertexIdx];
    }
    
    return 1000.0f; // Default precipitation
}

float HydrologicalSystem::getEvaporation(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(evaporation.size())) {
        return evaporation[vertexIdx];
    }
    
    return 1000.0f; // Default evaporation
}

float HydrologicalSystem::getRiverFlow(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(riverFlow.size())) {
        return riverFlow[vertexIdx];
    }
    
    return 0.0f; // Default river flow
}

void HydrologicalSystem::calculateEvaporation() {
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<float>& temps = atmosphere.getTemperatureData();
    
    for (size_t i = 0; i < evaporation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Get temperature at this vertex
        float temperature = 15.0f; // Default
        if (vertexIdx < static_cast<int>(temps.size())) {
            temperature = temps[vertexIdx];
        }
        
        // Calculate potential evaporation based on temperature
        // Simple approximation: Thornthwaite method
        float potentialEvaporation = 0.0f;
        
        if (temperature > 0.0f) {
            // Evaporation increases with temperature
            potentialEvaporation = 16.0f * std::pow((10.0f * temperature / 25.0f), 1.5f) * 30.0f;
        }
        
        // Scale by solar insolation (latitude-dependent)
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        float insolationFactor = std::cos(absLat * M_PI / 180.0f);
        potentialEvaporation *= insolationFactor;
        
        // Actual evaporation is limited by water availability
        float elevation = planet.getElevation(vertexIdx);
        
        if (elevation < 0.0f) {
            // Ocean - evaporation is close to potential
            evaporation[i] = potentialEvaporation;
        } else {
            // Land - evaporation limited by soil moisture
            float soilFactor = soilMoisture[i] / 100.0f; // 0-1 based on soil moisture percentage
            evaporation[i] = potentialEvaporation * soilFactor;
        }
        
        // Scale by global parameter
        evaporation[i] *= params.globalPrecipitation / 1000.0f;
    }
}

void HydrologicalSystem::calculateMoistureTransport() {
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<Vec3f>& windVectors = atmosphere.getWindData();
    
    // Reset atmospheric moisture
    std::fill(atmosphericMoisture.begin(), atmosphericMoisture.end(), 0.0f);
    
    // Start with evaporation as moisture source
    for (size_t i = 0; i < atmosphericMoisture.size(); ++i) {
        atmosphericMoisture[i] = evaporation[i] * 0.001f; // Initial moisture from local evaporation
    }
    
    // Transport moisture by wind
    // This is a simplified model - in reality, this would be a fluid dynamics simulation
    
    // First, identify all ocean cells as constant moisture sources
    for (size_t i = 0; i < atmosphericMoisture.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        if (planet.getElevation(vertexIdx) < 0.0f) {
            // Ocean cell - constant moisture source
            atmosphericMoisture[i] = std::max(atmosphericMoisture[i], 0.5f);
        }
    }
    
    // Perform several iterations of moisture transport
    const int transportIterations = 10;
    
    for (int iter = 0; iter < transportIterations; ++iter) {
        std::vector<float> newMoisture = atmosphericMoisture;
        
        for (size_t i = 0; i < atmosphericMoisture.size(); ++i) {
            int vertexIdx = static_cast<int>(i);
            Vec3f pos = grid.getVertex(vertexIdx);
            
            // Skip ocean cells (constant moisture sources)
            if (planet.getElevation(vertexIdx) < 0.0f) {
                continue;
            }
            
            // Get wind vector at this vertex
            Vec3f windVector = Vec3f::Zero();
            if (vertexIdx < static_cast<int>(windVectors.size())) {
                windVector = windVectors[vertexIdx];
            }
            
            if (windVector.norm() < 0.01f) {
                continue; // Negligible wind
            }
            
            // Find upwind position
            Vec3f windDir = windVector.normalized();
            Vec3f upwindPos = pos - windDir * 0.1f;
            
            // Find closest vertex to upwind position
            int upwindIdx = grid.findClosestVertex(upwindPos);
            
            if (upwindIdx >= 0 && upwindIdx < static_cast<int>(atmosphericMoisture.size())) {
                // Transport moisture from upwind
                float upwindMoisture = atmosphericMoisture[upwindIdx];
                
                // Transport factor depends on wind speed
                float transportFactor = std::min(0.5f, windVector.norm() * 0.1f);
                
                // Add transported moisture
                newMoisture[i] += upwindMoisture * transportFactor;
                
                // Reduce moisture upwind (conservation of mass)
                newMoisture[upwindIdx] *= (1.0f - transportFactor);
            }
        }
        
        // Update moisture for next iteration
        atmosphericMoisture = newMoisture;
    }
}

void HydrologicalSystem::calculateConvectionalPrecipitation() {
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<float>& temps = atmosphere.getTemperatureData();
    
    for (size_t i = 0; i < precipitation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Get temperature at this vertex
        float temperature = 15.0f; // Default
        if (vertexIdx < static_cast<int>(temps.size())) {
            temperature = temps[vertexIdx];
        }
        
        // Convectional precipitation occurs mainly in warm areas
        // Particularly strong near the equator (ITCZ)
        float convectionalPrecip = 0.0f;
        
        if (temperature > 18.0f) {
            // Warm air rises, cools, and forms precipitation
            float tempFactor = (temperature - 18.0f) / 12.0f; // 0-1 for temps 18-30°C
            tempFactor = std::min(1.0f, tempFactor);
            
            // Moisture availability affects precipitation
            float moistureFactor = std::min(1.0f, atmosphericMoisture[i] * 2.0f);
            
            convectionalPrecip = 2000.0f * tempFactor * moistureFactor * params.convectionalEffect;
        }
        
        // Add to precipitation total (will be combined with other types)
        precipitation[i] = convectionalPrecip;
    }
}

void HydrologicalSystem::calculateOrographicPrecipitation() {
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<Vec3f>& windVectors = atmosphere.getWindData();
    
    for (size_t i = 0; i < precipitation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Skip ocean cells for orographic calculation
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Get wind vector at this vertex
        Vec3f windVector = Vec3f::Zero();
        if (vertexIdx < static_cast<int>(windVectors.size())) {
            windVector = windVectors[vertexIdx];
        }
        
        if (windVector.norm() < 0.01f) {
            continue; // Negligible wind
        }
        
        // Calculate upwind slope
        Vec3f windDir = windVector.normalized();
        float maxUpwindSlope = 0.0f;
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            Vec3f neighborPos = grid.getVertex(neighborIdx);
            Vec3f toNeighbor = (neighborPos - pos).normalized();
            
            // Dot product determines if this is upwind
            float alignment = windDir.dot(toNeighbor);
            
            if (alignment < -0.5f) {
                // This is upwind
                float myElevation = planet.getElevation(vertexIdx);
                float neighborElevation = planet.getElevation(neighborIdx);
                float elevationDiff = myElevation - neighborElevation;
                float distance = (neighborPos - pos).norm();
                float slope = elevationDiff / distance;
                
                // Track maximum upwind slope
                maxUpwindSlope = std::max(maxUpwindSlope, slope);
            }
        }
        
        // Orographic precipitation increases with upwind slope
        float orographicPrecip = 0.0f;
        
        if (maxUpwindSlope > 0.05f) {
            // Significant upslope - air rises, cools, and precipitation forms
            float slopeFactor = std::min(1.0f, maxUpwindSlope / 0.2f);
            
            // Moisture availability affects precipitation
            float moistureFactor = std::min(1.0f, atmosphericMoisture[i] * 2.0f);
            
            orographicPrecip = 1500.0f * slopeFactor * moistureFactor * params.orographicEffect;
            
            // Reduce atmospheric moisture
            atmosphericMoisture[i] *= std::max(0.0f, 1.0f - slopeFactor * 0.5f);
        }
        
        // Add to precipitation total
        precipitation[i] += orographicPrecip;
        
        // Rain shadow effect on the leeward side
        if (maxUpwindSlope > 0.1f) {
            // Find downwind neighbors (leeward side)
            for (int neighborIdx : neighbors) {
                Vec3f neighborPos = grid.getVertex(neighborIdx);
                Vec3f toNeighbor = (neighborPos - pos).normalized();
                
                // Dot product determines if this is downwind
                float alignment = windDir.dot(toNeighbor);
                
                if (alignment > 0.5f && neighborIdx < static_cast<int>(atmosphericMoisture.size())) {
                    // This is downwind - reduce moisture for rain shadow effect
                    atmosphericMoisture[neighborIdx] *= std::max(0.1f, 1.0f - maxUpwindSlope);
                }
            }
        }
    }
}

void HydrologicalSystem::calculateFrontalPrecipitation() {
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<float>& temps = atmosphere.getTemperatureData();
    const std::vector<float>& pressures = atmosphere.getPressureData();
    
    // Frontal precipitation occurs where different air masses meet
    // Typically along the polar front and other convergence zones
    
    for (size_t i = 0; i < precipitation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Get temperature and pressure at this vertex
        float temperature = 15.0f; // Default
        float pressure = 1013.25f; // Default
        
        if (vertexIdx < static_cast<int>(temps.size())) {
            temperature = temps[vertexIdx];
        }
        
        if (vertexIdx < static_cast<int>(pressures.size())) {
            pressure = pressures[vertexIdx];
        }
        
        // Check for temperature gradient (indicator of fronts)
        float maxTempGradient = 0.0f;
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            if (neighborIdx < static_cast<int>(temps.size())) {
                float neighborTemp = temps[neighborIdx];
                float tempDiff = std::abs(temperature - neighborTemp);
                
                // Track maximum temperature gradient
                maxTempGradient = std::max(maxTempGradient, tempDiff);
            }
        }
        
        // Frontal precipitation increases with temperature gradient
        float frontalPrecip = 0.0f;
        
        if (maxTempGradient > 2.0f) {
            // Significant temperature gradient - potential front
            float frontFactor = std::min(1.0f, (maxTempGradient - 2.0f) / 8.0f);
            
            // Low pressure enhances precipitation at fronts
            float pressureFactor = std::max(0.0f, (1015.0f - pressure) / 15.0f);
            pressureFactor = std::min(1.0f, pressureFactor);
            
            // Moisture availability affects precipitation
            float moistureFactor = std::min(1.0f, atmosphericMoisture[i] * 2.0f);
            
            frontalPrecip = 1000.0f * frontFactor * pressureFactor * moistureFactor * params.frontalEffect;
            
            // Reduce atmospheric moisture
            atmosphericMoisture[i] *= std::max(0.0f, 1.0f - frontFactor * pressureFactor * 0.5f);
        }
        
        // Add to precipitation total
        precipitation[i] += frontalPrecip;
    }
    
    // Scale precipitation to match global average
    float totalPrecip = 0.0f;
    int landCount = 0;
    
    for (size_t i = 0; i < precipitation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only consider land cells for global average calculation
        if (planet.getElevation(vertexIdx) >= 0.0f) {
            totalPrecip += precipitation[i];
            landCount++;
        } else {
            // Ocean precipitation is less important for simulation
            // but should still be reasonable
            precipitation[i] = std::min(2000.0f, precipitation[i]);
        }
    }
    
    if (landCount > 0) {
        float avgPrecip = totalPrecip / landCount;
        float scaleFactor = params.globalPrecipitation / avgPrecip;
        
        // Scale all precipitation values
        for (size_t i = 0; i < precipitation.size(); ++i) {
            precipitation[i] *= scaleFactor;
        }
    }
}

void HydrologicalSystem::calculateFlowDirections() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Calculate flow direction for each vertex
    for (size_t i = 0; i < flowDirection.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        
        // Find lowest neighbor
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        flowDirection[i] = -1; // Default: no outflow (sink)
        float lowestElevation = elevation;
        
        for (int neighborIdx : neighbors) {
            float neighborElevation = planet.getElevation(neighborIdx);
            if (neighborElevation < lowestElevation) {
                lowestElevation = neighborElevation;
                flowDirection[i] = neighborIdx;
            }
        }
        
        // Calculate initial drainage area (base unit is one cell)
        drainage[i] = 1.0f;
    }
}

void HydrologicalSystem::simulateDetailedRiverNetworks() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Reset river flow
    std::fill(riverFlow.begin(), riverFlow.end(), 0.0f);
    std::fill(isRiver.begin(), isRiver.end(), false);
    
    // Calculate drainage area by propagating flow downstream
    // Sort vertices by elevation to process from highest to lowest
    struct DrainageVertex {
        int index;
        float elevation;
        
        bool operator<(const DrainageVertex& other) const {
            return elevation > other.elevation; // Higher elevations first
        }
    };
    
    std::vector<DrainageVertex> sortedVertices;
    sortedVertices.reserve(flowDirection.size());
    
    for (size_t i = 0; i < flowDirection.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        sortedVertices.push_back({vertexIdx, planet.getElevation(vertexIdx)});
    }
    
    std::sort(sortedVertices.begin(), sortedVertices.end());
    
    // Propagate drainage area downstream
    for (const auto& vertex : sortedVertices) {
        int vertexIdx = vertex.index;
        int flowTarget = flowDirection[vertexIdx];
        
        if (flowTarget >= 0 && flowTarget < static_cast<int>(drainage.size())) {
            // Add this cell's drainage to the downstream cell
            drainage[flowTarget] += drainage[vertexIdx];
        }
    }
    
    // Calculate river flows based on drainage area and precipitation/runoff
    for (size_t i = 0; i < riverFlow.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Calculate runoff from precipitation and evaporation
        float runoff = calculateRunoff(precipitation[i], evaporation[i]);
        
        // Average cell area (rough approximation)
        float cellArea = 4.0f * M_PI * planet.getRadius() * planet.getRadius() / grid.getVertexCount();
        cellArea *= 1.0e6f; // Convert to m²
        
        // Drainage area in m²
        float drainageAreaM2 = drainage[i] * cellArea;
        
        // Runoff in m/year
        float runoffM = runoff / 1000.0f;
        
        // River flow in m³/s = area * runoff / seconds_per_year
        float secondsPerYear = 365.24f * 24.0f * 3600.0f;
        riverFlow[i] = drainageAreaM2 * runoffM / secondsPerYear;
        
        // Mark as river if flow exceeds threshold
        if (riverFlow[i] > 10.0f) {
            isRiver[i] = true;
        }
    }
}

void HydrologicalSystem::calculateLakeFormation() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Reset lakes
    std::fill(isLake.begin(), isLake.end(), false);
    
    // Find depressions (sinks in flow directions)
    for (size_t i = 0; i < flowDirection.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Check if this is a sink (no outflow)
        if (flowDirection[i] == -1) {
            isLake[i] = true;
            
            // Flood adjacent cells if they flow into this sink
            std::queue<int> floodQueue;
            floodQueue.push(vertexIdx);
            
            while (!floodQueue.empty()) {
                int currentIdx = floodQueue.front();
                floodQueue.pop();
                
                // Find neighbors that flow into this cell
                const auto& neighbors = grid.getNeighbors(currentIdx);
                
                for (int neighborIdx : neighbors) {
                    if (neighborIdx < static_cast<int>(flowDirection.size()) && 
                        flowDirection[neighborIdx] == currentIdx && 
                        !isLake[neighborIdx] && 
                        planet.getElevation(neighborIdx) >= 0.0f) {
                        
                        // This neighbor flows into the sink/lake
                        isLake[neighborIdx] = true;
                        floodQueue.push(neighborIdx);
                    }
                }
            }
        }
    }
}

void HydrologicalSystem::calculateDeltaFormation() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Find river mouths (where rivers meet the ocean)
    for (size_t i = 0; i < isRiver.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip non-river cells
        if (!isRiver[i]) {
            continue;
        }
        
        // Check if this river flows into the ocean
        int flowTarget = flowDirection[i];
        
        if (flowTarget >= 0 && flowTarget < static_cast<int>(flowDirection.size())) {
            if (planet.getElevation(flowTarget) < 0.0f) {
                // This is a river mouth
                
                // Delta formation depends on river flow and coastal slope
                float deltaSize = std::log10(std::max(10.0f, riverFlow[i])) * 0.5f;
                
                // Distribute sediment around river mouth
                const auto& neighbors = grid.getNeighbors(flowTarget);
                
                for (int neighborIdx : neighbors) {
                    // Only deposit sediment in shallow ocean
                    if (planet.getElevation(neighborIdx) < 0.0f && 
                        planet.getElevation(neighborIdx) > -0.5f) {
                        
                        // Deltas tend to be near or at sea level
                        float targetElevation = -0.1f;
                        float currentElevation = planet.getElevation(neighborIdx);
                        
                        // Move toward target elevation based on river flow
                        float adjustment = (targetElevation - currentElevation) * 0.01f * deltaSize;
                        
                        // This would modify the planet's elevation, but we can't do that directly
                        // from here since the Planet reference is const. In a real implementation,
                        // this would be tracked and applied later or we'd have a non-const reference.
                        
                        // planet.setElevation(neighborIdx, currentElevation + adjustment);
                    }
                }
            }
        }
    }
}

float HydrologicalSystem::calculateRunoff(float precipitation, float evaporation) const {
    // Simple runoff model: precipitation - evaporation
    float runoff = precipitation - evaporation;
    
    // Runoff can't be negative
    return std::max(0.0f, runoff);
}

void HydrologicalSystem::applySeasonalVariations(float dayOfYear) {
    const SphericalGrid& grid = planet.getGrid();
    
    // Calculate seasonal factor (0-1 through the year)
    float seasonalFactor = sin(2.0f * M_PI * (dayOfYear - 80.0f) / params.yearLength);
    
    for (size_t i = 0; i < precipitation.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Seasonal variations are stronger at higher latitudes
        float absLat = std::abs(lat);
        float latitudeFactor = absLat / 90.0f;
        
        // Seasonal effect depends on hemisphere
        float hemisphereEffect = (lat >= 0.0f) ? seasonalFactor : -seasonalFactor;
        
        // Monsoon systems - strong seasonal variation in precipitation
        // Typically affects areas between 10° and 30° latitude
        if (absLat > 10.0f && absLat < 30.0f) {
            // Check for land near ocean (monsoon conditions)
            bool isLandNearOcean = false;
            
            const auto& neighbors = grid.getNeighbors(vertexIdx);
            for (int neighborIdx : neighbors) {
                if (planet.getElevation(neighborIdx) * planet.getElevation(vertexIdx) < 0.0f) {
                    // One is land, one is ocean
                    isLandNearOcean = true;
                    break;
                }
            }
            
            if (isLandNearOcean) {
                // Summer monsoon brings increased precipitation
                float monsoonEffect = hemisphereEffect * params.monsoonStrength;
                
                // Scale precipitation by monsoon effect
                precipitation[i] *= (1.0f + std::max(0.0f, monsoonEffect) * 1.5f);
            }
        }
        
        // General seasonal precipitation variation
        float seasonalPrecipEffect = hemisphereEffect * latitudeFactor * 0.3f;
        precipitation[i] *= (1.0f + seasonalPrecipEffect);
        
        // Seasonal variation in evaporation
        float seasonalEvapEffect = hemisphereEffect * latitudeFactor * 0.5f;
        evaporation[i] *= (1.0f + seasonalEvapEffect);
        
        // Update soil moisture based on precipitation and evaporation balance
        float waterBalance = precipitation[i] - evaporation[i];
        
        if (waterBalance > 0.0f) {
            // Increase soil moisture (with cap at 100%)
            soilMoisture[i] = std::min(100.0f, soilMoisture[i] + waterBalance * 0.01f);
        } else {
            // Decrease soil moisture (with floor at 0%)
            soilMoisture[i] = std::max(0.0f, soilMoisture[i] + waterBalance * 0.02f);
        }
    }
    
    // Update river flows based on updated precipitation
    simulateDetailedRiverNetworks();
}

const std::vector<float>& HydrologicalSystem::getPrecipitationData() const {
    return precipitation;
}

const std::vector<float>& HydrologicalSystem::getEvaporationData() const {
    return evaporation;
}

const std::vector<float>& HydrologicalSystem::getRiverFlowData() const {
    return riverFlow;
}

const std::vector<bool>& HydrologicalSystem::getRiverData() const {
    return isRiver;
}

const std::vector<bool>& HydrologicalSystem::getLakeData() const {
    return isLake;
}

} // namespace aeonterracpp