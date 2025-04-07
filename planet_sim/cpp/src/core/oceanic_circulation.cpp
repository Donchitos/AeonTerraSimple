#include "aeonterracpp/core/climate_system.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace aeonterracpp {

OceanicCirculation::OceanicCirculation(const Planet& planet, 
                                       const AtmosphericCirculation& atmosphere,
                                       const ClimateParams& params)
    : planet(planet), atmosphere(atmosphere), params(params) {
    // Initialize data vectors with the size of the planet's vertex count
    size_t vertexCount = planet.getVertexCount();
    currentVectors.resize(vertexCount, Vec3f::Zero());
    seaSurfaceTemp.resize(vertexCount, 15.0f); // Default temperature in Celsius
    salinity.resize(vertexCount, 35.0f); // Default salinity in psu (practical salinity units)
    upwelling.resize(vertexCount, 0.0f);
}

void OceanicCirculation::initialize() {
    std::cout << "Initializing oceanic circulation..." << std::endl;
    
    // Initialize sea surface temperatures based on latitude
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < seaSurfaceTemp.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Base SST based on latitude (simplified model)
        // Highest at equator, lowest at poles
        float latRad = lat * M_PI / 180.0f;
        float baseTemp = 28.0f * cos(latRad) * cos(latRad) - 2.0f;
        
        // Set temperature
        seaSurfaceTemp[i] = std::max(-2.0f, baseTemp); // Ocean doesn't go below -2°C (freezing point of seawater)
    }
    
    // Initialize ocean currents
    simulateGyres();
    simulateEquatorialCurrents();
    calculateCoastalCurrents();
    
    // Initialize deep ocean circulation
    simulateThermohalineCirculation();
    
    // Calculate upwelling regions
    calculateUpwellingRegions();
    
    std::cout << "Oceanic circulation initialized successfully." << std::endl;
}

void OceanicCirculation::simulate(double timeStep, float dayOfYear) {
    // Update gyre circulation
    simulateGyres();
    
    // Update equatorial currents
    simulateEquatorialCurrents();
    
    // Update coastal currents
    calculateCoastalCurrents();
    
    // Update thermohaline circulation
    simulateThermohalineCirculation();
    
    // Update upwelling regions
    calculateUpwellingRegions();
    
    // Calculate heat transport by ocean currents
    calculateHeatTransport();
    
    // Apply seasonal effects (e.g., sea ice formation, mixed layer depth changes)
    float seasonalFactor = sin(2.0f * M_PI * (dayOfYear - 80.0f) / params.yearLength);
    applySeasonalEffects(seasonalFactor);
}

bool OceanicCirculation::isOcean(int vertexIdx) const {
    // Check if this vertex represents ocean
    return planet.getElevation(vertexIdx) < 0.0f;
}

float OceanicCirculation::getOceanDepth(int vertexIdx) const {
    if (!isOcean(vertexIdx)) {
        return 0.0f;
    }
    
    return -planet.getElevation(vertexIdx); // Convert negative elevation to positive depth
}

Vec3f OceanicCirculation::getCurrentVector(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(currentVectors.size()) && isOcean(vertexIdx)) {
        return currentVectors[vertexIdx];
    }
    
    return Vec3f::Zero();
}

float OceanicCirculation::getSeaSurfaceTemperature(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(seaSurfaceTemp.size()) && isOcean(vertexIdx)) {
        return seaSurfaceTemp[vertexIdx];
    }
    
    return 15.0f; // Default temperature
}

void OceanicCirculation::simulateGyres() {
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<Vec3f>& windVectors = atmosphere.getWindData();
    
    for (size_t i = 0; i < currentVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Ocean gyres form in response to wind patterns and Coriolis force
        // Subtropical gyres between ~10° and ~45°
        // Subpolar gyres between ~45° and ~65°
        
        if (absLat < 45.0f && absLat > 10.0f) {
            // Subtropical gyre region
            
            // Start with wind-driven current
            Vec3f windVector = Vec3f::Zero();
            if (vertexIdx < static_cast<int>(windVectors.size())) {
                windVector = windVectors[vertexIdx];
            }
            
            // Calculate wind stress
            float windStress = calculateWindStress(windVector);
            
            // Calculate Ekman transport (90° to the right of wind in Northern hemisphere,
            // 90° to the left in Southern hemisphere)
            Vec3f ekmanDirection;
            if (lat > 0) {
                // Northern hemisphere - 90° clockwise from wind
                ekmanDirection = Vec3f(windVector.y(), -windVector.x(), 0.0f);
            } else {
                // Southern hemisphere - 90° counterclockwise from wind
                ekmanDirection = Vec3f(-windVector.y(), windVector.x(), 0.0f);
            }
            
            // Project onto tangent plane
            Vec3f posNorm = pos.normalized();
            float dotProduct = ekmanDirection.dot(posNorm);
            Vec3f tangentEkman = ekmanDirection - posNorm * dotProduct;
            tangentEkman.normalize();
            
            // Gyre strength increases toward center (~30°) and decreases toward edges
            float gyreStrength = 1.0f - std::abs((absLat - 30.0f) / 20.0f);
            gyreStrength = std::max(0.0f, gyreStrength);
            
            // Set current vector - combining wind stress and gyre circulation
            float currentSpeed = windStress * 0.3f * gyreStrength * params.gyreStrength;
            currentVectors[i] = tangentEkman * currentSpeed;
            
            // Subtropical gyres rotate clockwise in Northern hemisphere, counterclockwise in Southern
            if ((lat > 0 && absLat > 30.0f) || (lat < 0 && absLat < 30.0f)) {
                // Reverse direction in outer portions of gyre to create circulation
                currentVectors[i] = -currentVectors[i];
            }
        }
        else if (absLat >= 45.0f && absLat < 65.0f) {
            // Subpolar gyre region
            
            // Subpolar gyres rotate counterclockwise in Northern hemisphere, clockwise in Southern
            Vec3f gyreDirection;
            if (lat > 0) {
                // Northern hemisphere - counterclockwise
                gyreDirection = Vec3f(-lon / std::abs(lon), 0.0f, 0.0f);
            } else {
                // Southern hemisphere - clockwise
                gyreDirection = Vec3f(lon / std::abs(lon), 0.0f, 0.0f);
            }
            
            // Project onto tangent plane
            Vec3f posNorm = pos.normalized();
            float dotProduct = gyreDirection.dot(posNorm);
            Vec3f tangentGyre = gyreDirection - posNorm * dotProduct;
            tangentGyre.normalize();
            
            // Weaker than subtropical gyres
            float gyreStrength = 1.0f - std::abs((absLat - 55.0f) / 10.0f);
            gyreStrength = std::max(0.0f, gyreStrength);
            
            // Set current vector
            float currentSpeed = 0.2f * gyreStrength * params.gyreStrength;
            currentVectors[i] = tangentGyre * currentSpeed;
        }
    }
}

void OceanicCirculation::simulateEquatorialCurrents() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < currentVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Equatorial currents operate near the equator
        if (absLat < 10.0f) {
            // North and South Equatorial Currents (westward)
            // Equatorial Countercurrent (~5°) (eastward)
            
            Vec3f currentDirection;
            float currentSpeed;
            
            if (absLat < 3.0f) {
                // Directly at equator - South Equatorial Current (westward)
                currentDirection = Vec3f(-1.0f, 0.0f, 0.0f);
                currentSpeed = 0.5f;
            } else if (absLat >= 3.0f && absLat < 7.0f) {
                // Equatorial Countercurrent (eastward)
                currentDirection = Vec3f(1.0f, 0.0f, 0.0f);
                currentSpeed = 0.3f;
            } else {
                // North/South Equatorial Current (westward)
                currentDirection = Vec3f(-1.0f, 0.0f, 0.0f);
                currentSpeed = 0.4f;
            }
            
            // Project onto tangent plane
            Vec3f posNorm = pos.normalized();
            float dotProduct = currentDirection.dot(posNorm);
            Vec3f tangentCurrent = currentDirection - posNorm * dotProduct;
            tangentCurrent.normalize();
            
            // Set current vector
            currentVectors[i] = tangentCurrent * currentSpeed * params.gyreStrength;
        }
    }
}

void OceanicCirculation::simulateThermohalineCirculation() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Identify deep water formation regions (high latitudes, high salinity)
    std::vector<int> deepWaterFormationRegions;
    
    for (size_t i = 0; i < seaSurfaceTemp.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        Vec3f pos = grid.getVertex(vertexIdx);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Deep water forms at high latitudes (>60°) with cold temperature
        if (absLat > 60.0f && seaSurfaceTemp[i] < 2.0f) {
            // Check for high salinity
            if (salinity[i] > 34.5f) {
                deepWaterFormationRegions.push_back(vertexIdx);
                
                // Mark as downwelling region
                upwelling[i] = -1.0f * params.deepOceanCirculation;
            }
        }
    }
    
    // Simplified global conveyor belt (thermohaline circulation)
    // This implementation just marks upwelling regions rather than simulating the actual flow
    
    for (size_t i = 0; i < currentVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        // Deep ocean is where deep currents predominate
        float depth = getOceanDepth(vertexIdx);
        if (depth < 2.0f) {
            continue; // Too shallow for deep ocean circulation
        }
        
        Vec3f pos = grid.getVertex(vertexIdx);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Identify upwelling regions
        // Upwelling occurs primarily in the eastern boundaries of ocean basins
        bool isEasternBoundary = false;
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        
        for (int neighborIdx : neighbors) {
            if (!isOcean(neighborIdx)) {
                // This is at an ocean-land boundary
                
                // Determine if it's an eastern boundary
                auto [neighborLat, neighborLon] = grid.cartesianToLatLon(grid.getVertex(neighborIdx));
                
                // Eastern boundaries are where land is to the east of ocean
                if ((lon > 0 && neighborLon > lon) || (lon < 0 && neighborLon > lon && neighborLon < 0.0f)) {
                    isEasternBoundary = true;
                    break;
                }
            }
        }
        
        // Mark upwelling regions
        if (isEasternBoundary && absLat > 5.0f && absLat < 45.0f) {
            upwelling[i] = 0.5f * params.deepOceanCirculation;
        }
    }
}

void OceanicCirculation::calculateCoastalCurrents() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Identify coastal vertices
    for (size_t i = 0; i < currentVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        // Check if this is a coastal vertex (adjacent to land)
        bool isCoastal = false;
        Vec3f coastDirection = Vec3f::Zero();
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            if (!isOcean(neighborIdx)) {
                isCoastal = true;
                
                // Direction from land to ocean
                coastDirection += grid.getVertex(vertexIdx) - grid.getVertex(neighborIdx);
            }
        }
        
        if (isCoastal && coastDirection.norm() > 0.01f) {
            coastDirection.normalize();
            Vec3f pos = grid.getVertex(vertexIdx);
            auto [lat, lon] = grid.cartesianToLatLon(pos);
            
            // Coastal currents follow coastlines
            // In Northern hemisphere, they keep coast on the right
            // In Southern hemisphere, they keep coast on the left
            Vec3f coastalCurrent;
            
            if (lat > 0) {
                // Northern hemisphere - coast on right (90° clockwise from coast direction)
                coastalCurrent = Vec3f(coastDirection.y(), -coastDirection.x(), 0.0f);
            } else {
                // Southern hemisphere - coast on left (90° counterclockwise from coast direction)
                coastalCurrent = Vec3f(-coastDirection.y(), coastDirection.x(), 0.0f);
            }
            
            // Project onto tangent plane
            Vec3f posNorm = pos.normalized();
            float dotProduct = coastalCurrent.dot(posNorm);
            Vec3f tangentCurrent = coastalCurrent - posNorm * dotProduct;
            tangentCurrent.normalize();
            
            // Blend with existing current
            float coastalFactor = 0.5f;
            currentVectors[i] = currentVectors[i] * (1.0f - coastalFactor) + 
                               tangentCurrent * 0.3f * coastalFactor * params.gyreStrength;
        }
    }
}

void OceanicCirculation::calculateUpwellingRegions() {
    const SphericalGrid& grid = planet.getGrid();
    const std::vector<Vec3f>& windVectors = atmosphere.getWindData();
    
    for (size_t i = 0; i < currentVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        // Skip if already marked as upwelling/downwelling by thermohaline circulation
        if (std::abs(upwelling[i]) > 0.01f) {
            continue;
        }
        
        Vec3f pos = grid.getVertex(vertexIdx);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Calculate Ekman transport from wind
        Vec3f windVector = Vec3f::Zero();
        if (vertexIdx < static_cast<int>(windVectors.size())) {
            windVector = windVectors[vertexIdx];
        }
        
        // Check for coastal upwelling
        bool isCoastal = false;
        Vec3f coastDirection = Vec3f::Zero();
        
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            if (!isOcean(neighborIdx)) {
                isCoastal = true;
                
                // Direction from land to ocean
                coastDirection += grid.getVertex(vertexIdx) - grid.getVertex(neighborIdx);
            }
        }
        
        if (isCoastal && coastDirection.norm() > 0.01f && windVector.norm() > 0.01f) {
            coastDirection.normalize();
            windVector.normalize();
            
            // Calculate angle between wind and coast
            float dotProduct = windVector.dot(coastDirection);
            
            // Upwelling occurs when wind blows parallel to coast with coast on left (NH) or right (SH)
            float angle = std::acos(std::min(1.0f, std::max(-1.0f, dotProduct)));
            
            if (std::abs(angle - M_PI/2) < M_PI/6) { // Within 30° of perpendicular
                // Potential upwelling region
                
                // Determine if wind direction causes upwelling
                Vec3f crossProduct = windVector.cross(coastDirection);
                float upwellingFactor = crossProduct.dot(pos.normalized());
                
                if ((lat > 0 && upwellingFactor > 0) || (lat < 0 && upwellingFactor < 0)) {
                    // Conditions right for coastal upwelling
                    upwelling[i] = 0.7f * params.upwellingStrength;
                } else {
                    // Conditions right for coastal downwelling
                    upwelling[i] = -0.3f * params.upwellingStrength;
                }
            }
        }
        
        // Equatorial upwelling
        float absLat = std::abs(lat);
        if (absLat < 1.0f) {
            upwelling[i] = 0.5f * params.upwellingStrength;
        }
    }
}

void OceanicCirculation::calculateHeatTransport() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Calculate heat transport due to ocean currents
    // This is a simple model where heat is advected by currents
    
    // Create temporary copy of sea surface temperatures
    std::vector<float> newTemps = seaSurfaceTemp;
    
    for (size_t i = 0; i < currentVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        Vec3f currentVector = currentVectors[i];
        float currentSpeed = currentVector.norm();
        
        if (currentSpeed < 0.01f) {
            continue; // Negligible current
        }
        
        // Find upstream position
        Vec3f pos = grid.getVertex(vertexIdx);
        Vec3f currentDir = currentVector.normalized();
        
        // Approximate upstream position
        Vec3f upstreamPos = pos - currentDir * 0.1f;
        
        // Find closest vertex to upstream position
        int upstreamIdx = grid.findClosestVertex(upstreamPos);
        
        if (upstreamIdx >= 0 && upstreamIdx < static_cast<int>(seaSurfaceTemp.size()) && isOcean(upstreamIdx)) {
            // Advect heat from upstream
            float upstreamTemp = seaSurfaceTemp[upstreamIdx];
            float localTemp = seaSurfaceTemp[i];
            
            // Mix temperatures based on current speed
            // Stronger currents transport more heat
            float mixFactor = std::min(0.2f, currentSpeed * 0.1f);
            newTemps[i] = localTemp * (1.0f - mixFactor) + upstreamTemp * mixFactor;
        }
        
        // Additional heating/cooling effects
        
        // Upwelling brings cooler water to surface
        if (upwelling[i] > 0.0f) {
            newTemps[i] -= upwelling[i] * 0.5f;
        }
        
        // Downwelling doesn't directly affect SST much
    }
    
    // Update temperatures
    seaSurfaceTemp = newTemps;
}

float OceanicCirculation::calculateWindStress(const Vec3f& windVector) const {
    // Wind stress is approximately proportional to wind speed squared
    float windSpeed = windVector.norm();
    return windSpeed * windSpeed * 0.0025f; // Approximate drag coefficient
}

void OceanicCirculation::applySeasonalEffects(float seasonalFactor) {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < seaSurfaceTemp.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only process ocean cells
        if (!isOcean(vertexIdx)) {
            continue;
        }
        
        Vec3f pos = grid.getVertex(vertexIdx);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Seasonal temperature variations are smaller in the ocean than on land
        // due to water's high heat capacity
        float seasonalAdjustment = seasonalFactor * (absLat / 90.0f) * 3.0f;
        
        // Northern and southern hemispheres have opposite seasons
        if (lat < 0) {
            seasonalAdjustment = -seasonalAdjustment;
        }
        
        // Apply the seasonal adjustment
        seaSurfaceTemp[i] += seasonalAdjustment;
        
        // Sea ice forms when temperature drops below freezing
        if (seaSurfaceTemp[i] < -1.8f) {
            seaSurfaceTemp[i] = -1.8f; // Freezing point of typical seawater
            
            // Increase salinity due to ice formation (brine rejection)
            salinity[i] += 0.01f;
        }
    }
}

const std::vector<Vec3f>& OceanicCirculation::getCurrentData() const {
    return currentVectors;
}

const std::vector<float>& OceanicCirculation::getSeaSurfaceTemperatureData() const {
    return seaSurfaceTemp;
}

} // namespace aeonterracpp