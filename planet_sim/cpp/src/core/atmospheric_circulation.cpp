#include "aeonterracpp/core/climate_system.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace aeonterracpp {

AtmosphericCirculation::AtmosphericCirculation(const Planet& planet, const ClimateParams& params)
    : planet(planet), params(params) {
    // Initialize data vectors with the size of the planet's vertex count
    size_t vertexCount = planet.getVertexCount();
    windVectors.resize(vertexCount, Vec3f::Zero());
    pressure.resize(vertexCount, 1013.25f); // Default sea level pressure in hPa
    temperature.resize(vertexCount, 15.0f); // Default temperature in Celsius
}

void AtmosphericCirculation::initialize() {
    std::cout << "Initializing atmospheric circulation..." << std::endl;
    
    // Calculate base temperature distribution based on latitude
    calculateBaseTemperature();
    
    // Initialize pressure systems
    calculatePressureSystems();
    
    // Calculate initial wind patterns
    calculateWindPatterns();
    
    // Calculate jet streams
    calculateJetStreams();
    
    std::cout << "Atmospheric circulation initialized successfully." << std::endl;
}

void AtmosphericCirculation::simulate(double timeStep, float dayOfYear) {
    // Update seasonal effects
    calculateSeasonalShift(dayOfYear);
    
    // Simulate circulation cells
    simulateHadleyCells();
    simulateFerrelCells();
    simulatePolarCells();
    
    // Update pressure systems
    calculatePressureSystems();
    
    // Update wind patterns
    calculateWindPatterns();
    
    // Add orographic effects from terrain
    calculateOrographicEffects();
    
    // Update jet streams
    calculateJetStreams();
}

Vec3f AtmosphericCirculation::getWindVector(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(windVectors.size())) {
        return windVectors[vertexIdx];
    }
    
    return Vec3f::Zero();
}

float AtmosphericCirculation::getPressure(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(pressure.size())) {
        return pressure[vertexIdx];
    }
    
    return 1013.25f; // Default sea level pressure
}

float AtmosphericCirculation::getTemperature(const Vec3f& position) const {
    // Find closest vertex to the given position
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(temperature.size())) {
        return temperature[vertexIdx];
    }
    
    return 15.0f; // Default temperature
}

void AtmosphericCirculation::calculateBaseTemperature() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < temperature.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Base temperature based on latitude (simplified model)
        // Highest at equator, lowest at poles
        float latRad = lat * M_PI / 180.0f;
        float baseTemp = 30.0f * cos(latRad) * cos(latRad) - 5.0f;
        
        // Adjust for elevation (lapse rate of approximately 6.5°C per km)
        float elevation = planet.getElevation(vertexIdx);
        if (elevation > 0.0f) {
            baseTemp -= elevation * 6.5f;
        }
        
        // Set temperature
        temperature[i] = baseTemp;
    }
}

void AtmosphericCirculation::calculatePressureSystems() {
    const SphericalGrid& grid = planet.getGrid();
    
    // Define pressure belts based on idealized model
    const float equatorialLowPressure = 1008.0f; // Intertropical Convergence Zone (ITCZ)
    const float subtropicalHighPressure = 1020.0f; // Subtropical high
    const float subpolarLowPressure = 1000.0f; // Subpolar low
    const float polarHighPressure = 1025.0f; // Polar high
    
    for (size_t i = 0; i < pressure.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Assign pressure based on latitude belts
        if (absLat < 10.0f) {
            // Equatorial low pressure (ITCZ)
            pressure[i] = equatorialLowPressure;
        } else if (absLat < 35.0f) {
            // Subtropical high pressure
            float factor = (absLat - 10.0f) / 25.0f; // 0-1 from 10° to 35°
            float peakFactor = sin(factor * M_PI); // Peaks in the middle of the zone
            pressure[i] = equatorialLowPressure + peakFactor * (subtropicalHighPressure - equatorialLowPressure);
        } else if (absLat < 65.0f) {
            // Subpolar low pressure
            float factor = (absLat - 35.0f) / 30.0f; // 0-1 from 35° to 65°
            float peakFactor = sin(factor * M_PI); // Peaks in the middle of the zone
            pressure[i] = subtropicalHighPressure - peakFactor * (subtropicalHighPressure - subpolarLowPressure);
        } else {
            // Polar high pressure
            float factor = (absLat - 65.0f) / 25.0f; // 0-1 from 65° to 90°
            pressure[i] = subpolarLowPressure + factor * (polarHighPressure - subpolarLowPressure);
        }
        
        // Adjust pressure for elevation (approximately -11 hPa per 100m)
        float elevation = planet.getElevation(vertexIdx);
        if (elevation > 0.0f) {
            pressure[i] -= elevation * 110.0f;
        }
    }
}

void AtmosphericCirculation::calculateWindPatterns() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < windVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Default wind speed based on latitude band
        float windSpeed = 0.0f;
        Vec3f windDirection = Vec3f::Zero();
        
        // Calculate base wind direction and speed based on latitude bands
        if (absLat < 30.0f) {
            // Trade winds (easterly)
            windSpeed = params.tradeWindSpeed;
            
            // Northern hemisphere trades are northeasterly, southern are southeasterly
            if (lat > 0) {
                // Northern hemisphere - from northeast
                windDirection = Vec3f(-0.866f, -0.5f, 0.0f); // -60° from east
            } else {
                // Southern hemisphere - from southeast
                windDirection = Vec3f(-0.866f, 0.5f, 0.0f); // 60° from east
            }
        } else if (absLat < 60.0f) {
            // Westerlies
            windSpeed = params.westerliesSpeed;
            
            if (lat > 0) {
                // Northern hemisphere - from southwest
                windDirection = Vec3f(0.866f, 0.5f, 0.0f); // 30° from west
            } else {
                // Southern hemisphere - from northwest
                windDirection = Vec3f(0.866f, -0.5f, 0.0f); // -30° from west
            }
        } else {
            // Polar easterlies
            windSpeed = params.polarEasterliesSpeed;
            
            if (lat > 0) {
                // Northern hemisphere - from northeast
                windDirection = Vec3f(-0.866f, -0.5f, 0.0f); // -60° from east
            } else {
                // Southern hemisphere - from southeast
                windDirection = Vec3f(-0.866f, 0.5f, 0.0f); // 60° from east
            }
        }
        
        // Scale direction by wind speed
        windDirection *= windSpeed;
        
        // Project wind vector onto tangent plane at this position
        Vec3f posNorm = pos.normalized();
        float dotProduct = windDirection.dot(posNorm);
        Vec3f tangentWind = windDirection - posNorm * dotProduct;
        
        // Set final wind vector
        windVectors[i] = tangentWind;
    }
}

void AtmosphericCirculation::calculateSeasonalShift(float dayOfYear) {
    // Calculate solar declination (angle between equatorial plane and sun)
    // Ranges from +23.5° at summer solstice to -23.5° at winter solstice
    float solarDeclination = params.axialTilt * sin(2.0f * M_PI * (dayOfYear - 80.0f) / params.yearLength);
    
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < temperature.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Adjust temperature based on season
        // The effect is strongest near poles and minimal at equator
        float seasonalFactor = sin(2.0f * M_PI * (dayOfYear - 80.0f) / params.yearLength);
        float latitudeFactor = sin((90.0f - std::abs(lat)) * M_PI / 180.0f);
        float seasonalShift = seasonalFactor * (1.0f - latitudeFactor) * 20.0f;
        
        // Northern hemisphere has opposite seasons from southern hemisphere
        if (lat < 0.0f) {
            seasonalShift = -seasonalShift;
        }
        
        // Apply seasonal adjustment
        temperature[i] += seasonalShift * params.seasonalVariation;
        
        // Also shift pressure systems
        // ITCZ moves north in northern summer, south in southern summer
        if (std::abs(lat - solarDeclination) < 15.0f) {
            // Near current position of ITCZ
            pressure[i] -= 2.0f; // Strengthen equatorial low
        }
        
        // Adjust wind patterns based on ITCZ shift
        // Just shifting the latitude bands used in wind pattern calculation
        float effectiveLatitude = lat - solarDeclination;
        
        // Adjust wind vectors based on shifted latitude
        if (std::abs(effectiveLatitude) < 30.0f && std::abs(lat) >= 30.0f) {
            // Now in trade wind belt but was in westerlies
            // Change wind direction
            Vec3f adjustedWind;
            if (effectiveLatitude > 0) {
                // Northern hemisphere - from northeast
                adjustedWind = Vec3f(-0.866f, -0.5f, 0.0f) * params.tradeWindSpeed;
            } else {
                // Southern hemisphere - from southeast
                adjustedWind = Vec3f(-0.866f, 0.5f, 0.0f) * params.tradeWindSpeed;
            }
            
            // Project onto tangent plane
            Vec3f posNorm = pos.normalized();
            float dotProduct = adjustedWind.dot(posNorm);
            windVectors[i] = adjustedWind - posNorm * dotProduct;
        }
    }
}

void AtmosphericCirculation::calculateOrographicEffects() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < windVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Check if this is land
        float elevation = planet.getElevation(vertexIdx);
        if (elevation <= 0.0f) {
            continue; // Skip ocean vertices
        }
        
        // Get neighbors to determine slope
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        Vec3f currentPos = grid.getVertex(vertexIdx);
        
        // Calculate average upwind/downwind slope
        Vec3f currentWind = windVectors[i];
        if (currentWind.norm() < 0.1f) {
            continue; // Negligible wind
        }
        
        Vec3f windDirection = currentWind.normalized();
        float maxUpwindSlope = 0.0f;
        float maxDownwindSlope = 0.0f;
        
        for (int neighborIdx : neighbors) {
            Vec3f neighborPos = grid.getVertex(neighborIdx);
            Vec3f toNeighbor = (neighborPos - currentPos).normalized();
            
            // Dot product determines if this is upwind or downwind
            float alignment = windDirection.dot(toNeighbor);
            
            // Calculate slope
            float neighborElevation = planet.getElevation(neighborIdx);
            float elevationDiff = neighborElevation - elevation;
            float distance = (neighborPos - currentPos).norm();
            float slope = elevationDiff / distance;
            
            if (alignment > 0.5f) {
                // Downwind slope
                maxDownwindSlope = std::max(maxDownwindSlope, -slope); // Negative slopes are downward
            } else if (alignment < -0.5f) {
                // Upwind slope
                maxUpwindSlope = std::max(maxUpwindSlope, slope); // Positive slopes are upward
            }
        }
        
        // Adjust wind based on terrain
        float windSpeed = currentWind.norm();
        
        // Upslope decreases speed, downslope increases it
        float speedAdjustment = 1.0f - maxUpwindSlope * 0.5f + maxDownwindSlope * 0.3f;
        speedAdjustment = std::max(0.5f, std::min(1.5f, speedAdjustment));
        
        // High terrain forces wind direction changes
        if (elevation > 2.0f && maxUpwindSlope > 0.2f) {
            // Significant mountain - winds tend to flow around
            // Find tangent direction around mountain
            Vec3f up = currentPos.normalized();
            Vec3f tangent = up.cross(windDirection).normalized();
            
            // Blend original direction with deflection
            float deflectionStrength = std::min(1.0f, maxUpwindSlope);
            windDirection = (windDirection * (1.0f - deflectionStrength) + 
                            tangent * deflectionStrength).normalized();
        }
        
        // Update wind vector
        windVectors[i] = windDirection * (windSpeed * speedAdjustment);
        
        // Orographic effects on temperature and pressure
        // Windward side is cooler/wetter, leeward side is warmer/drier
        if (maxUpwindSlope > 0.1f) {
            // Windward side - adiabatic cooling
            temperature[i] -= maxUpwindSlope * 2.0f;
        } else if (maxDownwindSlope > 0.1f) {
            // Leeward side - foehn effect warming
            temperature[i] += maxDownwindSlope * 3.0f;
        }
    }
}

void AtmosphericCirculation::calculateJetStreams() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < windVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Jet streams occur near polar front (~60°) and subtropical front (~30°)
        float polarJetDistance = std::abs(std::abs(lat) - params.jetStreamLatitude);
        float subtropicalJetDistance = std::abs(std::abs(lat) - (params.jetStreamLatitude / 2.0f));
        
        // Check if we're near a jet stream
        const float jetWidth = 5.0f; // Width of jet stream in degrees
        if (polarJetDistance < jetWidth || subtropicalJetDistance < jetWidth) {
            float jetFactor;
            
            if (polarJetDistance < jetWidth) {
                // Polar jet (~60°) - stronger
                jetFactor = 1.0f - polarJetDistance / jetWidth;
                jetFactor = jetFactor * jetFactor; // Squared for sharper peak
            } else {
                // Subtropical jet (~30°) - weaker
                jetFactor = 0.7f * (1.0f - subtropicalJetDistance / jetWidth);
                jetFactor = jetFactor * jetFactor;
            }
            
            // Enhance wind speed for jet streams (predominantly westerly)
            Vec3f jetDirection;
            if (lat > 0) {
                // Northern hemisphere
                jetDirection = Vec3f(1.0f, 0.0f, 0.0f); // West to East
            } else {
                // Southern hemisphere
                jetDirection = Vec3f(1.0f, 0.0f, 0.0f); // West to East
            }
            
            // Project onto tangent plane
            Vec3f posNorm = pos.normalized();
            float dotProduct = jetDirection.dot(posNorm);
            Vec3f tangentJet = jetDirection - posNorm * dotProduct;
            tangentJet.normalize();
            
            // Apply jet stream effect (add to existing wind)
            // Jet streams can reach 70-150 m/s (factor of 2-5x normal westerlies)
            float jetSpeed = params.westerliesSpeed * 3.0f * jetFactor;
            windVectors[i] += tangentJet * jetSpeed;
        }
    }
}

void AtmosphericCirculation::simulateHadleyCells() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < windVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Hadley cells operate between equator and ~30°
        if (absLat < params.hadleyCellWidth) {
            // Calculate vertical motion (rising at equator, sinking at ~30°)
            float cellPosition = absLat / params.hadleyCellWidth; // 0 at equator, 1 at cell edge
            float verticalMotion = sin(cellPosition * M_PI); // +1 at middle of cell, 0 at edges
            
            // Rising air cools, sinking air warms (adiabatic processes)
            // This affects temperature in addition to base latitude effect
            if (cellPosition < 0.5f) {
                // Rising air near equator (cooling)
                temperature[i] -= verticalMotion * 2.0f;
            } else {
                // Sinking air near 30° (warming)
                temperature[i] += verticalMotion * 2.0f;
            }
            
            // Pressure is affected too - low at equator, high at 30°
            if (cellPosition < 0.3f) {
                // Equatorial low (ITCZ)
                pressure[i] -= (1.0f - cellPosition / 0.3f) * 5.0f;
            } else if (cellPosition > 0.7f) {
                // Subtropical high
                pressure[i] += ((cellPosition - 0.7f) / 0.3f) * 7.0f;
            }
        }
    }
}

void AtmosphericCirculation::simulateFerrelCells() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < windVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Ferrel cells operate between ~30° and ~60°
        if (absLat >= params.hadleyCellWidth && absLat < (params.hadleyCellWidth + params.ferrelCellWidth)) {
            // Calculate position within cell (0 at 30°, 1 at 60°)
            float cellPosition = (absLat - params.hadleyCellWidth) / params.ferrelCellWidth;
            
            // Ferrel cell has sinking air at 30° and rising air at 60° (opposite of Hadley)
            float verticalMotion = sin(cellPosition * M_PI);
            
            // Adjust temperature and pressure based on vertical motion
            if (cellPosition < 0.5f) {
                // Sinking air near 30° (already handled in Hadley cell)
            } else {
                // Rising air near 60° (cooling)
                temperature[i] -= verticalMotion * 1.5f;
            }
            
            // Pressure is high at 30° (from Hadley) and low at 60° (subpolar low)
            if (cellPosition > 0.7f) {
                // Subpolar low
                pressure[i] -= ((cellPosition - 0.7f) / 0.3f) * 6.0f;
            }
        }
    }
}

void AtmosphericCirculation::simulatePolarCells() {
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < windVectors.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        
        // Convert to latitude/longitude
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        float absLat = std::abs(lat);
        
        // Polar cells operate between ~60° and 90°
        if (absLat >= (params.hadleyCellWidth + params.ferrelCellWidth)) {
            // Calculate position within cell (0 at 60°, 1 at 90°)
            float cellPosition = (absLat - (params.hadleyCellWidth + params.ferrelCellWidth)) / 
                               (90.0f - (params.hadleyCellWidth + params.ferrelCellWidth));
            
            // Polar cell has rising air at 60° and sinking air at poles
            float verticalMotion = sin(cellPosition * M_PI);
            
            // Adjust temperature and pressure
            if (cellPosition > 0.5f) {
                // Sinking air at poles (warming, but still very cold)
                temperature[i] += verticalMotion * 1.0f;
                
                // High pressure at poles
                pressure[i] += verticalMotion * 5.0f;
            }
        }
    }
}

float AtmosphericCirculation::calculateCoriolisParameter(float latitude) const {
    // Coriolis parameter f = 2Ω sin(φ)
    // where Ω is Earth's rotation rate and φ is latitude
    
    // Earth's rotation rate (rad/s)
    const float earthRotationRate = 7.2921159e-5f;
    
    // Scale by planet's rotation (day length relative to Earth)
    float planetRotationRate = earthRotationRate * (24.0f / params.dayLength);
    
    // Calculate Coriolis parameter
    float latRad = latitude * M_PI / 180.0f;
    return 2.0f * planetRotationRate * sin(latRad);
}

const std::vector<Vec3f>& AtmosphericCirculation::getWindData() const {
    return windVectors;
}

const std::vector<float>& AtmosphericCirculation::getPressureData() const {
    return pressure;
}

const std::vector<float>& AtmosphericCirculation::getTemperatureData() const {
    return temperature;
}

} // namespace aeonterracpp