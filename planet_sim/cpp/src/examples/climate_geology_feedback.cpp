#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/tectonics.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace aeonterracpp {

/**
 * @brief ClimateGeologyFeedback implements bidirectional feedback between climate and geological processes
 * 
 * This system simulates:
 * 1. How elevation and continental structure affect climate patterns
 * 2. How climate affects erosion, weathering, and sedimentation
 * 3. Long-term feedback loops between geology and climate
 */
class ClimateGeologyFeedback {
public:
    /**
     * @brief Climate types based on temperature and precipitation
     */
    enum class ClimateType {
        POLAR,
        CONTINENTAL,
        TEMPERATE,
        DESERT,
        TROPICAL,
        OCEANIC,
        MOUNTAIN
    };
    
    /**
     * @brief Constructor
     * 
     * @param planet Reference to the planet
     * @param params Simulation parameters
     */
    ClimateGeologyFeedback(Planet& planet, const SimulationParameters& params) 
        : planet(planet), params(params), axialTilt(23.5f) {
        
        // Initialize climate arrays
        size_t vertexCount = planet.getVertexCount();
        temperature.resize(vertexCount, 15.0f);     // Default 15°C
        precipitation.resize(vertexCount, 1000.0f);  // Default 1000mm
        windDirection.resize(vertexCount, Vec3f::Zero());
        climateTypes.resize(vertexCount, ClimateType::TEMPERATE);
        weatheringRate.resize(vertexCount, 1.0f);    // Default rate multiplier
        evaporationRate.resize(vertexCount, 1.0f);   // Default rate multiplier
        
        // Calculate initial climate
        updateClimateFromGeology();
    }
    
    /**
     * @brief Update climate patterns based on geology
     * Takes into account elevation, latitude, continentality, etc.
     */
    void updateClimateFromGeology() {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        // Reset cumulative effects
        std::vector<float> baseTemperature(vertexCount);
        std::vector<float> basePrecipitation(vertexCount);
        
        // 1. Calculate base climate patterns from latitude
        calculateLatitudinalClimateBands();
        
        // 2. Calculate wind patterns (important for rain shadows)
        calculateWindPatterns();
        
        // 3. Apply elevation effects
        calculateElevationEffects();
        
        // 4. Calculate rain shadow effects from mountains
        calculateRainShadowEffect();
        
        // 5. Calculate coastal and oceanic influences
        calculateOceanicInfluence();
        
        // 6. Apply orographic lifting (mountains cause increased precipitation)
        calculateOrographicLifting();
        
        // 7. Determine climate types for each vertex
        for (size_t i = 0; i < vertexCount; ++i) {
            climateTypes[i] = determineClimateType(static_cast<int>(i));
        }
        
        std::cout << "Updated climate patterns based on geological features" << std::endl;
        printClimateStatistics();
    }
    
    /**
     * @brief Update geological processes based on climate
     * Takes into account precipitation, temperature, etc.
     */
    void updateGeologyFromClimate(double timeStep) {
        size_t vertexCount = planet.getVertexCount();
        
        // 1. Update weathering rates based on climate
        calculateWeatheringRates();
        
        // 2. Apply differential erosion based on precipitation and temperature
        applyClimateBasedErosion(timeStep);
        
        // 3. Apply climate-specific sedimentation patterns
        applyClimateSedimentation(timeStep);
        
        std::cout << "Updated geological processes based on climate for " 
                  << timeStep << " million years" << std::endl;
    }
    
    /**
     * @brief Run a full feedback cycle for a given time step
     * 
     * @param timeStep Time in million years
     */
    void runFeedbackCycle(double timeStep) {
        // Update climate based on current geology
        updateClimateFromGeology();
        
        // Update geology based on climate
        updateGeologyFromClimate(timeStep);
        
        std::cout << "Completed climate-geology feedback cycle for " 
                  << timeStep << " million years" << std::endl;
    }
    
    /**
     * @brief Get the climate type at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return ClimateType Climate classification
     */
    ClimateType getClimateType(int vertexIdx) const {
        return climateTypes[vertexIdx];
    }
    
    /**
     * @brief Get a string description of a climate type
     * 
     * @param type Climate type
     * @return std::string Description
     */
    std::string getClimateTypeName(ClimateType type) const {
        switch (type) {
            case ClimateType::POLAR: return "Polar";
            case ClimateType::CONTINENTAL: return "Continental";
            case ClimateType::TEMPERATE: return "Temperate";
            case ClimateType::DESERT: return "Desert";
            case ClimateType::TROPICAL: return "Tropical";
            case ClimateType::OCEANIC: return "Oceanic";
            case ClimateType::MOUNTAIN: return "Mountain";
            default: return "Unknown";
        }
    }
    
    /**
     * @brief Get temperature at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Temperature in Celsius
     */
    float getTemperature(int vertexIdx) const {
        return temperature[vertexIdx];
    }
    
    /**
     * @brief Get precipitation at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Precipitation in mm/year
     */
    float getPrecipitation(int vertexIdx) const {
        return precipitation[vertexIdx];
    }
    
private:
    Planet& planet;
    SimulationParameters params;
    float axialTilt;  // Planet's axial tilt in degrees
    
    // Climate data for each vertex
    std::vector<float> temperature;       // Temperature in °C
    std::vector<float> precipitation;     // Precipitation in mm/year
    std::vector<Vec3f> windDirection;     // Prevailing wind direction
    std::vector<ClimateType> climateTypes; // Climate classification
    std::vector<float> weatheringRate;    // Weathering rate multiplier
    std::vector<float> evaporationRate;   // Evaporation rate multiplier
    
    /**
     * @brief Calculate climate bands based on latitude
     * This is the starting point for climate simulation
     */
    void calculateLatitudinalClimateBands() {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            
            // Get position and convert to latitude/longitude
            Vec3f pos = grid.getVertex(vertexIdx);
            auto [lat, lon] = grid.cartesianToLatLon(pos);
            
            // Temperature varies with latitude (simplified model)
            // Highest at equator, lowest at poles
            float tempLatFactor = std::cos(toRadians(lat));
            temperature[i] = 30.0f * tempLatFactor * tempLatFactor - 5.0f;
            
            // Adjust for axial tilt (seasonality)
            float seasonalFactor = 1.0f;  // Would vary with time of year
            temperature[i] += 5.0f * std::sin(toRadians(lat)) * seasonalFactor;
            
            // Precipitation varies with latitude (simplistic model)
            // Highest at equator and mid-latitudes, lower at horse latitudes and poles
            float latAbs = std::abs(lat);
            float precipFactor = 0.0f;
            
            if (latAbs < 10.0f) {
                // Equatorial (ITCZ)
                precipFactor = 1.0f;
            } else if (latAbs < 30.0f) {
                // Horse latitudes (dry)
                precipFactor = 0.3f;
            } else if (latAbs < 60.0f) {
                // Mid-latitudes (moderately wet)
                precipFactor = 0.7f;
            } else {
                // Polar (dry)
                precipFactor = 0.2f;
            }
            
            precipitation[i] = 2000.0f * precipFactor;
        }
    }
    
    /**
     * @brief Calculate global wind patterns
     * Models trade winds, westerlies, polar easterlies
     */
    void calculateWindPatterns() {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            
            // Get position and convert to latitude
            Vec3f pos = grid.getVertex(vertexIdx);
            auto [lat, lon] = grid.cartesianToLatLon(pos);
            
            // Wind direction varies by latitude
            Vec3f eastward = Vec3f(-std::sin(toRadians(lon)), std::cos(toRadians(lon)), 0.0f);
            Vec3f westward = -eastward;
            Vec3f northward = Vec3f(
                -std::cos(toRadians(lon)) * std::sin(toRadians(lat)),
                -std::sin(toRadians(lon)) * std::sin(toRadians(lat)),
                std::cos(toRadians(lat))
            );
            Vec3f southward = -northward;
            
            // Apply Coriolis effect based on latitude
            if (std::abs(lat) < 30.0f) {
                // Trade winds (easterlies)
                windDirection[i] = eastward * 0.8f + 
                                  (lat < 0 ? northward : southward) * 0.2f;
            } else if (std::abs(lat) < 60.0f) {
                // Westerlies
                windDirection[i] = westward * 0.8f + 
                                  (lat < 0 ? southward : northward) * 0.2f;
            } else {
                // Polar easterlies
                windDirection[i] = eastward * 0.8f + 
                                  (lat < 0 ? northward : southward) * 0.2f;
            }
            
            // Normalize to get direction only
            windDirection[i].normalize();
        }
    }
    
    /**
     * @brief Apply elevation effects on temperature and precipitation
     * Higher elevations are colder and often drier
     */
    void calculateElevationEffects() {
        size_t vertexCount = planet.getVertexCount();
        
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Only apply to land areas
            if (elevation > 0.0f) {
                // Temperature decreases with elevation (lapse rate: ~6.5°C per km)
                temperature[i] -= elevation * 6.5f;
                
                // Precipitation generally decreases at very high elevations
                if (elevation > 3.0f) {
                    precipitation[i] *= (1.0f - (elevation - 3.0f) * 0.15f);
                    precipitation[i] = std::max(precipitation[i], 200.0f); // Minimum precipitation
                }
            }
        }
    }
    
    /**
     * @brief Calculate rain shadow effects from mountain ranges
     * Areas downwind of mountains receive less precipitation
     */
    void calculateRainShadowEffect() {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        // Temporary array to store rain shadow factors
        std::vector<float> rainShadowFactor(vertexCount, 1.0f);
        
        // Find high-elevation vertices (mountains)
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Only mountains create rain shadows
            if (elevation > 1.5f) {
                // Find vertices downwind of this mountain
                Vec3f mountainPos = grid.getVertex(vertexIdx);
                Vec3f windDir = windDirection[i];
                
                // Approximate rain shadow area
                for (size_t j = 0; j < vertexCount; ++j) {
                    // Skip self
                    if (i == j) continue;
                    
                    // Calculate alignment with wind direction
                    Vec3f posJ = grid.getVertex(static_cast<int>(j));
                    Vec3f relativePos = posJ - mountainPos;
                    
                    // If this vertex is downwind of the mountain
                    float alignmentWithWind = relativePos.normalized().dot(windDir);
                    float distance = relativePos.norm();
                    
                    // Vertices downwind and within shadow distance are affected
                    if (alignmentWithWind > 0.7f && distance < elevation * 200.0f) {
                        // Stronger effect close to the mountain and at higher elevations
                        float shadowStrength = (elevation - 1.5f) * 0.4f; // 0-1 scale
                        float distanceFactor = 1.0f - (distance / (elevation * 200.0f));
                        float totalEffect = shadowStrength * distanceFactor;
                        
                        // Accumulate shadow effect (multiplicative, so stronger shadows win)
                        rainShadowFactor[j] *= (1.0f - totalEffect);
                    }
                }
            }
        }
        
        // Apply rain shadow effect to precipitation
        for (size_t i = 0; i < vertexCount; ++i) {
            precipitation[i] *= rainShadowFactor[i];
        }
    }
    
    /**
     * @brief Calculate oceanic influence on climate
     * Coastal areas have moderated temperatures and often higher precipitation
     */
    void calculateOceanicInfluence() {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        // Find coast vertices and calculate continental interiors
        std::vector<bool> isCoast(vertexCount, false);
        std::vector<float> continentalityFactor(vertexCount, 0.0f);
        
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Process only land areas
            if (elevation > 0.0f) {
                const auto& neighbors = grid.getNeighbors(vertexIdx);
                bool hasOceanNeighbor = false;
                
                // Check if this is a coastal vertex
                for (int neighborIdx : neighbors) {
                    float neighborElevation = planet.getElevation(neighborIdx);
                    if (neighborElevation < 0.0f) {
                        hasOceanNeighbor = true;
                        break;
                    }
                }
                
                isCoast[i] = hasOceanNeighbor;
            }
        }
        
        // Calculate continentality (distance from coast)
        // Simple approach: multiple passes to propagate distance
        const int MAX_PASSES = 10;
        
        for (int pass = 0; pass < MAX_PASSES; pass++) {
            for (size_t i = 0; i < vertexCount; ++i) {
                int vertexIdx = static_cast<int>(i);
                float elevation = planet.getElevation(vertexIdx);
                
                // Skip ocean and coast
                if (elevation <= 0.0f || isCoast[i]) {
                    continue;
                }
                
                const auto& neighbors = grid.getNeighbors(vertexIdx);
                
                // Find minimum continentality among neighbors and add 0.1
                float minNeighborContinentality = continentalityFactor[i];
                for (int neighborIdx : neighbors) {
                    if (planet.getElevation(neighborIdx) > 0.0f) {
                        minNeighborContinentality = std::min(minNeighborContinentality, 
                                                           continentalityFactor[neighborIdx]);
                    }
                }
                
                continentalityFactor[i] = minNeighborContinentality + 0.1f;
            }
        }
        
        // Normalize continentality
        float maxContinentality = 0.0f;
        for (size_t i = 0; i < vertexCount; ++i) {
            maxContinentality = std::max(maxContinentality, continentalityFactor[i]);
        }
        
        if (maxContinentality > 0.0f) {
            for (size_t i = 0; i < vertexCount; ++i) {
                continentalityFactor[i] /= maxContinentality;
            }
        }
        
        // Apply oceanic influence
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Ocean and coastal areas
            if (elevation < 0.0f) {
                // Ocean has stable temperature
                temperature[i] = std::max(0.0f, temperature[i]);
                temperature[i] = std::min(28.0f, temperature[i]);
            } 
            else if (isCoast[i]) {
                // Coasts have moderate temperature
                float originalTemp = temperature[i];
                float moderatedTemp = std::max(5.0f, std::min(27.0f, originalTemp));
                temperature[i] = originalTemp * 0.3f + moderatedTemp * 0.7f;
                
                // Coasts often have higher precipitation due to onshore winds
                precipitation[i] *= 1.3f;
            }
            else {
                // Continental interiors have more extreme temperatures
                float extremeFactor = 0.3f * continentalityFactor[i];
                if (temperature[i] > 15.0f) {
                    temperature[i] += extremeFactor * 10.0f; // Hotter in summer
                } else {
                    temperature[i] -= extremeFactor * 15.0f; // Colder in winter
                }
                
                // Continental interiors tend to be drier
                precipitation[i] *= (1.0f - 0.5f * continentalityFactor[i]);
            }
        }
    }
    
    /**
     * @brief Calculate orographic lifting (mountains forcing air to rise)
     * Windward sides of mountains get increased precipitation
     */
    void calculateOrographicLifting() {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Skip ocean areas
            if (elevation <= 0.0f) {
                continue;
            }
            
            const auto& neighbors = grid.getNeighbors(vertexIdx);
            
            // Find upwind neighbor
            Vec3f pos = grid.getVertex(vertexIdx);
            Vec3f wind = windDirection[i];
            
            float maxUphillSlope = 0.0f;
            
            for (int neighborIdx : neighbors) {
                Vec3f neighborPos = grid.getVertex(neighborIdx);
                Vec3f toNeighbor = (neighborPos - pos).normalized();
                
                // Calculate if this neighbor is upwind
                float windAlignment = -toNeighbor.dot(wind);
                
                if (windAlignment > 0.5f) {
                    // Calculate elevation difference
                    float neighborElevation = planet.getElevation(neighborIdx);
                    float elevationDiff = elevation - neighborElevation;
                    
                    if (elevationDiff > 0.0f) {
                        // This is an uphill slope from the upwind direction
                        float uphillSlope = elevationDiff * windAlignment;
                        maxUphillSlope = std::max(maxUphillSlope, uphillSlope);
                    }
                }
            }
            
            // Apply orographic lifting effect (more precipitation on windward slopes)
            if (maxUphillSlope > 0.0f) {
                // Increase precipitation based on slope and elevation
                float liftingFactor = 1.0f + (maxUphillSlope * 2.0f * (elevation / 5.0f));
                precipitation[i] *= liftingFactor;
            }
        }
    }
    
    /**
     * @brief Determine climate type based on temperature and precipitation
     * 
     * @param vertexIdx Vertex index
     * @return ClimateType Classification
     */
    ClimateType determineClimateType(int vertexIdx) const {
        float temp = temperature[vertexIdx];
        float precip = precipitation[vertexIdx];
        float elevation = planet.getElevation(vertexIdx);
        
        // Ocean areas
        if (elevation < 0.0f) {
            return ClimateType::OCEANIC;
        }
        
        // High mountains
        if (elevation > 3.0f) {
            return ClimateType::MOUNTAIN;
        }
        
        // Polar regions
        if (temp < 0.0f) {
            return ClimateType::POLAR;
        }
        
        // Desert (hot or cold)
        if (precip < 500.0f) {
            return ClimateType::DESERT;
        }
        
        // Tropical
        if (temp > 20.0f) {
            return ClimateType::TROPICAL;
        }
        
        // Continental (wide temperature range)
        const SphericalGrid& grid = planet.getGrid();
        Vec3f pos = grid.getVertex(vertexIdx);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        if (std::abs(lat) > 30.0f && std::abs(lat) < 60.0f && precip < 1000.0f) {
            return ClimateType::CONTINENTAL;
        }
        
        // Default: temperate
        return ClimateType::TEMPERATE;
    }
    
    /**
     * @brief Calculate weathering rates based on climate
     * Different types of weathering occur in different climates
     */
    void calculateWeatheringRates() {
        size_t vertexCount = planet.getVertexCount();
        
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Only calculate for land areas
            if (elevation <= 0.0f) {
                weatheringRate[i] = 0.1f; // Minimal weathering underwater
                continue;
            }
            
            // Base factors
            float tempFactor = 0.0f;
            float precipFactor = 0.0f;
            
            // Temperature effect on weathering
            if (temperature[i] < 0.0f) {
                // Mechanical weathering dominates in cold climates
                tempFactor = 0.5f - temperature[i] * 0.05f; // Stronger at extreme cold
            } else {
                // Chemical weathering dominates in warm climates
                tempFactor = temperature[i] / 30.0f;
            }
            
            // Precipitation effect on weathering
            precipFactor = precipitation[i] / 2000.0f;
            
            // Combined effect
            weatheringRate[i] = tempFactor * 0.5f + precipFactor * 0.5f;
            
            // Adjust for elevation (more mechanical weathering at high elevations)
            if (elevation > 1.5f) {
                weatheringRate[i] *= (1.0f + (elevation - 1.5f) * 0.2f);
            }
            
            // Clamp to reasonable range
            weatheringRate[i] = std::max(0.1f, std::min(2.0f, weatheringRate[i]));
        }
    }
    
    /**
     * @brief Apply erosion based on climate factors
     * 
     * @param timeStep Time step in million years
     */
    void applyClimateBasedErosion(double timeStep) {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        // Temporary array to store elevation changes
        std::vector<float> elevationChanges(vertexCount, 0.0f);
        
        // Calculate erosion for each vertex
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Skip ocean areas
            if (elevation <= 0.0f) {
                continue;
            }
            
            // Base erosion rate depends on weathering rate
            float erosionRate = weatheringRate[i] * params.erosionRate;
            
            // Apply climate-specific erosion factors
            switch (climateTypes[i]) {
                case ClimateType::POLAR:
                    // Glacial erosion in polar regions
                    erosionRate *= 1.2f;
                    break;
                case ClimateType::TROPICAL:
                    // High chemical weathering in tropical regions
                    erosionRate *= 1.5f;
                    break;
                case ClimateType::DESERT:
                    // Wind erosion dominates in deserts
                    erosionRate *= 0.7f;
                    break;
                case ClimateType::MOUNTAIN:
                    // Enhanced mechanical weathering in mountains
                    erosionRate *= 1.3f;
                    break;
                default:
                    break;
            }
            
            // Calculate slope
            const auto& neighbors = grid.getNeighbors(vertexIdx);
            float maxSlope = 0.0f;
            
            for (int neighborIdx : neighbors) {
                float neighborElevation = planet.getElevation(neighborIdx);
                float slope = elevation - neighborElevation;
                maxSlope = std::max(maxSlope, slope);
            }
            
            // Erosion increases with slope
            float slopeFactor = 1.0f + maxSlope * 2.0f;
            
            // Final erosion amount
            float erosionAmount = erosionRate * slopeFactor * static_cast<float>(timeStep);
            
            // Apply erosion
            elevationChanges[i] -= erosionAmount;
            
            // Distribute eroded material to lowest neighbor
            if (maxSlope > 0.0f) {
                int lowestNeighborIdx = -1;
                float lowestElevation = std::numeric_limits<float>::max();
                
                for (int neighborIdx : neighbors) {
                    float neighborElevation = planet.getElevation(neighborIdx);
                    if (neighborElevation < lowestElevation) {
                        lowestElevation = neighborElevation;
                        lowestNeighborIdx = neighborIdx;
                    }
                }
                
                if (lowestNeighborIdx >= 0) {
                    // Deposit some eroded material downhill
                    elevationChanges[lowestNeighborIdx] += erosionAmount * 0.3f;
                }
            }
        }
        
        // Apply all elevation changes
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float newElevation = planet.getElevation(vertexIdx) + elevationChanges[i];
            planet.setElevation(vertexIdx, newElevation);
        }
    }
    
    /**
     * @brief Apply climate-specific sedimentation patterns
     * 
     * @param timeStep Time step in million years
     */
    void applyClimateSedimentation(double timeStep) {
        size_t vertexCount = planet.getVertexCount();
        const SphericalGrid& grid = planet.getGrid();
        
        // Temporary array to store elevation changes
        std::vector<float> elevationChanges(vertexCount, 0.0f);
        
        // Calculate sediment deposition for each vertex
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float elevation = planet.getElevation(vertexIdx);
            
            // Different sedimentation patterns based on environment
            
            // Deltas and coastlines
            if (elevation > -0.2f && elevation < 0.2f) {
                const auto& neighbors = grid.getNeighbors(vertexIdx);
                bool hasLand = false;
                bool hasOcean = false;
                
                for (int neighborIdx : neighbors) {
                    float neighborElevation = planet.getElevation(neighborIdx);
                    if (neighborElevation > 0.2f) hasLand = true;
                    if (neighborElevation < -0.2f) hasOcean = true;
                }
                
                // Coastal areas accumulate sediment
                if (hasLand && hasOcean) {
                    float sedimentRate = params.sedimentationRate * 1.5f;
                    elevationChanges[i] += sedimentRate * static_cast<float>(timeStep);
                }
            }
            
            // Lakes and flat areas accumulate sediment
            if (elevation > 0.0f) {
                const auto& neighbors = grid.getNeighbors(vertexIdx);
                bool isDepression = true;
                
                for (int neighborIdx : neighbors) {
                    if (planet.getElevation(neighborIdx) < elevation) {
                        isDepression = false;
                        break;
                    }
                }
                
                if (isDepression) {
                    // Fill depressions (lakes) with sediment
                    float sedimentRate = params.sedimentationRate * 2.0f;
                    elevationChanges[i] += sedimentRate * static_cast<float>(timeStep);
                }
            }
            
            // Underwater sedimentation
            if (elevation < 0.0f) {
                // Sediment accumulation decreases with depth
                float depthFactor = std::max(0.1f, 1.0f + elevation * 0.5f);
                float sedimentRate = params.sedimentationRate * 0.5f * depthFactor;
                elevationChanges[i] += sedimentRate * static_cast<float>(timeStep);
            }
        }
        
        // Apply all elevation changes
        for (size_t i = 0; i < vertexCount; ++i) {
            int vertexIdx = static_cast<int>(i);
            float newElevation = planet.getElevation(vertexIdx) + elevationChanges[i];
            planet.setElevation(vertexIdx, newElevation);
        }
    }
    
    /**
     * @brief Print some statistics about the current climate
     */
    void printClimateStatistics() {
        size_t vertexCount = planet.getVertexCount();
        
        // Count climate types
        std::map<ClimateType, int> climateCounts;
        for (size_t i = 0; i < vertexCount; ++i) {
            climateCounts[climateTypes[i]]++;
        }
        
        // Find temperature and precipitation ranges
        float minTemp = std::numeric_limits<float>::max();
        float maxTemp = -std::numeric_limits<float>::max();
        float avgTemp = 0.0f;
        float minPrecip = std::numeric_limits<float>::max();
        float maxPrecip = -std::numeric_limits<float>::max();
        float avgPrecip = 0.0f;
        
        for (size_t i = 0; i < vertexCount; ++i) {
            minTemp = std::min(minTemp, temperature[i]);
            maxTemp = std::max(maxTemp, temperature[i]);
            avgTemp += temperature[i];
            
            minPrecip = std::min(minPrecip, precipitation[i]);
            maxPrecip = std::max(maxPrecip, precipitation[i]);
            avgPrecip += precipitation[i];
        }
        
        avgTemp /= vertexCount;
        avgPrecip /= vertexCount;
        
        // Print statistics
        std::cout << "Climate Statistics:" << std::endl;
        std::cout << "  Temperature range: " << minTemp << "°C to " << maxTemp 
                  << "°C (avg: " << avgTemp << "°C)" << std::endl;
        std::cout << "  Precipitation range: " << minPrecip << "mm to " << maxPrecip 
                  << "mm (avg: " << avgPrecip << "mm)" << std::endl;
        std::cout << "  Climate types:" << std::endl;
        
        for (const auto& [type, count] : climateCounts) {
            float percentage = 100.0f * static_cast<float>(count) / vertexCount;
            std::cout << "    " << getClimateTypeName(type) << ": " 
                      << count << " vertices (" << percentage << "%)" << std::endl;
        }
    }
};

/**
 * @brief Simulation parameters for climate-geology feedback
 */
struct SimulationParameters {
    float planetRadius = 6371.0f;         // Earth radius in km
    float tectonicActivityScale = 0.5f;    // 0-1 scale
    float erosionRate = 0.05f;             // km/Myr
    float sedimentationRate = 0.02f;       // km/Myr
    float globalTemperature = 15.0f;       // Global average in °C
    float oceanCoverage = 0.7f;            // Fraction of surface covered by oceans
};

// Example usage function
void runClimateGeologyExample() {
    // Create a planet with resolution level 6 (40,962 vertices)
    Planet planet(6371.0f, 6, "Earth");
    
    // Initialize the planet with some basic terrain
    initializeSimpleTerrain(planet);
    
    // Create simulation parameters
    SimulationParameters params;
    
    // Create climate-geology feedback system
    ClimateGeologyFeedback feedback(planet, params);
    
    // Run simulation for 100 million years with 10-million-year steps
    double timeStep = 10.0;
    int steps = 10;
    
    std::cout << "Starting climate-geology feedback simulation over " 
              << (timeStep * steps) << " million years..." << std::endl;
    
    for (int i = 0; i < steps; ++i) {
        // Run one feedback cycle
        feedback.runFeedbackCycle(timeStep);
        
        // Report current age
        std::cout << "Simulation age: " << ((i+1) * timeStep) << " million years" << std::endl;
        
        // Every 50 million years, print detailed statistics
        if ((i+1) % 5 == 0) {
            reportPlanetStatistics(planet);
        }
    }
    
    std::cout << "Climate-geology feedback simulation complete!" << std::endl;
}

/**
 * @brief Initialize a simple terrain with continents and oceans
 * 
 * @param planet Planet to initialize
 */
void initializeSimpleTerrain(Planet& planet) {
    size_t vertexCount = planet.getVertexCount();
    const SphericalGrid& grid = planet.getGrid();
    
    // Create a simple continent-ocean pattern
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        Vec3f pos = grid.getVertex(vertexIdx);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Simple continent pattern - higher in northern hemisphere
        float elevation = 0.0f;
        
        // Supercontinent in northern hemisphere
        if (lat > 0 && std::abs(lon) < 120.0f) {
            elevation = 0.5f + 1.5f * std::exp(-std::pow(lat - 30.0f, 2) / 1000.0f 
                                            - std::pow(lon, 2) / 5000.0f);
        }
        // Island chain
        else if (lat < 0 && lon > 30.0f && lon < 150.0f) {
            elevation = 0.5f * std::exp(-std::pow(lat + 20.0f, 2) / 500.0f 
                                    - std::pow(lon - 90.0f, 2) / 2000.0f);
        }
        // Another continent
        else if (lat < -20.0f && lon < -30.0f) {
            elevation = 1.0f * std::exp(-std::pow(lat + 50.0f, 2) / 1000.0f 
                                    - std::pow(lon + 90.0f, 2) / 4000.0f);
        }
        else {
            // Ocean
            elevation = -3.0f + 0.5f * std::sin(lat * 0.05f) * std::cos(lon * 0.03f);
        }
        
        // Set elevation
        planet.setElevation(vertexIdx, elevation);
        
        // Set crust type based on elevation
        if (elevation > 0.0f) {
            planet.setCrustType(vertexIdx, 1);  // Continental
            planet.setCrustThickness(vertexIdx, 35.0f);
            planet.setCrustDensity(vertexIdx, 2.7f);
        } else {
            planet.setCrustType(vertexIdx, 0);  // Oceanic
            planet.setCrustThickness(vertexIdx, 7.0f);
            planet.setCrustDensity(vertexIdx, 3.0f);
        }
    }
    
    std::cout << "Initialized planet with simple continent-ocean pattern" << std::endl;
}

/**
 * @brief Report statistics about the planet
 * 
 * @param planet Planet to analyze
 */
void reportPlanetStatistics(const Planet& planet) {
    size_t vertexCount = planet.getVertexCount();
    
    // Calculate elevation statistics
    float minElevation = std::numeric_limits<float>::max();
    float maxElevation = -std::numeric_limits<float>::max();
    float avgElevation = 0.0f;
    int landCount = 0;
    int oceanCount = 0;
    
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        
        minElevation = std::min(minElevation, elevation);
        maxElevation = std::max(maxElevation, elevation);
        avgElevation += elevation;
        
        if (elevation > 0.0f) {
            landCount++;
        } else {
            oceanCount++;
        }
    }
    
    avgElevation /= vertexCount;
    float landPercentage = 100.0f * static_cast<float>(landCount) / vertexCount;
    
    std::cout << "Planet Statistics:" << std::endl;
    std::cout << "  Elevation range: " << minElevation << "km to " << maxElevation 
              << "km (avg: " << avgElevation << "km)" << std::endl;
    std::cout << "  Land coverage: " << landCount << " vertices (" 
              << landPercentage << "%)" << std::endl;
    std::cout << "  Ocean coverage: " << oceanCount << " vertices (" 
              << (100.0f - landPercentage) << "%)" << std::endl;
}

} // namespace aeonterracpp

// Main function for standalone testing
int main() {
    aeonterracpp::runClimateGeologyExample();
    return 0;
}