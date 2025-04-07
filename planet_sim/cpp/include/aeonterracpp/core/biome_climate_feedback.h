#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/climate_system.h"
#include "aeonterracpp/core/biome_system.h"
#include <vector>
#include <string>

namespace aeonterracpp {

/**
 * @brief Parameters for biome-climate feedback
 */
struct BiomeClimateFeedbackParams {
    // Feedback strength parameters (1.0 = Earth-like)
    float albedoFeedbackStrength = 1.0f;         // Vegetation effect on surface reflectivity
    float evapotranspirationStrength = 1.0f;     // Vegetation effect on water cycle
    float roughnessLengthEffect = 1.0f;          // Vegetation effect on air flow
    float vegetationGrowthRate = 1.0f;           // Rate of vegetation growth under favorable conditions
    float vegetationDiebackRate = 1.0f;          // Rate of vegetation decline under stress
    float carbonSequestrationRate = 1.0f;        // Carbon uptake by vegetation
    float localWarmingEffect = 1.0f;             // Local climate modification by vegetation
};

/**
 * @brief Manages the bidirectional feedback between biomes and climate
 * 
 * This class implements the various ways that vegetation and biomes
 * influence climate systems, and vice versa. Key mechanisms include:
 * - Albedo modification by vegetation (darker surfaces = more heat absorption)
 * - Evapotranspiration affecting atmospheric moisture
 * - Surface roughness affecting wind patterns
 * - Carbon sequestration by different plant communities
 * - Vegetation responses to changing climate conditions
 */
class BiomeClimateFeedback {
public:
    /**
     * @brief Construct a new Biome Climate Feedback system
     * 
     * @param planet Reference to the planet
     * @param climate Reference to the climate system
     * @param biomes Reference to the biome system
     */
    BiomeClimateFeedback(Planet& planet, 
                       ClimateSystem& climate, 
                       BiomeSystem& biomes);
    
    /**
     * @brief Initialize the feedback system
     */
    void initialize();
    
    /**
     * @brief Set feedback parameters
     * 
     * @param params New feedback parameters
     */
    void setParameters(const BiomeClimateFeedbackParams& params);
    
    /**
     * @brief Get current feedback parameters
     * 
     * @return const BiomeClimateFeedbackParams& Current parameters
     */
    const BiomeClimateFeedbackParams& getParameters() const;
    
    /**
     * @brief Update the feedback system for a time step
     * 
     * @param timeStep Time increment in million years
     * @param dayOfYear Current day of year (0-364)
     */
    void update(double timeStep, float dayOfYear = 182.0f);
    
    /**
     * @brief Calculate how biomes modify surface albedo
     */
    void calculateAlbedoEffects();
    
    /**
     * @brief Calculate evapotranspiration rates for different biomes
     */
    void calculateEvapotranspirationEffects();
    
    /**
     * @brief Calculate surface roughness based on vegetation
     */
    void calculateRoughnessEffects();
    
    /**
     * @brief Calculate carbon sequestration by vegetation
     * 
     * @param timeStep Time increment in million years
     */
    void calculateCarbonSequestration(double timeStep);
    
    /**
     * @brief Apply calculated feedback effects to the climate system
     */
    void applyFeedbackToClimate();
    
    /**
     * @brief Track changes in biomes due to climate conditions
     * 
     * @param timeStep Time increment in million years
     */
    void trackBiomeChanges(double timeStep);
    
    /**
     * @brief Record current state for history tracking
     */
    void recordCurrentState();
    
    /**
     * @brief Get albedo modifier at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Albedo modifier value
     */
    float getAlbedoModifier(const Vec3f& position) const;
    
    /**
     * @brief Get evapotranspiration rate at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Evapotranspiration rate in mm/year
     */
    float getEvapotranspirationRate(const Vec3f& position) const;
    
    /**
     * @brief Get surface roughness length at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Roughness length in meters
     */
    float getRoughnessLength(const Vec3f& position) const;
    
    /**
     * @brief Get sequestered carbon at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Carbon sequestered in tons
     */
    float getSequesteredCarbon(const Vec3f& position) const;
    
    /**
     * @brief Get history of average albedo values
     * 
     * @return const std::vector<float>& History of albedo values
     */
    const std::vector<float>& getAlbedoHistory() const;
    
    /**
     * @brief Get history of average temperature values
     * 
     * @return const std::vector<float>& History of temperature values
     */
    const std::vector<float>& getTemperatureHistory() const;
    
    /**
     * @brief Get history of average vegetation values
     * 
     * @return const std::vector<float>& History of vegetation values
     */
    const std::vector<float>& getVegetationHistory() const;
    
private:
    Planet& planet;
    ClimateSystem& climate;
    BiomeSystem& biomes;
    BiomeClimateFeedbackParams params;
    
    // Feedback effect buffers
    std::vector<float> albedoModifier;         // How biomes modify surface albedo
    std::vector<float> evapotranspirationRate; // Water vapor added to atmosphere by vegetation
    std::vector<float> roughnessLength;        // Surface roughness affecting wind patterns
    std::vector<float> sequesteredCarbon;      // Accumulated carbon sequestered
    
    // History tracking for long-term trends
    std::vector<float> albedoHistory;
    std::vector<float> temperatureHistory;
    std::vector<float> vegetationHistory;
};

} // namespace aeonterracpp