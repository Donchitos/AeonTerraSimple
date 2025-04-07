#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/climate_system.h"
#include "aeonterracpp/core/biome_system.h"
#include <vector>
#include <memory>
#include <random>

namespace aeonterracpp {

/**
 * @brief Enum for different erosion process types
 */
enum class ErosionProcessType {
    Fluvial,            // Water erosion by rivers and streams
    Coastal,            // Erosion by ocean waves and currents
    Glacial,            // Erosion by glaciers and ice sheets
    Aeolian,            // Wind erosion
    Chemical,           // Chemical weathering
    Thermal,            // Thermal weathering (freeze-thaw cycles)
    Gravity,            // Mass movements like landslides
    Biotic,             // Erosion related to biological activity
    Tectonic,           // Erosion due to tectonic activity
    
    Count               // Number of erosion process types
};

/**
 * @brief Parameters for erosion simulation
 */
struct ErosionParams {
    // Global erosion rate multiplier (1.0 = Earth-like)
    float globalErosionRate = 1.0f;
    
    // Process-specific multipliers (1.0 = normal strength)
    float fluvialErosionStrength = 1.0f;
    float coastalErosionStrength = 1.0f;
    float glacialErosionStrength = 1.0f;
    float aeolianErosionStrength = 1.0f;
    float chemicalWeatheringStrength = 1.0f;
    float thermalWeatheringStrength = 1.0f;
    float massMovementStrength = 1.0f;
    float bioticErosionStrength = 1.0f;
    
    // Process thresholds and parameters
    float riverFlowThreshold = 10.0f;      // Min flow for river erosion (m³/s)
    float coastalWaveEnergy = 1.0f;        // Wave energy for coastal erosion
    float glacialTemperatureThreshold = -2.0f; // Max temp for glacier formation (°C)
    float windErosionThreshold = 8.0f;     // Min wind speed for aeolian erosion (m/s)
    float criticalSlope = 0.6f;            // Slope threshold for mass movements
    float sedimentCapacityFactor = 1.0f;   // Factor for sediment carrying capacity
    
    // Timescale adjustment (1.0 = real time)
    float timeScaleFactor = 1.0f;
};

/**
 * @brief Enhanced Erosion System with multiple erosion processes
 * 
 * This class manages various erosion processes acting on a planet's surface,
 * taking into account climate, biomes, and physical properties to simulate
 * realistic landscape evolution.
 */
class EnhancedErosion {
public:
    /**
     * @brief Construct a new Enhanced Erosion System
     * 
     * @param planet The planet to erode
     * @param climate The climate system providing weather data
     * @param biomes The biome system providing surface properties
     */
    EnhancedErosion(Planet& planet, 
                  const ClimateSystem& climate,
                  const BiomeSystem& biomes);
    
    /**
     * @brief Initialize the erosion system
     * 
     * @param params Erosion parameters
     */
    void initialize(const ErosionParams& params = ErosionParams());
    
    /**
     * @brief Set erosion parameters
     * 
     * @param params New erosion parameters
     */
    void setParameters(const ErosionParams& params);
    
    /**
     * @brief Get current erosion parameters
     * 
     * @return const ErosionParams& Current parameters
     */
    const ErosionParams& getParameters() const;
    
    /**
     * @brief Simulate erosion for a time step
     * 
     * @param timeStep Time increment in million years
     */
    void simulateStep(double timeStep);
    
    /**
     * @brief Get total erosion rate at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Erosion rate in meters per million years
     */
    float getErosionRate(const Vec3f& position) const;
    
    /**
     * @brief Get erosion rate for a specific process at a position
     * 
     * @param position Position on the planet surface
     * @param processType Type of erosion process
     * @return float Process-specific erosion rate in meters per million years
     */
    float getProcessErosionRate(const Vec3f& position, ErosionProcessType processType) const;
    
    /**
     * @brief Get accumulated erosion amount at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Total erosion in meters
     */
    float getAccumulatedErosion(const Vec3f& position) const;
    
    /**
     * @brief Get accumulated deposition amount at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Total deposition in meters
     */
    float getAccumulatedDeposition(const Vec3f& position) const;
    
    /**
     * @brief Simulate fluvial (river) erosion
     * 
     * @param timeStep Time increment in million years
     */
    void simulateFluvialErosion(double timeStep);
    
    /**
     * @brief Simulate coastal erosion
     * 
     * @param timeStep Time increment in million years
     */
    void simulateCoastalErosion(double timeStep);
    
    /**
     * @brief Simulate glacial erosion
     * 
     * @param timeStep Time increment in million years
     */
    void simulateGlacialErosion(double timeStep);
    
    /**
     * @brief Simulate aeolian (wind) erosion
     * 
     * @param timeStep Time increment in million years
     */
    void simulateAeolianErosion(double timeStep);
    
    /**
     * @brief Simulate chemical weathering
     * 
     * @param timeStep Time increment in million years
     */
    void simulateChemicalWeathering(double timeStep);
    
    /**
     * @brief Simulate thermal weathering (freeze-thaw)
     * 
     * @param timeStep Time increment in million years
     */
    void simulateThermalWeathering(double timeStep);
    
    /**
     * @brief Simulate mass movements (landslides, slumps, etc.)
     * 
     * @param timeStep Time increment in million years
     */
    void simulateMassMovements(double timeStep);
    
    /**
     * @brief Simulate biotic erosion (vegetation, animals, etc.)
     * 
     * @param timeStep Time increment in million years
     */
    void simulateBioticErosion(double timeStep);
    
    /**
     * @brief Calculate amount of material moved during tectonic processes
     * 
     * @param timeStep Time increment in million years
     */
    void calculateTectonicErosion(double timeStep);
    
    /**
     * @brief Apply calculated erosion to the planet surface
     * This actually modifies the planet's elevation
     */
    void applyErosionToSurface();
    
    /**
     * @brief Get total erosion data
     * 
     * @return const std::vector<float>& Erosion rates at each vertex
     */
    const std::vector<float>& getErosionData() const;
    
    /**
     * @brief Get process-specific erosion data
     * 
     * @param processType Type of erosion process
     * @return const std::vector<float>& Process-specific erosion rates
     */
    const std::vector<float>& getProcessErosionData(ErosionProcessType processType) const;
    
    /**
     * @brief Get accumulated sediment data
     * 
     * @return const std::vector<float>& Accumulated sediment at each vertex
     */
    const std::vector<float>& getSedimentData() const;
    
    /**
     * @brief Calculate the effect of accumulated erosion on biomes
     * This provides feedback to the biome system based on erosion results
     * 
     * @param biomes The biome system to update
     */
    void updateBiomeEffects(BiomeSystem& biomes);
    
    /**
     * @brief Calculate impacts of erosion on river networks
     * This provides feedback to the hydrological system
     * 
     * @param climate The climate system to influence
     */
    void updateHydrologyEffects(ClimateSystem& climate);

private:
    Planet& planet;
    const ClimateSystem& climate;
    const BiomeSystem& biomes;
    ErosionParams params;
    
    // Grid reference
    const SphericalGrid& grid;
    
    // Vector of per-vertex erosion data for each process
    std::vector<std::vector<float>> processErosionRates;
    
    // Total erosion rate per vertex
    std::vector<float> totalErosionRate;
    
    // Accumulated erosion and deposition
    std::vector<float> accumulatedErosion;
    std::vector<float> accumulatedDeposition;
    
    // Temporary buffers for erosion calculations
    std::vector<float> erosionBuffer;
    std::vector<float> sedimentBuffer;
    std::vector<float> sedimentCapacity;
    
    // Vertex classification flags
    std::vector<bool> isGlaciated;
    std::vector<bool> isCoastal;
    
    // Flow information from hydrology
    std::vector<float> flowBuffer;              // River flow at each vertex
    std::vector<int> flowDirectionBuffer;       // Flow direction at each vertex
    std::vector<float> upstreamAreaBuffer;      // Upstream catchment area
    
    // Random number generator for stochastic processes
    std::mt19937 rng;
    
    // Initialize erosion buffers
    void initializeBuffers();
    
    // Update flow information from climate/hydrology
    void updateFlowInfo();
    
    // Classify vertices for different erosion processes
    void classifyVertices();
    
    // Calculate slopes for all vertices
    void calculateSlopes(std::vector<float>& slopes);
    
    // Calculate sediment capacity based on flow and slope
    void calculateSedimentCapacity();
    
    // Calculate erosion based on excess shear stress
    float calculateExcessShearStress(float flow, float slope, float criticalShear) const;
    
    // Apply a minimum erosion amount regardless of other factors
    // This represents background processes not explicitly modeled
    void applyBackgroundErosion(double timeStep);
    
    // Transfer sediment downstream following flow directions
    void transportSediment();
};

} // namespace aeonterracpp