#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/vector_types.h"
#include <vector>
#include <memory>
#include <random>

namespace aeonterracpp {

/**
 * @brief Parameters for global climate system
 */
struct ClimateParams {
    // General parameters
    float solarConstant = 1361.0f;         // Solar constant in W/m²
    float planetaryAlbedo = 0.3f;          // Average planetary albedo
    float atmosphericComposition = 1.0f;    // 1.0 = Earth-like (affects greenhouse)
    float axialTilt = 23.5f;               // Axial tilt in degrees
    float dayLength = 24.0f;               // Length of day in hours
    float yearLength = 365.24f;            // Length of year in days
    float eccentricity = 0.0167f;          // Orbital eccentricity
    
    // Atmospheric parameters
    float hadleyCellWidth = 30.0f;         // Width of Hadley cells in degrees latitude
    float ferrelCellWidth = 30.0f;         // Width of Ferrel cells in degrees latitude
    float polarCellWidth = 30.0f;          // Width of Polar cells in degrees latitude
    float jetStreamLatitude = 60.0f;       // Latitude of jet streams
    float tradeWindSpeed = 15.0f;          // Trade wind speed in m/s
    float westerliesSpeed = 20.0f;         // Westerlies speed in m/s
    float polarEasterliesSpeed = 10.0f;    // Polar easterlies speed in m/s
    
    // Oceanic parameters
    float oceanHeatCapacity = 4.2e6f;      // Heat capacity of ocean in J/m³·K
    float gyreStrength = 1.0f;             // Strength of ocean gyres (1.0 = Earth-like)
    float deepOceanCirculation = 1.0f;     // Strength of thermohaline circulation
    float upwellingStrength = 1.0f;        // Strength of upwelling zones
    
    // Precipitation parameters
    float globalPrecipitation = 1000.0f;   // Global average precipitation in mm/year
    float orographicEffect = 1.0f;         // Strength of orographic precipitation
    float convectionalEffect = 1.0f;       // Strength of convectional precipitation
    float frontalEffect = 1.0f;            // Strength of frontal precipitation
    
    // Seasonal parameters
    float seasonalVariation = 1.0f;        // Strength of seasonal temperature variation
    float monsoonStrength = 1.0f;          // Strength of monsoon systems
    
    // Constructor with default Earth-like parameters
    ClimateParams() {}
};

/**
 * @brief Atmospheric circulation system
 */
class AtmosphericCirculation {
public:
    /**
     * @brief Construct a new Atmospheric Circulation
     * 
     * @param planet Reference to the planet
     * @param params Climate parameters
     */
    AtmosphericCirculation(const Planet& planet, const ClimateParams& params);
    
    /**
     * @brief Initialize atmospheric circulation patterns
     */
    void initialize();
    
    /**
     * @brief Simulate atmospheric circulation for a time step
     * 
     * @param timeStep Time increment in million years
     * @param dayOfYear Current day of year (0-364)
     */
    void simulate(double timeStep, float dayOfYear = 182.0f);
    
    /**
     * @brief Get wind vector at a specific position
     * 
     * @param position Position on the planet surface
     * @return Vec3f Wind vector (direction + magnitude)
     */
    Vec3f getWindVector(const Vec3f& position) const;
    
    /**
     * @brief Get pressure at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Atmospheric pressure
     */
    float getPressure(const Vec3f& position) const;
    
    /**
     * @brief Get temperature at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Temperature in Celsius
     */
    float getTemperature(const Vec3f& position) const;
    
    /**
     * @brief Calculate pressure systems
     * Creates high and low pressure regions based on circulation cells
     */
    void calculatePressureSystems();
    
    /**
     * @brief Calculate global wind patterns
     * Implements trade winds, westerlies, and polar easterlies
     */
    void calculateWindPatterns();
    
    /**
     * @brief Calculate seasonal variations in circulation
     * 
     * @param dayOfYear Current day of year (0-364)
     */
    void calculateSeasonalShift(float dayOfYear);
    
    /**
     * @brief Calculate orographic effects on wind patterns
     * Implements lifting, blocking, and channeling by topography
     */
    void calculateOrographicEffects();
    
    /**
     * @brief Calculate jet streams
     * Implements polar and subtropical jet streams
     */
    void calculateJetStreams();
    
    /**
     * @brief Access wind data
     * 
     * @return const std::vector<Vec3f>& Vector of wind vectors at each vertex
     */
    const std::vector<Vec3f>& getWindData() const;
    
    /**
     * @brief Access pressure data
     * 
     * @return const std::vector<float>& Pressure values at each vertex
     */
    const std::vector<float>& getPressureData() const;
    
    /**
     * @brief Access temperature data
     * 
     * @return const std::vector<float>& Temperature values at each vertex
     */
    const std::vector<float>& getTemperatureData() const;
    
private:
    const Planet& planet;
    const ClimateParams& params;
    
    // Atmospheric data
    std::vector<Vec3f> windVectors;   // Wind velocity at each vertex
    std::vector<float> pressure;      // Atmospheric pressure at each vertex
    std::vector<float> temperature;   // Temperature at each vertex
    
    // Circulation cells
    void simulateHadleyCells();       // Tropical circulation (~0-30° latitude)
    void simulateFerrelCells();       // Mid-latitude circulation (~30-60°)
    void simulatePolarCells();        // Polar circulation (~60-90°)
    
    // Calculate base temperature distribution
    void calculateBaseTemperature();
    
    // Calculate Coriolis effect
    float calculateCoriolisParameter(float latitude) const;
};

/**
 * @brief Oceanic circulation system
 */
class OceanicCirculation {
public:
    /**
     * @brief Construct a new Oceanic Circulation
     * 
     * @param planet Reference to the planet
     * @param atmosphere Reference to the atmospheric circulation
     * @param params Climate parameters
     */
    OceanicCirculation(const Planet& planet, 
                       const AtmosphericCirculation& atmosphere,
                       const ClimateParams& params);
    
    /**
     * @brief Initialize oceanic circulation patterns
     */
    void initialize();
    
    /**
     * @brief Simulate oceanic circulation for a time step
     * 
     * @param timeStep Time increment in million years
     * @param dayOfYear Current day of year (0-364)
     */
    void simulate(double timeStep, float dayOfYear = 182.0f);
    
    /**
     * @brief Get current vector at a specific position
     * 
     * @param position Position on the planet surface
     * @return Vec3f Current vector (direction + magnitude)
     */
    Vec3f getCurrentVector(const Vec3f& position) const;
    
    /**
     * @brief Get sea surface temperature at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Sea surface temperature in Celsius
     */
    float getSeaSurfaceTemperature(const Vec3f& position) const;
    
    /**
     * @brief Calculate major ocean gyres
     * Implements subtropical and subpolar gyres
     */
    void simulateGyres();
    
    /**
     * @brief Calculate equatorial currents
     * Implements equatorial currents and countercurrents
     */
    void simulateEquatorialCurrents();
    
    /**
     * @brief Calculate deep ocean circulation
     * Implements thermohaline circulation (global conveyor belt)
     */
    void simulateThermohalineCirculation();
    
    /**
     * @brief Calculate coastal currents
     * Implements currents shaped by coastlines
     */
    void calculateCoastalCurrents();
    
    /**
     * @brief Calculate upwelling and downwelling zones
     * Identifies areas of vertical water movement
     */
    void calculateUpwellingRegions();
    
    /**
     * @brief Calculate heat transport by ocean currents
     */
    void calculateHeatTransport();
    
    /**
     * @brief Access current data
     * 
     * @return const std::vector<Vec3f>& Vector of current vectors at each vertex
     */
    const std::vector<Vec3f>& getCurrentData() const;
    
    /**
     * @brief Access sea surface temperature data
     * 
     * @return const std::vector<float>& Sea surface temperature values at each vertex
     */
    const std::vector<float>& getSeaSurfaceTemperatureData() const;
    
private:
    const Planet& planet;
    const AtmosphericCirculation& atmosphere;
    const ClimateParams& params;
    
    // Oceanic data
    std::vector<Vec3f> currentVectors;   // Current velocity at each vertex
    std::vector<float> seaSurfaceTemp;   // Sea surface temperature at each vertex
    std::vector<float> salinity;         // Salinity at each vertex
    std::vector<float> upwelling;        // Upwelling rate at each vertex
    
    // Helper methods
    bool isOcean(int vertexIdx) const;
    float getOceanDepth(int vertexIdx) const;
    float calculateWindStress(const Vec3f& windVector) const;
};

/**
 * @brief Enhanced hydrological system
 */
class HydrologicalSystem {
public:
    /**
     * @brief Construct a new Hydrological System
     * 
     * @param planet Reference to the planet
     * @param atmosphere Reference to the atmospheric circulation
     * @param ocean Reference to the oceanic circulation
     * @param params Climate parameters
     */
    HydrologicalSystem(const Planet& planet,
                       const AtmosphericCirculation& atmosphere,
                       const OceanicCirculation& ocean,
                       const ClimateParams& params);
    
    /**
     * @brief Initialize hydrological system
     */
    void initialize();
    
    /**
     * @brief Simulate hydrological cycle for a time step
     * 
     * @param timeStep Time increment in million years
     * @param dayOfYear Current day of year (0-364)
     */
    void simulate(double timeStep, float dayOfYear = 182.0f);
    
    /**
     * @brief Get precipitation at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Precipitation in mm/year
     */
    float getPrecipitation(const Vec3f& position) const;
    
    /**
     * @brief Get evaporation rate at a specific position
     * 
     * @param position Position on the planet surface
     * @return float Evaporation rate in mm/year
     */
    float getEvaporation(const Vec3f& position) const;
    
    /**
     * @brief Get river flow at a specific position
     * 
     * @param position Position on the planet surface
     * @return float River flow in m³/s
     */
    float getRiverFlow(const Vec3f& position) const;
    
    /**
     * @brief Calculate evaporation
     * Models water vapor entering the atmosphere
     */
    void calculateEvaporation();
    
    /**
     * @brief Calculate moisture transport
     * Models movement of water vapor by winds
     */
    void calculateMoistureTransport();
    
    /**
     * @brief Calculate convectional precipitation
     * Models precipitation from rising air masses
     */
    void calculateConvectionalPrecipitation();
    
    /**
     * @brief Calculate orographic precipitation
     * Models precipitation from terrain-forced lifting
     */
    void calculateOrographicPrecipitation();
    
    /**
     * @brief Calculate frontal precipitation
     * Models precipitation from weather fronts
     */
    void calculateFrontalPrecipitation();
    
    /**
     * @brief Calculate detailed river networks
     * Models river formation based on terrain and rainfall
     */
    void simulateDetailedRiverNetworks();
    
    /**
     * @brief Calculate lake formation
     * Models formation of lakes in basins
     */
    void calculateLakeFormation();
    
    /**
     * @brief Calculate delta formation
     * Models formation of deltas at river mouths
     */
    void calculateDeltaFormation();
    
    /**
     * @brief Access precipitation data
     * 
     * @return const std::vector<float>& Precipitation values at each vertex
     */
    const std::vector<float>& getPrecipitationData() const;
    
    /**
     * @brief Access evaporation data
     * 
     * @return const std::vector<float>& Evaporation values at each vertex
     */
    const std::vector<float>& getEvaporationData() const;
    
    /**
     * @brief Access river flow data
     * 
     * @return const std::vector<float>& River flow values at each vertex
     */
    const std::vector<float>& getRiverFlowData() const;
    
    /**
     * @brief Check if a vertex is a river
     * 
     * @return const std::vector<bool>& Boolean array indicating river presence
     */
    const std::vector<bool>& getRiverData() const;
    
    /**
     * @brief Check if a vertex is a lake
     * 
     * @return const std::vector<bool>& Boolean array indicating lake presence
     */
    const std::vector<bool>& getLakeData() const;
    
private:
    const Planet& planet;
    const AtmosphericCirculation& atmosphere;
    const OceanicCirculation& ocean;
    const ClimateParams& params;
    
    // Hydrological data
    std::vector<float> precipitation;   // Precipitation at each vertex (mm/year)
    std::vector<float> evaporation;     // Evaporation at each vertex (mm/year)
    std::vector<float> soilMoisture;    // Soil moisture at each vertex (%)
    std::vector<float> riverFlow;       // River flow at each vertex (m³/s)
    std::vector<int> flowDirection;     // Flow direction for each vertex
    std::vector<float> drainage;        // Drainage area for each vertex (km²)
    std::vector<bool> isRiver;          // Whether a vertex is part of a river
    std::vector<bool> isLake;           // Whether a vertex is a lake
    
    // Moisture transport
    std::vector<float> atmosphericMoisture; // Moisture content in atmosphere
    
    // Helper methods
    float calculateRunoff(float precipitation, float evaporation) const;
    void accumulateFlowDownstream();
};

/**
 * @brief Complete climate system integrating atmosphere, ocean, and hydrology
 */
class ClimateSystem {
public:
    /**
     * @brief Construct a new Climate System
     * 
     * @param planet Reference to the planet
     */
    ClimateSystem(Planet& planet);
    
    /**
     * @brief Set climate parameters
     * 
     * @param params The parameters to use
     */
    void setParameters(const ClimateParams& params);
    
    /**
     * @brief Get climate parameters
     * 
     * @return const ClimateParams& The current parameters
     */
    const ClimateParams& getParameters() const;
    
    /**
     * @brief Initialize climate system
     */
    void initialize();
    
    /**
     * @brief Simulate climate for a time step
     * 
     * @param timeStep Time increment in million years
     * @param dayOfYear Current day of year (0-364)
     */
    void simulate(double timeStep, float dayOfYear = 182.0f);
    
    /**
     * @brief Get atmospheric circulation system
     * 
     * @return const AtmosphericCirculation& Reference to atmosphere
     */
    const AtmosphericCirculation& getAtmosphere() const;
    
    /**
     * @brief Get oceanic circulation system
     * 
     * @return const OceanicCirculation& Reference to ocean
     */
    const OceanicCirculation& getOcean() const;
    
    /**
     * @brief Get hydrological system
     * 
     * @return const HydrologicalSystem& Reference to hydrology
     */
    const HydrologicalSystem& getHydrology() const;
    
    /**
     * @brief Calculate climate zones using Köppen classification
     */
    void calculateClimateZones();
    
    /**
     * @brief Simulate weathering and erosion based on climate
     * 
     * @param timeStep Time increment in million years
     */
    void simulateWeathering(double timeStep);
    
    /**
     * @brief Save climate data to file
     * 
     * @param filename Output filename
     * @return bool Success or failure
     */
    bool saveToFile(const std::string& filename) const;
    
    /**
     * @brief Load climate data from file
     * 
     * @param filename Input filename
     * @return bool Success or failure
     */
    bool loadFromFile(const std::string& filename);
    
private:
    Planet& planet;
    ClimateParams params;
    
    // Component systems
    std::unique_ptr<AtmosphericCirculation> atmosphere;
    std::unique_ptr<OceanicCirculation> ocean;
    std::unique_ptr<HydrologicalSystem> hydrology;
    
    // Climate zone data
    std::vector<int> climateZones;  // Köppen climate classification
    
    // Season tracking
    float currentDayOfYear;
    
    // Initialize climate zones
    void initializeClimateZones();
    
    // Random number generator for stochastic processes
    std::mt19937 rng;
};

} // namespace aeonterracpp