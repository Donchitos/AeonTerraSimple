#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/climate_system.h"
#include "aeonterracpp/core/enhanced_erosion.h"
#include <vector>
#include <array>
#include <string>

namespace aeonterracpp {

/**
 * @brief Chemical element/compound tracked in river-ocean system
 */
enum class ChemicalElement {
    Sediment,       // Suspended sediment particles
    Nitrogen,       // Nitrogen compounds (NO3, NH4, etc.)
    Phosphorus,     // Phosphorus compounds (phosphates)
    Carbon,         // Dissolved organic/inorganic carbon
    Oxygen,         // Dissolved oxygen
    Calcium,        // Calcium (Ca2+)
    Silica,         // Silica (SiO2)
    Iron,           // Iron (Fe2+, Fe3+)
    
    Count           // Number of chemical elements tracked
};

/**
 * @brief Parameters for river-ocean chemistry
 */
struct RiverOceanChemistryParams {
    // River parameters
    float riverErosionSedimentFactor = 1.0f;   // Conversion from erosion to sediment
    float riverNutrientLeachingRate = 1.0f;    // Rate of nutrient leaching from soils
    float riverCarbonExportRate = 1.0f;        // Rate of carbon export from land
    
    // Ocean parameters
    float oceanCirculationMixingRate = 1.0f;   // Mixing rate for chemicals in ocean
    float oceanSedimentationRate = 1.0f;       // Rate of sediment settling
    float oceanPrimaryProductionRate = 1.0f;   // Biological productivity in oceans
    float oceanOxygenationRate = 1.0f;         // Rate of ocean oxygenation
    
    // Chemistry parameters
    float carbonatePrecipitationRate = 1.0f;   // Rate of carbonate precipitation
    float silicaPrecipitationRate = 1.0f;      // Rate of silica precipitation
    float ironOxidationRate = 1.0f;            // Rate of iron oxidation
    
    // Time scaling
    float timeScaleFactor = 1.0f;              // Time acceleration factor
};

/**
 * @brief Models the chemical interaction between rivers and oceans
 * 
 * This class tracks how chemicals and sediments move from land to oceans via
 * rivers, and models the resulting ocean chemistry and feedback effects on
 * climate and coastal ecosystems.
 */
class RiverOceanChemistry {
public:
    /**
     * @brief Construct a new River Ocean Chemistry system
     * 
     * @param planet Reference to the planet
     * @param climate Reference to the climate system
     * @param erosion Reference to the erosion system
     */
    RiverOceanChemistry(Planet& planet,
                       ClimateSystem& climate,
                       const EnhancedErosion& erosion);
    
    /**
     * @brief Initialize the chemistry system
     * 
     * @param params Chemistry parameters
     */
    void initialize(const RiverOceanChemistryParams& params = RiverOceanChemistryParams());
    
    /**
     * @brief Set chemistry parameters
     * 
     * @param params New chemistry parameters
     */
    void setParameters(const RiverOceanChemistryParams& params);
    
    /**
     * @brief Get current chemistry parameters
     * 
     * @return const RiverOceanChemistryParams& Current parameters
     */
    const RiverOceanChemistryParams& getParameters() const;
    
    /**
     * @brief Update the chemistry system for a time step
     * 
     * @param timeStep Time increment in million years
     */
    void update(double timeStep);
    
    /**
     * @brief Calculate river input to oceans
     */
    void calculateRiverInput();
    
    /**
     * @brief Calculate ocean circulation and mixing
     * 
     * @param timeStep Time increment in million years
     */
    void calculateOceanMixing(double timeStep);
    
    /**
     * @brief Calculate chemical precipitation and sedimentation
     * 
     * @param timeStep Time increment in million years
     */
    void calculatePrecipitationAndSedimentation(double timeStep);
    
    /**
     * @brief Calculate biological processes in oceans
     * 
     * @param timeStep Time increment in million years
     */
    void calculateBiologicalProcesses(double timeStep);
    
    /**
     * @brief Apply chemical effects to climate system
     */
    void applyChemicalEffectsToClimate();
    
    /**
     * @brief Get chemical concentration at a specific position
     * 
     * @param position Position on the planet surface
     * @param element Chemical element to query
     * @return float Concentration value
     */
    float getChemicalConcentration(const Vec3f& position, ChemicalElement element) const;
    
    /**
     * @brief Get total river input to oceans for a chemical element
     * 
     * @param element Chemical element to query
     * @return float Total input value
     */
    float getTotalRiverInput(ChemicalElement element) const;
    
    /**
     * @brief Get average ocean concentration for a chemical element
     * 
     * @param element Chemical element to query
     * @return float Average concentration value
     */
    float getAverageOceanConcentration(ChemicalElement element) const;
    
    /**
     * @brief Get history of river inputs
     * 
     * @param element Chemical element to query
     * @return const std::vector<float>& History of input values
     */
    const std::vector<float>& getRiverInputHistory(ChemicalElement element) const;
    
    /**
     * @brief Get history of ocean concentrations
     * 
     * @param element Chemical element to query
     * @return const std::vector<float>& History of concentration values
     */
    const std::vector<float>& getOceanConcentrationHistory(ChemicalElement element) const;
    
    /**
     * @brief Get element name as string
     * 
     * @param element Chemical element
     * @return std::string Name of the element
     */
    std::string getElementName(ChemicalElement element) const;
    
    /**
     * @brief Get measurement unit for an element
     * 
     * @param element Chemical element
     * @return std::string Measurement unit
     */
    std::string getElementUnit(ChemicalElement element) const;
    
private:
    Planet& planet;
    ClimateSystem& climate;
    const EnhancedErosion& erosion;
    RiverOceanChemistryParams params;
    
    // Chemical concentrations
    std::vector<std::array<float, static_cast<size_t>(ChemicalElement::Count)>> chemicalConcentrations;
    
    // Total river input to oceans
    std::array<float, static_cast<size_t>(ChemicalElement::Count)> totalRiverInput;
    
    // Average ocean concentrations
    std::array<float, static_cast<size_t>(ChemicalElement::Count)> averageOceanConcentration;
    
    // History tracking
    std::array<std::vector<float>, static_cast<size_t>(ChemicalElement::Count)> riverInputHistory;
    std::array<std::vector<float>, static_cast<size_t>(ChemicalElement::Count)> oceanConcentrationHistory;
    
    // Element names and units
    std::array<std::string, static_cast<size_t>(ChemicalElement::Count)> elementNames;
    std::array<std::string, static_cast<size_t>(ChemicalElement::Count)> elementUnits;
    
    // Initialize element names and units
    void initializeElementInfo();
    
    // Record current state for history tracking
    void recordCurrentState();
    
    // Get ocean vertex indices
    std::vector<int> getOceanVertices() const;
    
    // Get river mouth vertices
    std::vector<int> getRiverMouths() const;
    
    // Calculate river flow at a vertex
    float getRiverFlow(int vertexIdx) const;
};

} // namespace aeonterracpp