#include "aeonterracpp/core/river_ocean_chemistry.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace aeonterracpp {

RiverOceanChemistry::RiverOceanChemistry(Planet& planet,
                                       ClimateSystem& climate,
                                       const EnhancedErosion& erosion)
    : planet(planet), climate(climate), erosion(erosion) {
    // Initialize with default parameters
    initialize();
}

void RiverOceanChemistry::initialize(const RiverOceanChemistryParams& params) {
    this->params = params;
    
    // Initialize element names and units
    initializeElementInfo();
    
    // Initialize chemical concentrations
    size_t vertexCount = planet.getVertexCount();
    chemicalConcentrations.resize(vertexCount);
    
    for (auto& concentrations : chemicalConcentrations) {
        concentrations.fill(0.0f);
    }
    
    // Initialize total river input
    totalRiverInput.fill(0.0f);
    
    // Initialize average ocean concentration
    averageOceanConcentration.fill(0.0f);
    
    // Clear history vectors
    for (auto& history : riverInputHistory) {
        history.clear();
    }
    
    for (auto& history : oceanConcentrationHistory) {
        history.clear();
    }
    
    std::cout << "River-ocean chemistry system initialized with " 
              << vertexCount << " vertices." << std::endl;
}

void RiverOceanChemistry::setParameters(const RiverOceanChemistryParams& params) {
    this->params = params;
}

const RiverOceanChemistryParams& RiverOceanChemistry::getParameters() const {
    return params;
}

void RiverOceanChemistry::update(double timeStep) {
    // Scale time step by the time scale factor
    double scaledTimeStep = timeStep * params.timeScaleFactor;
    
    // Record current state for history
    recordCurrentState();
    
    // Calculate river input to oceans
    calculateRiverInput();
    
    // Calculate ocean circulation and mixing
    calculateOceanMixing(scaledTimeStep);
    
    // Calculate chemical precipitation and sedimentation
    calculatePrecipitationAndSedimentation(scaledTimeStep);
    
    // Calculate biological processes in oceans
    calculateBiologicalProcesses(scaledTimeStep);
    
    // Apply chemical effects to climate system
    applyChemicalEffectsToClimate();
    
    std::cout << "Updated river-ocean chemistry for " << scaledTimeStep << " million years." << std::endl;
}

void RiverOceanChemistry::calculateRiverInput() {
    // Get river mouths (land cells adjacent to ocean)
    std::vector<int> riverMouths = getRiverMouths();
    
    // Reset total river input
    totalRiverInput.fill(0.0f);
    
    // Get the hydrological system
    const HydrologicalSystem& hydrology = climate.getHydrology();
    
    for (int vertexIdx : riverMouths) {
        // Get position
        Vec3f pos = planet.getGrid().getVertex(vertexIdx);
        
        // Get river flow at this vertex
        float riverFlow = getRiverFlow(vertexIdx);
        
        // Skip if no significant flow
        if (riverFlow < 1.0f) {
            continue;
        }
        
        // Calculate sediment input from erosion
        float erosionRate = erosion.getErosionRate(pos);
        float sedimentInput = erosionRate * params.riverErosionSedimentFactor * riverFlow;
        
        // Calculate nutrient input (nitrogen, phosphorus)
        float precipitation = hydrology.getPrecipitation(pos);
        float nitrogenInput = precipitation * params.riverNutrientLeachingRate * 0.01f * riverFlow;
        float phosphorusInput = precipitation * params.riverNutrientLeachingRate * 0.001f * riverFlow;
        
        // Calculate carbon input
        float carbonInput = params.riverCarbonExportRate * 0.1f * riverFlow;
        
        // Calculate oxygen input (from turbulence)
        float oxygenInput = riverFlow * 0.5f;
        
        // Calculate mineral inputs
        float calciumInput = erosionRate * 0.2f * riverFlow;
        float silicaInput = erosionRate * 0.3f * riverFlow;
        float ironInput = erosionRate * 0.05f * riverFlow;
        
        // Add to local concentrations
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Sediment)] += sedimentInput;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Nitrogen)] += nitrogenInput;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Phosphorus)] += phosphorusInput;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Carbon)] += carbonInput;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Oxygen)] += oxygenInput;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Calcium)] += calciumInput;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Silica)] += silicaInput;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Iron)] += ironInput;
        
        // Add to total river input
        totalRiverInput[static_cast<size_t>(ChemicalElement::Sediment)] += sedimentInput;
        totalRiverInput[static_cast<size_t>(ChemicalElement::Nitrogen)] += nitrogenInput;
        totalRiverInput[static_cast<size_t>(ChemicalElement::Phosphorus)] += phosphorusInput;
        totalRiverInput[static_cast<size_t>(ChemicalElement::Carbon)] += carbonInput;
        totalRiverInput[static_cast<size_t>(ChemicalElement::Oxygen)] += oxygenInput;
        totalRiverInput[static_cast<size_t>(ChemicalElement::Calcium)] += calciumInput;
        totalRiverInput[static_cast<size_t>(ChemicalElement::Silica)] += silicaInput;
        totalRiverInput[static_cast<size_t>(ChemicalElement::Iron)] += ironInput;
    }
}

void RiverOceanChemistry::calculateOceanMixing(double timeStep) {
    // Get ocean vertices
    std::vector<int> oceanVertices = getOceanVertices();
    
    // Skip if no ocean
    if (oceanVertices.empty()) {
        return;
    }
    
    // Calculate mixing for each chemical element
    for (size_t elementIdx = 0; elementIdx < static_cast<size_t>(ChemicalElement::Count); ++elementIdx) {
        // Create buffer for new concentrations
        std::vector<float> newConcentrations(oceanVertices.size(), 0.0f);
        
        // For each ocean vertex
        for (size_t i = 0; i < oceanVertices.size(); ++i) {
            int vertexIdx = oceanVertices[i];
            
            // Get current concentration
            float currentConc = chemicalConcentrations[vertexIdx][elementIdx];
            
            // Get position
            Vec3f pos = planet.getGrid().getVertex(vertexIdx);
            
            // Get ocean currents from climate system
            Vec3f oceanCurrents = climate.getOceans().getCurrentVector(pos);
            float currentSpeed = oceanCurrents.norm();
            
            // Higher mixing for stronger currents
            float mixingFactor = params.oceanCirculationMixingRate * 
                                (0.5f + 0.5f * std::min(1.0f, currentSpeed / 10.0f));
            
            // Get neighbors
            const auto& neighbors = planet.getGrid().getNeighbors(vertexIdx);
            
            float neighborSum = 0.0f;
            int oceanNeighborCount = 0;
            
            // Sum up neighbor concentrations
            for (int neighborIdx : neighbors) {
                // Only mix with other ocean cells
                if (planet.getElevation(neighborIdx) < 0.0f) {
                    neighborSum += chemicalConcentrations[neighborIdx][elementIdx];
                    oceanNeighborCount++;
                }
            }
            
            // Calculate average neighbor concentration
            float neighborAvg = (oceanNeighborCount > 0) ? 
                               neighborSum / oceanNeighborCount : currentConc;
            
            // Mix current with neighbors
            float mixedConc = (1.0f - mixingFactor) * currentConc + 
                             mixingFactor * neighborAvg;
            
            // Store in buffer
            newConcentrations[i] = mixedConc;
        }
        
        // Update concentrations from buffer
        for (size_t i = 0; i < oceanVertices.size(); ++i) {
            int vertexIdx = oceanVertices[i];
            chemicalConcentrations[vertexIdx][elementIdx] = newConcentrations[i];
        }
    }
    
    // Calculate average ocean concentrations
    for (size_t elementIdx = 0; elementIdx < static_cast<size_t>(ChemicalElement::Count); ++elementIdx) {
        float sum = 0.0f;
        
        for (int vertexIdx : oceanVertices) {
            sum += chemicalConcentrations[vertexIdx][elementIdx];
        }
        
        averageOceanConcentration[elementIdx] = sum / oceanVertices.size();
    }
}

void RiverOceanChemistry::calculatePrecipitationAndSedimentation(double timeStep) {
    // Get ocean vertices
    std::vector<int> oceanVertices = getOceanVertices();
    
    // Skip if no ocean
    if (oceanVertices.empty()) {
        return;
    }
    
    // For each ocean vertex
    for (int vertexIdx : oceanVertices) {
        // Get position
        Vec3f pos = planet.getGrid().getVertex(vertexIdx);
        
        // Get depth (negative elevation)
        float depth = -planet.getElevation(vertexIdx);
        
        // Get current concentrations
        float sedimentConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Sediment)];
        float calciumConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Calcium)];
        float carbonConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Carbon)];
        float silicaConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Silica)];
        float ironConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Iron)];
        float oxygenConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Oxygen)];
        
        // Calculate sedimentation (more in shallow water)
        float depthFactor = std::max(0.1f, 1.0f - depth / 10.0f);
        float sedimentationRate = params.oceanSedimentationRate * depthFactor;
        float sedimentDeposited = sedimentConc * sedimentationRate * static_cast<float>(timeStep);
        
        // Calculate carbonate precipitation (calcium + carbon)
        float carbonatePrecipRate = params.carbonatePrecipitationRate * 
                                   std::min(calciumConc, carbonConc);
        float carbonateDeposited = carbonatePrecipRate * static_cast<float>(timeStep);
        
        // Calculate silica precipitation
        float silicaPrecipRate = params.silicaPrecipitationRate * silicaConc;
        float silicaDeposited = silicaPrecipRate * static_cast<float>(timeStep);
        
        // Calculate iron oxidation (iron + oxygen)
        float ironOxidationRate = params.ironOxidationRate * 
                                 std::min(ironConc, oxygenConc);
        float ironOxideDeposited = ironOxidationRate * static_cast<float>(timeStep);
        
        // Update concentrations
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Sediment)] -= sedimentDeposited;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Calcium)] -= carbonateDeposited;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Carbon)] -= carbonateDeposited;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Silica)] -= silicaDeposited;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Iron)] -= ironOxideDeposited;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Oxygen)] -= ironOxideDeposited;
        
        // Ensure non-negative concentrations
        for (size_t elementIdx = 0; elementIdx < static_cast<size_t>(ChemicalElement::Count); ++elementIdx) {
            chemicalConcentrations[vertexIdx][elementIdx] = 
                std::max(0.0f, chemicalConcentrations[vertexIdx][elementIdx]);
        }
    }
}

void RiverOceanChemistry::calculateBiologicalProcesses(double timeStep) {
    // Get ocean vertices
    std::vector<int> oceanVertices = getOceanVertices();
    
    // Skip if no ocean
    if (oceanVertices.empty()) {
        return;
    }
    
    // For each ocean vertex
    for (int vertexIdx : oceanVertices) {
        // Get position
        Vec3f pos = planet.getGrid().getVertex(vertexIdx);
        
        // Get ocean temperature from climate
        float temperature = climate.getOceans().getTemperature(pos);
        
        // Get current concentrations
        float nitrogenConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Nitrogen)];
        float phosphorusConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Phosphorus)];
        float carbonConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Carbon)];
        float oxygenConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Oxygen)];
        float silicaConc = chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Silica)];
        
        // Calculate temperature factor (productivity peaks around 15-25°C)
        float temperatureFactor = std::exp(-std::pow((temperature - 20.0f) / 10.0f, 2.0f));
        
        // Calculate nutrient limitation factor (Liebig's Law of the Minimum)
        float nitrogenFactor = nitrogenConc / (nitrogenConc + 0.1f);
        float phosphorusFactor = phosphorusConc / (phosphorusConc + 0.01f);
        float nutrientFactor = std::min(nitrogenFactor, phosphorusFactor);
        
        // Calculate primary production rate
        float primaryProduction = params.oceanPrimaryProductionRate * 
                                 temperatureFactor * nutrientFactor * 
                                 static_cast<float>(timeStep);
        
        // Limit by available carbon
        primaryProduction = std::min(primaryProduction, carbonConc * 0.5f);
        
        // Produce oxygen from primary production
        float oxygenProduced = primaryProduction * 2.0f;
        
        // Consume nutrients
        float nitrogenConsumed = primaryProduction * 0.15f;
        float phosphorusConsumed = primaryProduction * 0.01f;
        float silicaConsumed = primaryProduction * 0.1f * silicaConc / (silicaConc + 0.1f);
        
        // Update concentrations
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Nitrogen)] -= nitrogenConsumed;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Phosphorus)] -= phosphorusConsumed;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Carbon)] -= primaryProduction;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Oxygen)] += oxygenProduced;
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Silica)] -= silicaConsumed;
        
        // Add atmospheric exchange for oxygen
        float oxygenAtmosphericExchange = params.oceanOxygenationRate * 
                                         (20.0f - oxygenConc) * 0.1f * 
                                         static_cast<float>(timeStep);
        
        chemicalConcentrations[vertexIdx][static_cast<size_t>(ChemicalElement::Oxygen)] += oxygenAtmosphericExchange;
        
        // Ensure non-negative concentrations
        for (size_t elementIdx = 0; elementIdx < static_cast<size_t>(ChemicalElement::Count); ++elementIdx) {
            chemicalConcentrations[vertexIdx][elementIdx] = 
                std::max(0.0f, chemicalConcentrations[vertexIdx][elementIdx]);
        }
    }
}

void RiverOceanChemistry::applyChemicalEffectsToClimate() {
    // In a real implementation, this would modify climate system parameters
    // For example, river sediment could affect coastal currents,
    // ocean chemistry could affect heat absorption, etc.
    
    // For now, we'll just print the average ocean concentrations
    std::cout << "Ocean chemistry summary:" << std::endl;
    
    for (size_t elementIdx = 0; elementIdx < static_cast<size_t>(ChemicalElement::Count); ++elementIdx) {
        std::string name = elementNames[elementIdx];
        std::string unit = elementUnits[elementIdx];
        float concentration = averageOceanConcentration[elementIdx];
        
        std::cout << "  " << name << ": " << concentration << " " << unit << std::endl;
    }
}

float RiverOceanChemistry::getChemicalConcentration(const Vec3f& position, ChemicalElement element) const {
    int vertexIdx = planet.getGrid().findClosestVertex(position);
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(chemicalConcentrations.size())) {
        return chemicalConcentrations[vertexIdx][static_cast<size_t>(element)];
    }
    
    return 0.0f;
}

float RiverOceanChemistry::getTotalRiverInput(ChemicalElement element) const {
    return totalRiverInput[static_cast<size_t>(element)];
}

float RiverOceanChemistry::getAverageOceanConcentration(ChemicalElement element) const {
    return averageOceanConcentration[static_cast<size_t>(element)];
}

const std::vector<float>& RiverOceanChemistry::getRiverInputHistory(ChemicalElement element) const {
    return riverInputHistory[static_cast<size_t>(element)];
}

const std::vector<float>& RiverOceanChemistry::getOceanConcentrationHistory(ChemicalElement element) const {
    return oceanConcentrationHistory[static_cast<size_t>(element)];
}

std::string RiverOceanChemistry::getElementName(ChemicalElement element) const {
    return elementNames[static_cast<size_t>(element)];
}

std::string RiverOceanChemistry::getElementUnit(ChemicalElement element) const {
    return elementUnits[static_cast<size_t>(element)];
}

void RiverOceanChemistry::initializeElementInfo() {
    // Initialize element names
    elementNames[static_cast<size_t>(ChemicalElement::Sediment)] = "Sediment";
    elementNames[static_cast<size_t>(ChemicalElement::Nitrogen)] = "Nitrogen";
    elementNames[static_cast<size_t>(ChemicalElement::Phosphorus)] = "Phosphorus";
    elementNames[static_cast<size_t>(ChemicalElement::Carbon)] = "Carbon";
    elementNames[static_cast<size_t>(ChemicalElement::Oxygen)] = "Oxygen";
    elementNames[static_cast<size_t>(ChemicalElement::Calcium)] = "Calcium";
    elementNames[static_cast<size_t>(ChemicalElement::Silica)] = "Silica";
    elementNames[static_cast<size_t>(ChemicalElement::Iron)] = "Iron";
    
    // Initialize element units
    elementUnits[static_cast<size_t>(ChemicalElement::Sediment)] = "mg/L";
    elementUnits[static_cast<size_t>(ChemicalElement::Nitrogen)] = "mg/L";
    elementUnits[static_cast<size_t>(ChemicalElement::Phosphorus)] = "mg/L";
    elementUnits[static_cast<size_t>(ChemicalElement::Carbon)] = "mg/L";
    elementUnits[static_cast<size_t>(ChemicalElement::Oxygen)] = "mg/L";
    elementUnits[static_cast<size_t>(ChemicalElement::Calcium)] = "mg/L";
    elementUnits[static_cast<size_t>(ChemicalElement::Silica)] = "mg/L";
    elementUnits[static_cast<size_t>(ChemicalElement::Iron)] = "mg/L";
}

void RiverOceanChemistry::recordCurrentState() {
    // Save current state for history tracking
    for (size_t elementIdx = 0; elementIdx < static_cast<size_t>(ChemicalElement::Count); ++elementIdx) {
        riverInputHistory[elementIdx].push_back(totalRiverInput[elementIdx]);
        oceanConcentrationHistory[elementIdx].push_back(averageOceanConcentration[elementIdx]);
    }
}

std::vector<int> RiverOceanChemistry::getOceanVertices() const {
    std::vector<int> oceanVertices;
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        if (planet.getElevation(vertexIdx) < 0.0f) {
            oceanVertices.push_back(vertexIdx);
        }
    }
    
    return oceanVertices;
}

std::vector<int> RiverOceanChemistry::getRiverMouths() const {
    std::vector<int> riverMouths;
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Skip ocean cells
        if (planet.getElevation(vertexIdx) < 0.0f) {
            continue;
        }
        
        // Check if this is adjacent to an ocean cell
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        bool isRiverMouth = false;
        
        for (int neighborIdx : neighbors) {
            if (planet.getElevation(neighborIdx) < 0.0f) {
                isRiverMouth = true;
                break;
            }
        }
        
        if (isRiverMouth) {
            riverMouths.push_back(vertexIdx);
        }
    }
    
    return riverMouths;
}

float RiverOceanChemistry::getRiverFlow(int vertexIdx) const {
    const HydrologicalSystem& hydrology = climate.getHydrology();
    const std::vector<float>& riverFlow = hydrology.getRiverFlowData();
    
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(riverFlow.size())) {
        return riverFlow[vertexIdx];
    }
    
    return 0.0f;
}

} // namespace aeonterracpp