#include "aeonterracpp/core/enhanced_planet.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace aeonterracpp {

EnhancedPlanet::EnhancedPlanet(float radius, int resolution, const std::string& name)
    : Planet(radius, resolution, name) {
    initializeEnhancedProperties();
}

EnhancedPlanet::EnhancedPlanet(EnhancedPlanet&& other) noexcept
    : Planet(std::move(other)),
      lithosphereThickness(std::move(other.lithosphereThickness)),
      crustAge(std::move(other.crustAge)),
      flexuralParameter(std::move(other.flexuralParameter)),
      heatFlow(std::move(other.heatFlow)),
      strainRate(std::move(other.strainRate)) {
}

EnhancedPlanet& EnhancedPlanet::operator=(EnhancedPlanet&& other) noexcept {
    if (this != &other) {
        Planet::operator=(std::move(other));
        lithosphereThickness = std::move(other.lithosphereThickness);
        crustAge = std::move(other.crustAge);
        flexuralParameter = std::move(other.flexuralParameter);
        heatFlow = std::move(other.heatFlow);
        strainRate = std::move(other.strainRate);
    }
    return *this;
}

EnhancedPlanet::~EnhancedPlanet() {
    // Nothing specific to clean up
}

float EnhancedPlanet::getLithosphereThickness(int vertexIdx) const {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(lithosphereThickness.size())) {
        return lithosphereThickness[vertexIdx];
    }
    return 0.0f;
}

void EnhancedPlanet::setLithosphereThickness(int vertexIdx, float value) {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(lithosphereThickness.size())) {
        lithosphereThickness[vertexIdx] = value;
    }
}

float EnhancedPlanet::getCrustAge(int vertexIdx) const {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(crustAge.size())) {
        return crustAge[vertexIdx];
    }
    return 0.0f;
}

void EnhancedPlanet::setCrustAge(int vertexIdx, float value) {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(crustAge.size())) {
        crustAge[vertexIdx] = value;
    }
}

float EnhancedPlanet::getFlexuralParameter(int vertexIdx) const {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(flexuralParameter.size())) {
        return flexuralParameter[vertexIdx];
    }
    return 0.0f;
}

void EnhancedPlanet::setFlexuralParameter(int vertexIdx, float value) {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(flexuralParameter.size())) {
        flexuralParameter[vertexIdx] = value;
    }
}

float EnhancedPlanet::getHeatFlow(int vertexIdx) const {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(heatFlow.size())) {
        return heatFlow[vertexIdx];
    }
    return 0.0f;
}

void EnhancedPlanet::setHeatFlow(int vertexIdx, float value) {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(heatFlow.size())) {
        heatFlow[vertexIdx] = value;
    }
}

float EnhancedPlanet::getStrainRate(int vertexIdx) const {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(strainRate.size())) {
        return strainRate[vertexIdx];
    }
    return 0.0f;
}

void EnhancedPlanet::setStrainRate(int vertexIdx, float value) {
    if (vertexIdx >= 0 && vertexIdx < static_cast<int>(strainRate.size())) {
        strainRate[vertexIdx] = value;
    }
}

void EnhancedPlanet::initializeEnhancedProperties() {
    size_t vertexCount = getVertexCount();
    lithosphereThickness.resize(vertexCount, 0.0f);
    crustAge.resize(vertexCount, 0.0f);
    flexuralParameter.resize(vertexCount, 0.0f);
    heatFlow.resize(vertexCount, 0.0f);
    strainRate.resize(vertexCount, 0.0f);
    
    // Initialize with default values
    initializeLithosphereProperties();
}

void EnhancedPlanet::initializeLithosphereProperties() {
    size_t vertexCount = getVertexCount();
    
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        int crustType = getCrustType(vertexIdx);
        
        // Set lithosphere thickness based on crust type
        if (crustType == 0) { // Oceanic
            // Oceanic lithosphere is typically thinner
            lithosphereThickness[i] = 80.0f; // Typical oceanic lithosphere thickness (km)
        } else { // Continental
            // Continental lithosphere is thicker
            lithosphereThickness[i] = 150.0f; // Typical continental lithosphere thickness (km)
        }
        
        // Initialize crust age (new crust at plate boundaries has age 0)
        crustAge[i] = getAge(); // Default to planet age
        
        // Set flexural parameter based on lithosphere thickness
        // This follows the relationship D = E*T^3/(12*(1-v^2)) where:
        // D = flexural rigidity, E = Young's modulus, T = thickness, v = Poisson's ratio
        float thickness = lithosphereThickness[i];
        flexuralParameter[i] = 0.01f * std::pow(thickness, 3.0f);
        
        // Set heat flow based on crust type and age
        if (crustType == 0) { // Oceanic
            // Oceanic heat flow decreases with age
            float age = std::max(1.0f, static_cast<float>(getAge()));
            heatFlow[i] = 250.0f / std::sqrt(age); // mW/m²
        } else { // Continental
            // Continental heat flow is lower and more stable
            heatFlow[i] = 65.0f; // mW/m²
        }
        
        // Initialize strain rate to zero
        strainRate[i] = 0.0f;
    }
    
    std::cout << "Initialized enhanced lithosphere properties" << std::endl;
}

void EnhancedPlanet::updateCrustAge(double timeStep) {
    for (size_t i = 0; i < crustAge.size(); ++i) {
        crustAge[i] += static_cast<float>(timeStep);
    }
}

void EnhancedPlanet::calculateThermalSubsidence(double timeStep) {
    // Implement oceanic lithosphere thermal subsidence
    // As the oceanic lithosphere moves away from ridges, it cools, contracts, and subsides
    // This follows a square root of age relationship: d = d0 + c * sqrt(age)
    // where d is depth, d0 is ridge depth, c is a constant, age is lithosphere age
    
    const float subsidenceConstant = 0.35f; // km/sqrt(Myr)
    const float baseOceanicDepth = -2.5f;   // km (depth at ridge)
    
    for (size_t i = 0; i < crustAge.size(); ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Only apply to oceanic crust
        if (getCrustType(vertexIdx) == 0) {
            // Get current elevation and age
            float elevation = getElevation(vertexIdx);
            float age = crustAge[i];
            
            // Skip very young crust (at ridges)
            if (age < 2.0f) {
                continue;
            }
            
            // Calculate target depth based on thermal subsidence model
            float targetDepth = baseOceanicDepth - subsidenceConstant * std::sqrt(age);
            
            // Limit maximum depth
            targetDepth = std::max(targetDepth, -11.0f);
            
            // Gradually adjust toward target depth (smooth transition)
            float adjustmentRate = 0.1f * static_cast<float>(timeStep);
            float newElevation = elevation * (1.0f - adjustmentRate) + targetDepth * adjustmentRate;
            
            // Apply the new elevation
            setElevation(vertexIdx, newElevation);
            
            // Update lithosphere thickness based on age
            // Lithosphere thickens as it cools: T = c * sqrt(age)
            float newThickness = 10.0f + 10.0f * std::sqrt(age);
            newThickness = std::min(newThickness, 100.0f); // Cap maximum thickness
            lithosphereThickness[i] = newThickness;
            
            // Update heat flow based on age
            // Heat flow decreases with age: Q = Q0 / sqrt(age)
            float newHeatFlow = 300.0f / std::sqrt(std::max(1.0f, age));
            heatFlow[i] = newHeatFlow;
        }
    }
}

} // namespace aeonterracpp