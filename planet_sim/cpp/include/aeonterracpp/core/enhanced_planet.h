#pragma once

#include "aeonterracpp/core/planet.h"
#include <vector>

namespace aeonterracpp {

/**
 * @brief Extended Planet class with additional properties for enhanced tectonic simulation
 * 
 * This class extends the base Planet class with properties needed for more
 * realistic tectonic simulations, including lithospheric properties, crust age,
 * and flexural parameters.
 */
class EnhancedPlanet : public Planet {
public:
    /**
     * @brief Construct a new Enhanced Planet
     * 
     * @param radius Planet radius in kilometers
     * @param resolution Grid resolution level (0-9)
     * @param name Optional name for the planet
     */
    EnhancedPlanet(float radius = 6371.0f, int resolution = 5, const std::string& name = "AeonTerra");
    
    /**
     * @brief Move constructor
     */
    EnhancedPlanet(EnhancedPlanet&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    EnhancedPlanet& operator=(EnhancedPlanet&& other) noexcept;
    
    /**
     * @brief Destructor
     */
    ~EnhancedPlanet();
    
    //-------------------------------------------------------------------------
    // Enhanced tectonic properties
    //-------------------------------------------------------------------------
    
    /**
     * @brief Get lithospheric thickness at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Lithospheric thickness in kilometers
     */
    float getLithosphereThickness(int vertexIdx) const;
    
    /**
     * @brief Set lithospheric thickness at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Lithospheric thickness in kilometers
     */
    void setLithosphereThickness(int vertexIdx, float value);
    
    /**
     * @brief Get crust age at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Crust age in million years
     */
    float getCrustAge(int vertexIdx) const;
    
    /**
     * @brief Set crust age at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Crust age in million years
     */
    void setCrustAge(int vertexIdx, float value);
    
    /**
     * @brief Get flexural parameter at a specific vertex
     * This parameter determines how the lithosphere responds to loading
     * 
     * @param vertexIdx Vertex index
     * @return float Flexural parameter (higher values = stiffer lithosphere)
     */
    float getFlexuralParameter(int vertexIdx) const;
    
    /**
     * @brief Set flexural parameter at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Flexural parameter
     */
    void setFlexuralParameter(int vertexIdx, float value);
    
    /**
     * @brief Get heat flow at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Heat flow in mW/m²
     */
    float getHeatFlow(int vertexIdx) const;
    
    /**
     * @brief Set heat flow at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Heat flow in mW/m²
     */
    void setHeatFlow(int vertexIdx, float value);
    
    /**
     * @brief Get strain rate at a specific vertex
     * This represents the deformation rate of the lithosphere
     * 
     * @param vertexIdx Vertex index
     * @return float Strain rate
     */
    float getStrainRate(int vertexIdx) const;
    
    /**
     * @brief Set strain rate at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Strain rate
     */
    void setStrainRate(int vertexIdx, float value);
    
    /**
     * @brief Initialize lithosphere properties based on crust type
     * Sets reasonable defaults for lithospheric thickness, crust age, etc.
     */
    void initializeLithosphereProperties();
    
    /**
     * @brief Update crust age during a simulation step
     * 
     * @param timeStep Time increment in million years
     */
    void updateCrustAge(double timeStep);
    
    /**
     * @brief Calculate lithospheric thermal subsidence
     * As oceanic lithosphere ages, it cools and subsides
     * 
     * @param timeStep Time increment in million years
     */
    void calculateThermalSubsidence(double timeStep);

private:
    // Additional properties for enhanced tectonic simulation
    std::vector<float> lithosphereThickness;  // Thickness of lithosphere (km)
    std::vector<float> crustAge;              // Age of crust (million years)
    std::vector<float> flexuralParameter;     // Flexural rigidity parameter
    std::vector<float> heatFlow;              // Heat flow (mW/m²)
    std::vector<float> strainRate;            // Lithospheric strain rate
    
    // Initialize enhanced properties
    void initializeEnhancedProperties();
};

} // namespace aeonterracpp