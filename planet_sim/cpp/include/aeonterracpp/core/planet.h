#pragma once

#include "aeonterracpp/core/spherical_grid.h"
#include <string>
#include <vector>
#include <memory>

namespace aeonterracpp {

/**
 * @brief Represents a snapshot of planet state at a specific point in time
 */
struct HistoricalSnapshot {
    double age;                       // Age in million years
    std::vector<float> elevation;     // Snapshot of elevation values
    std::vector<int> plateId;         // Snapshot of plate assignments
    
    // Additional statistics
    float maxElevation;
    float minElevation;
    float oceanPercentage;
    int plateCount;
    
    // Constructor
    HistoricalSnapshot(double _age, size_t vertexCount);
};

/**
 * @brief Core class representing a planet and its properties
 * 
 * The Planet class encapsulates all data related to a planet, including
 * its grid structure, physical properties (elevation, crust thickness, etc.),
 * and history. It provides methods for accessing and modifying properties,
 * controlling simulation, and serialization.
 */
class Planet {
public:
    /**
     * @brief Construct a new Planet
     * 
     * @param radius Planet radius in kilometers
     * @param resolution Grid resolution level (0-9)
     * @param name Optional name for the planet
     */
    Planet(float radius = 6371.0f, int resolution = 5, const std::string& name = "AeonTerra");
    
    /**
     * @brief Copy constructor is deleted because SphericalGrid contains a unique_ptr
     */
    Planet(const Planet&) = delete;
    
    /**
     * @brief Copy assignment is deleted because SphericalGrid contains a unique_ptr
     */
    Planet& operator=(const Planet&) = delete;
    
    /**
     * @brief Move constructor
     */
    Planet(Planet&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    Planet& operator=(Planet&& other) noexcept;
    
    /**
     * @brief Destructor
     */
    ~Planet();
    
    //-------------------------------------------------------------------------
    // Property access methods
    //-------------------------------------------------------------------------
    
    /**
     * @brief Get elevation at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Elevation in kilometers relative to sea level
     */
    float getElevation(int vertexIdx) const;
    
    /**
     * @brief Set elevation at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Elevation in kilometers relative to sea level
     */
    void setElevation(int vertexIdx, float value);
    
    /**
     * @brief Get crust thickness at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Crust thickness in kilometers
     */
    float getCrustThickness(int vertexIdx) const;
    
    /**
     * @brief Set crust thickness at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Crust thickness in kilometers
     */
    void setCrustThickness(int vertexIdx, float value);
    
    /**
     * @brief Get crust density at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Crust density in g/cm³
     */
    float getCrustDensity(int vertexIdx) const;
    
    /**
     * @brief Set crust density at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Crust density in g/cm³
     */
    void setCrustDensity(int vertexIdx, float value);
    
    /**
     * @brief Get crust type at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return int Crust type (0 = oceanic, 1 = continental, others as defined)
     */
    int getCrustType(int vertexIdx) const;
    
    /**
     * @brief Set crust type at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Crust type
     */
    void setCrustType(int vertexIdx, int value);
    
    /**
     * @brief Get plate ID at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @return int Plate ID (-1 if not assigned to a plate)
     */
    int getPlateId(int vertexIdx) const;
    
    /**
     * @brief Set plate ID at a specific vertex
     * 
     * @param vertexIdx Vertex index
     * @param value Plate ID
     */
    void setPlateId(int vertexIdx, int value);
    
    //-------------------------------------------------------------------------
    // Grid access methods
    //-------------------------------------------------------------------------
    
    /**
     * @brief Get reference to the underlying grid
     * 
     * @return const SphericalGrid& Reference to the grid
     */
    const SphericalGrid& getGrid() const;
    
    /**
     * @brief Get vertex count
     * 
     * @return size_t Number of vertices in the planet grid
     */
    size_t getVertexCount() const;
    
    /**
     * @brief Get planet radius
     * 
     * @return float Radius in kilometers
     */
    float getRadius() const;
    
    //-------------------------------------------------------------------------
    // Simulation control methods
    //-------------------------------------------------------------------------
    
    /**
     * @brief Advance planet age
     * 
     * @param timeStep Time increment in million years
     */
    void advanceAge(double timeStep);
    
    /**
     * @brief Get current planet age
     * 
     * @return double Age in million years
     */
    double getAge() const;
    
    /**
     * @brief Record current state as a historical snapshot
     */
    void recordSnapshot();
    
    /**
     * @brief Get planet name
     * 
     * @return const std::string& Planet name
     */
    const std::string& getName() const;
    
    /**
     * @brief Set planet name
     * 
     * @param name New planet name
     */
    void setName(const std::string& name);
    
    //-------------------------------------------------------------------------
    // History access methods
    //-------------------------------------------------------------------------
    
    /**
     * @brief Get historical snapshots
     * 
     * @return const std::vector<HistoricalSnapshot>& Vector of snapshots
     */
    const std::vector<HistoricalSnapshot>& getHistory() const;
    
    /**
     * @brief Clear historical snapshots
     */
    void clearHistory();
    
    //-------------------------------------------------------------------------
    // Serialization methods
    //-------------------------------------------------------------------------
    
    /**
     * @brief Save planet to a file
     * 
     * @param filename Output filename
     */
    void saveToFile(const std::string& filename) const;
    
    /**
     * @brief Load planet from a file
     * 
     * @param filename Input filename
     * @return Planet Loaded planet
     */
    static Planet loadFromFile(const std::string& filename);
    
private:
    // Core geometry
    SphericalGrid grid;
    
    // Properties (using SOA layout for performance)
    std::vector<float> elevation;       // Relative to sea level (km)
    std::vector<float> crustThickness;  // Thickness of crust (km)
    std::vector<float> crustDensity;    // Density of crust (g/cm³)
    std::vector<int> crustType;         // Type of crust (0=oceanic, 1=continental)
    std::vector<int> plateId;           // Tectonic plate assignment
    
    // Metadata
    double age;            // Planet age in million years
    std::string name;      // Planet name
    
    // History tracking
    std::vector<HistoricalSnapshot> history;
    
    // Initialize planet properties
    void initializeProperties();
};

} // namespace aeonterracpp