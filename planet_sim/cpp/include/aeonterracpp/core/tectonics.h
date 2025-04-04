#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/vector_types.h"
#include <memory>
#include <random>
#include <vector>

namespace aeonterracpp {

/**
 * @brief Represents a tectonic plate
 */
class TectonicPlate {
public:
    /**
     * @brief Construct a new Tectonic Plate
     * 
     * @param id Unique plate identifier
     * @param center Center position of the plate
     */
    TectonicPlate(int id, const Vec3f& center);
    
    /**
     * @brief Get the plate ID
     */
    int getId() const;
    
    /**
     * @brief Get the plate center
     */
    const Vec3f& getCenter() const;
    
    /**
     * @brief Get the plate movement vector
     */
    const Vec3f& getMovement() const;
    
    /**
     * @brief Set the plate movement vector
     * 
     * @param movement Movement vector in 3D space
     */
    void setMovement(const Vec3f& movement);
    
    /**
     * @brief Get plate vertices
     */
    const std::vector<int>& getVertices() const;
    
    /**
     * @brief Add a vertex to the plate
     * 
     * @param vertexIdx Vertex index
     */
    void addVertex(int vertexIdx);
    
    /**
     * @brief Calculate the plate area
     */
    float calculateArea() const;
    
    /**
     * @brief Is the plate oceanic?
     */
    bool isOceanic() const;
    
    /**
     * @brief Set the plate type
     * 
     * @param oceanic True if oceanic, false if continental
     */
    void setOceanic(bool oceanic);
    
    /**
     * @brief Calculate the plate's bounding box
     */
    BoundingBox calculateBoundingBox() const;
    
private:
    int id;                 // Unique identifier
    Vec3f center;           // Center position
    Vec3f movement;         // Movement vector
    std::vector<int> vertices;  // Vertices that belong to this plate
    bool oceanic;           // True if oceanic, false if continental
};

/**
 * @brief Represents a convection cell in the mantle
 */
class ConvectionCell {
public:
    /**
     * @brief Construct a new Convection Cell
     * 
     * @param center Center position of the cell
     * @param strength Convection strength
     */
    ConvectionCell(const Vec3f& center, float strength);
    
    /**
     * @brief Get the cell center
     */
    const Vec3f& getCenter() const;
    
    /**
     * @brief Get the cell strength
     */
    float getStrength() const;
    
    /**
     * @brief Calculate the force at a specific point
     * 
     * @param position Position to calculate force at
     * @return Vec3f Force vector
     */
    Vec3f calculateForceAt(const Vec3f& position) const;
    
private:
    Vec3f center;    // Center of the convection cell
    float strength;  // Strength of convection (positive=upwelling, negative=downwelling)
    float radius;    // Influence radius
};

/**
 * @brief Enumeration of different plate interaction types
 */
enum class PlateInteractionType {
    None,             // No interaction
    Divergent,        // Plates moving away from each other (create rift)
    Convergent,       // Plates moving toward each other (create mountains/trenches)
    Transform,        // Plates sliding past each other
    Subduction,       // Oceanic plate goes under another plate
    Collision         // Continental collision
};

/**
 * @brief Simulation of tectonic plate movement and interactions
 */
class TectonicSimulation {
public:
    /**
     * @brief Construct a new Tectonic Simulation
     * 
     * @param planet Reference to the planet
     * @param numPlates Number of initial plates to create
     * @param seed Random seed for reproducibility (0 for random)
     */
    TectonicSimulation(Planet& planet, int numPlates = 7, unsigned int seed = 0);
    
    /**
     * @brief Initialize plates using Voronoi-based segmentation
     */
    void initializePlates();
    
    /**
     * @brief Initialize mantle convection cells
     * 
     * @param numCells Number of convection cells to create
     */
    void initializeMantle(int numCells);
    
    /**
     * @brief Simulate a single time step
     * 
     * @param timeStep Time increment in million years
     */
    void simulateStep(double timeStep);
    
    /**
     * @brief Get all plate boundary vertices
     * 
     * @return std::vector<int> Indices of vertices on plate boundaries
     */
    std::vector<int> getBoundaryVertices() const;
    
    /**
     * @brief Get the interaction type between adjacent plates
     * 
     * @param v1 First vertex index
     * @param v2 Second vertex index
     * @return PlateInteractionType Type of interaction
     */
    PlateInteractionType getInteractionType(int v1, int v2) const;
    
    /**
     * @brief Get all tectonic plates
     * 
     * @return const std::vector<TectonicPlate>& Vector of plates
     */
    const std::vector<TectonicPlate>& getPlates() const;
    
    /**
     * @brief Set tectonic activity level
     * 
     * @param value Activity level (0.0-1.0, where 1.0 is maximum activity)
     */
    void setTectonicActivity(float value);
    
    /**
     * @brief Get tectonic activity level
     * 
     * @return float Activity level (0.0-1.0)
     */
    float getTectonicActivity() const;
    
    /**
     * @brief Calculate isostatic adjustment for a vertex
     * 
     * @param vertexIdx Vertex index
     * @return float Height adjustment in km
     */
    float calculateIsostaticAdjustment(int vertexIdx) const;
    
    /**
     * @brief Get current supercontinent cycle phase
     *
     * @return float Phase value between 0.0-1.0
     */
    float getSuperContinentCyclePhase() const;
    
private:
    // Reference to the planet
    Planet& planet;
    
    // Plates
    std::vector<TectonicPlate> plates;
    
    // Mantle convection cells
    std::vector<ConvectionCell> convectionCells;
    
    // Random number generator
    std::mt19937 rng;
    
    // Parameters
    struct {
        float mantleDensity;         // Density of mantle (g/cm³)
        float oceanicDensity;        // Density of oceanic crust (g/cm³)
        float continentalDensity;    // Density of continental crust (g/cm³)
        float tectonicActivityScale; // Scale factor for tectonic movement (0.0-1.0)
        float erosionRate;           // Rate of erosion
        float sedimentationRate;     // Rate of sedimentation
        float superContinentCyclePhase; // Position in the supercontinent cycle (0.0-1.0)
        float activityVariabilityPeriod; // Period for activity variation (million years)
        float riverFlowRate;         // Rate of river flow for hydrological processes
        float weatheringRate;        // Rate of chemical weathering
        bool enableClimateEffects;   // Whether to enable climate effects
    } params;
    
    // Mass balance tracking
    struct MassBalanceTracker {
        double initialContinentalVolume; // Initial volume of continental crust (km³)
        double initialOceanicVolume;     // Initial volume of oceanic crust (km³)
        double continentalVolumeCreated; // Volume of continental crust created (km³)
        double continentalVolumeEroded;  // Volume of continental crust eroded (km³)
        double oceanicVolumeCreated;     // Volume of oceanic crust created (km³)
        double oceanicVolumeSubducted;   // Volume of oceanic crust subducted (km³)
        
        // Reset the tracker
        void reset() {
            initialContinentalVolume = 0.0;
            initialOceanicVolume = 0.0;
            continentalVolumeCreated = 0.0;
            continentalVolumeEroded = 0.0;
            oceanicVolumeCreated = 0.0;
            oceanicVolumeSubducted = 0.0;
        }
        
        // Calculate net continental growth
        double getNetContinentalGrowth() const {
            return continentalVolumeCreated - continentalVolumeEroded;
        }
        
        // Calculate net oceanic growth
        double getNetOceanicGrowth() const {
            return oceanicVolumeCreated - oceanicVolumeSubducted;
        }
        
        // Get total continental volume
        double getTotalContinentalVolume() const {
            return initialContinentalVolume + getNetContinentalGrowth();
        }
        
        // Get total oceanic volume
        double getTotalOceanicVolume() const {
            return initialOceanicVolume + getNetOceanicGrowth();
        }
    };
    
    // Climate system
    struct ClimateSystem {
        std::vector<float> temperature; // Temperature at each vertex (°C)
        std::vector<float> precipitation; // Precipitation at each vertex (mm/year)
        std::vector<float> weatheringRate; // Weathering rate at each vertex
        
        // Initialize climate arrays
        void initialize(size_t vertexCount) {
            temperature.resize(vertexCount, 15.0f); // Default 15°C
            precipitation.resize(vertexCount, 1000.0f); // Default 1000mm/year
            weatheringRate.resize(vertexCount, 1.0f); // Default rate 1.0
        }
        
        // Update climate based on elevation and latitude
        void update(const SphericalGrid& grid, size_t vertexCount);
    };
    
    // Hydrological system
    struct HydrologicalSystem {
        std::vector<float> drainage; // Drainage area at each vertex
        std::vector<float> riverFlow; // River flow at each vertex
        std::vector<int> flowDirection; // Flow direction for each vertex
        std::vector<bool> isRiver; // Whether vertex is part of a river
        std::vector<bool> isLake; // Whether vertex is a lake
        
        // Initialize hydrological arrays
        void initialize(size_t vertexCount) {
            drainage.resize(vertexCount, 0.0f);
            riverFlow.resize(vertexCount, 0.0f);
            flowDirection.resize(vertexCount, -1);
            isRiver.resize(vertexCount, false);
            isLake.resize(vertexCount, false);
        }
        
        // Calculate drainage network
        void calculateDrainageNetwork(const Planet& planet, const SphericalGrid& grid);
    };
    
    // Mass balance tracker
    MassBalanceTracker massBalance;
    
    // Climate system
    ClimateSystem climate;
    // Hydrological system
    HydrologicalSystem hydrology;
    
    /**
     * @brief Get reference to climate system
     *
     * @return const ClimateSystem& Reference to climate system
     */
    const ClimateSystem& getClimateSystem() const {
        return climate;
    }
    
    /**
     * @brief Get reference to hydrological system
     *
     * @return const HydrologicalSystem& Reference to hydrological system
     */
    const HydrologicalSystem& getHydrologicalSystem() const {
        return hydrology;
    }
    
    
    // Helper methods
    
    /**
     * @brief Calculate the force on a plate from convection cells
     * 
     * @param plate Tectonic plate
     * @return Vec3f Resulting force vector
     */
    Vec3f calculateConvectionForce(const TectonicPlate& plate) const;
    
    /**
     * @brief Process interactions between plates
     */
    void processPlateInteractions();
    
    /**
     * @brief Calculate plate movement based on forces
     * 
     * @param timeStep Time increment in million years
     */
    void calculatePlateMovements(double timeStep);
    
    /**
     * @brief Apply erosion and sedimentation
     * 
     * @param timeStep Time increment in million years
     */
    void applyErosionAndSedimentation(double timeStep);
    
    /**
     * @brief Apply isostatic adjustments to the planet
     */
    void applyIsostaticAdjustments();
    
    /**
     * @brief Identify the closest plate for an unassigned vertex
     * 
     * @param vertexIdx Vertex index
     * @return int Plate ID or -1 if none found
     */
    int identifyClosestPlate(int vertexIdx) const;
    
    /**
     * @brief Apply forces based on supercontinent cycle phase
     * Modifies plate movements based on cycle phase:
     * - 0.0-0.3: Assembly phase - continents move toward each other
     * - 0.3-0.5: Supercontinent phase - continental movement reduced
     * - 0.5-1.0: Dispersal phase - continents move away from each other
     */
    void applySuperContinentCycleForces();
    
    /**
     * @brief Calculate hydrological processes (rivers, lakes, etc.)
     *
     * @param timeStep Time increment in million years
     */
    void calculateHydrologicalProcesses(double timeStep);
    
    /**
     * @brief Print statistics about continental growth
     * Calculates current volumes and growth percentages of continental and oceanic crust
     */
    void printContinentalGrowthStats();
    
    /**
     * @brief Initialize mass balance tracking
     * Calculate initial continental and oceanic crust volumes
     */
    void initializeMassBalance();
};

} // namespace aeonterracpp