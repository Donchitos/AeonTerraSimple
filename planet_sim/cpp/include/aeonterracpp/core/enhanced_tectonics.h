#pragma once

#include "aeonterracpp/core/enhanced_planet.h"
#include "aeonterracpp/core/tectonics.h"
#include "aeonterracpp/core/vector_types.h"
#include <memory>
#include <random>
#include <vector>

namespace aeonterracpp {

// Forward declaration of SupercontinentCycle
class SupercontinentCycle;

/**
 * @brief Enhanced tectonic simulation with improved physical models
 * 
 * This class extends the base TectonicSimulation with more realistic
 * physical models for plate interactions, mantle dynamics, and 
 * lithospheric processes.
 */
class EnhancedTectonicSimulation : public TectonicSimulation {
public:
    /**
     * @brief Construct a new Enhanced Tectonic Simulation
     * 
     * @param planet Reference to the enhanced planet
     * @param numPlates Number of initial plates to create
     * @param seed Random seed for reproducibility (0 for random)
     */
    EnhancedTectonicSimulation(EnhancedPlanet& planet, int numPlates = 7, unsigned int seed = 0);
    
    /**
     * @brief Destructor
     */
    ~EnhancedTectonicSimulation();
    
    /**
     * @brief Simulate a single time step with enhanced physics
     * 
     * @param timeStep Time increment in million years
     */
    void simulateStep(double timeStep) override;
    
    /**
     * @brief Get detailed interaction type between adjacent plates
     * Expands on the base class with more specific interaction types
     * 
     * @param v1 First vertex index
     * @param v2 Second vertex index
     * @return PlateInteractionType Type of interaction
     */
    PlateInteractionType getInteractionType(int v1, int v2) const override;
    
    /**
     * @brief Calculate detailed isostatic adjustment for a vertex
     * Includes flexural effects and multi-layer density model
     * 
     * @param vertexIdx Vertex index
     * @return float Height adjustment in km
     */
    float calculateIsostaticAdjustment(int vertexIdx) const override;
    
    /**
     * @brief Add a mantle plume at a specified location
     * 
     * @param position Position on the planet surface
     * @param strength Plume strength (0.0-2.0)
     * @param radius Influence radius in km
     * @return int Plume ID
     */
    int addMantlePlume(const Vec3f& position, float strength = 1.0f, float radius = 200.0f);
    
    /**
     * @brief Generate random mantle plumes
     * 
     * @param count Number of plumes to create
     */
    void generateRandomMantlePlumes(int count);
    
    /**
     * @brief Update plume dynamics and effects on lithosphere
     * 
     * @param timeStep Time increment in million years
     */
    void updatePlumeDynamics(double timeStep);
    
    /**
     * @brief Calculate dynamic topography from mantle flow
     */
    void calculateDynamicTopography();
    
    /**
     * @brief Enable or disable supercontinent cycle forces
     * 
     * @param enable Whether to enable supercontinent cycle
     */
    void enableSupercontinentCycle(bool enable);
    
    /**
     * @brief Calculate slab pull force for a plate at a boundary
     * 
     * @param plateIdx Plate index
     * @param boundaryVertexIdx Boundary vertex index
     * @return float Slab pull force magnitude
     */
    float calculateSlabPull(int plateIdx, int boundaryVertexIdx) const;
    
    /**
     * @brief Calculate flexural adjustment for a load
     * 
     * @param vertexIdx Central vertex index
     * @param load Load magnitude in km
     * @return float Flexural adjustment in km
     */
    float calculateFlexuralAdjustment(int vertexIdx, float load) const;
    
    /**
     * @brief Create a large igneous province at a mantle plume
     * 
     * @param plumeIdx Plume index
     */
    void createLargeIgneousProvince(int plumeIdx);
    
    /**
     * @brief Process a transform boundary with enhanced physics
     * 
     * @param v1 First vertex index
     * @param v2 Second vertex index
     */
    void processTransformBoundary(int v1, int v2);
    
    /**
     * @brief Calculate the tangent vector along a plate boundary
     * 
     * @param v1 First vertex index
     * @param v2 Second vertex index
     * @return Vec3f Tangent vector along boundary
     */
    Vec3f calculateBoundaryTangent(int v1, int v2) const;
    
    /**
     * @brief Calculate the bend angle along a plate boundary
     * 
     * @param v1 First vertex index
     * @param v2 Second vertex index
     * @return float Bend angle in degrees
     */
    float calculateBoundaryBendAngle(int v1, int v2) const;
    
    /**
     * @brief Calculate relative motion along a boundary
     * 
     * @param v1 First vertex index
     * @param v2 Second vertex index
     * @param tangent Tangent vector along boundary
     * @return float Relative motion rate
     */
    float calculateRelativeMotion(int v1, int v2, const Vec3f& tangent) const;
    
    /**
     * @brief Create a rift zone at a specified location
     * 
     * @param vertexIdx Center vertex of the rift
     * @param plateIdx Plate ID
     */
    void createRiftZone(int vertexIdx, int plateIdx);
    
private:
    // Reference to enhanced planet
    EnhancedPlanet& enhancedPlanet;
    
    // Mantle plume system
    struct MantlePlume {
        Vec3f position;        // Position on sphere
        float strength;        // Plume strength (0.0-2.0)
        float radius;          // Influence radius in km
        float age;             // Age in million years
        bool active;           // Whether the plume is active
        std::vector<int> affectedVertices; // Vertices influenced by this plume
        
        MantlePlume(const Vec3f& pos, float str, float rad)
            : position(pos), strength(str), radius(rad), age(0.0f), active(true) {}
    };
    
    std::vector<MantlePlume> mantlePlumes;
    
    // Supercontinent cycle controller
    std::unique_ptr<SupercontinentCycle> supercontinentCycle;
    bool supercontinentCycleEnabled;
    
    // Enhanced parameters
    struct EnhancedParameters {
        // Lithosphere properties
        float mantleViscosity;           // Mantle viscosity (Pa·s)
        float lithosphereYoungsModulus;  // Young's modulus (GPa)
        float lithospherePoissonsRatio;  // Poisson's ratio
        
        // Slab dynamics
        float slabDensityIncrease;       // Density increase of subducting slab (g/cm³)
        float slabPullEfficiency;        // Efficiency of slab pull (0.0-1.0)
        
        // Flexural isostasy
        float flexuralRigidityScale;     // Scaling factor for flexural rigidity
        
        // Plume dynamics
        float plumeTemperature;          // Temperature of mantle plumes (°C)
        float backgroundMantleTemp;      // Background mantle temperature (°C)
        float plumeBuoyancy;             // Plume buoyancy factor
        
        // Rift mechanics
        float riftWeakening;             // Lithospheric weakening at rifts (0.0-1.0)
        float riftMagmaSupply;           // Magma supply rate at rifts (km³/Myr)
    };
    
    EnhancedParameters enhancedParams;
    
    // Override methods from base class with enhanced versions
    void processPlateInteractions() override;
    void calculatePlateMovements(double timeStep) override;
    void applyErosionAndSedimentation(double timeStep) override;
    void applyIsostaticAdjustments() override;
    
    // Enhanced methods
    void enhanceCollisionBoundary(int v1, int v2);
    void processSubductionZone(int oceanicVertex, int continentalVertex);
    void simulateSupercontinentBreakup();
    void updateMantlePlumes(double timeStep);
    Vec3f generateRandomSpherePoint();
    void initializeEnhancedParameters();
};

} // namespace aeonterracpp