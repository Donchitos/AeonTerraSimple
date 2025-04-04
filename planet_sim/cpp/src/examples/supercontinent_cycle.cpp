#include "aeonterracpp/core/tectonics.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

namespace aeonterracpp {

/**
 * @brief SupercontinentCycle implements the Wilson Cycle for tectonic simulation
 * This models the 500+ million year process where continents assemble into a supercontinent
 * and then break apart again, as seen with Pangea, Rodinia, and other supercontinents in Earth's history.
 */
class SupercontinentCycle {
public:
    /**
     * @brief Phase of the supercontinent cycle
     */
    enum class Phase {
        DISPERSAL,       // Continents moving apart (like post-Pangea)
        DRIFTING,        // Continents moving independently
        CONVERGENCE,     // Continents starting to gather
        ASSEMBLY,        // Formation of supercontinent
        STABILITY        // Stable supercontinent phase (like Pangea)
    };
    
    /**
     * @brief Construct a new Supercontinent Cycle object
     * 
     * @param tectonics The tectonic simulation to influence
     */
    SupercontinentCycle(TectonicSimulation& tectonics) 
        : tectonics(tectonics), 
          currentPhase(Phase::DISPERSAL),
          cycleAge(0.0),
          cycleProgress(0.0),
          cycleDuration(500.0) { // Full cycle takes ~500 million years
    }
    
    /**
     * @brief Update the cycle state based on simulation time
     * 
     * @param timeStep Time step in million years
     */
    void update(double timeStep) {
        // Update cycle age
        cycleAge += timeStep;
        
        // Reset if we've completed a full cycle
        if (cycleAge > cycleDuration) {
            cycleAge = std::fmod(cycleAge, cycleDuration);
            std::cout << "Completed a full supercontinent cycle!" << std::endl;
        }
        
        // Calculate cycle progress and determine current phase
        cycleProgress = cycleAge / cycleDuration;
        updateCurrentPhase();
        
        // Apply the appropriate forces for the current phase
        influenceTectonics();
    }
    
    /**
     * @brief Get the current phase of the supercontinent cycle
     * 
     * @return Phase Current phase
     */
    Phase getCurrentPhase() const {
        return currentPhase;
    }
    
    /**
     * @brief Get a string description of the current phase
     * 
     * @return std::string Phase description
     */
    std::string getPhaseDescription() const {
        switch (currentPhase) {
            case Phase::DISPERSAL:
                return "Dispersal - Continents moving apart";
            case Phase::DRIFTING:
                return "Drifting - Continents moving independently";
            case Phase::CONVERGENCE:
                return "Convergence - Continents beginning to gather";
            case Phase::ASSEMBLY:
                return "Assembly - Supercontinent forming";
            case Phase::STABILITY:
                return "Stability - Stable supercontinent";
            default:
                return "Unknown phase";
        }
    }
    
    /**
     * @brief Influence plate tectonics based on the current cycle phase
     * This modifies plate movements to create realistic supercontinent formation and breakup
     */
    void influenceTectonics() {
        // Get all plates
        const auto& plates = tectonics.getPlates();
        
        // Identify continental plates
        std::vector<int> continentalPlateIndices;
        for (size_t i = 0; i < plates.size(); ++i) {
            if (!plates[i].isOceanic()) {
                continentalPlateIndices.push_back(static_cast<int>(i));
            }
        }
        
        // Apply different forces based on current phase
        switch (currentPhase) {
            case Phase::DISPERSAL:
                applyDispersalForces(continentalPlateIndices);
                break;
            case Phase::DRIFTING:
                applyDriftingForces(continentalPlateIndices);
                break;
            case Phase::CONVERGENCE:
                applyConvergenceForces(continentalPlateIndices);
                break;
            case Phase::ASSEMBLY:
                applyAssemblyForces(continentalPlateIndices);
                break;
            case Phase::STABILITY:
                applyStabilityForces(continentalPlateIndices);
                break;
        }
    }
    
private:
    TectonicSimulation& tectonics;
    Phase currentPhase;
    double cycleAge;        // Age within the current cycle (million years)
    double cycleProgress;   // Progress through cycle (0-1)
    double cycleDuration;   // Duration of a full cycle (million years)
    
    /**
     * @brief Update the current phase based on cycle progress
     */
    void updateCurrentPhase() {
        // Phase transitions based on cycle progress
        if (cycleProgress < 0.2) {
            currentPhase = Phase::DISPERSAL;
        } else if (cycleProgress < 0.4) {
            currentPhase = Phase::DRIFTING;
        } else if (cycleProgress < 0.6) {
            currentPhase = Phase::CONVERGENCE;
        } else if (cycleProgress < 0.8) {
            currentPhase = Phase::ASSEMBLY;
        } else {
            currentPhase = Phase::STABILITY;
        }
    }
    
    /**
     * @brief Calculate the centroid of continental plates
     * 
     * @return Vec3f Centroid position
     */
    Vec3f calculateContinentalCentroid(const std::vector<int>& continentalPlateIndices) const {
        Vec3f centroid = Vec3f::Zero();
        
        if (continentalPlateIndices.empty()) {
            return centroid;
        }
        
        const auto& plates = tectonics.getPlates();
        for (int idx : continentalPlateIndices) {
            centroid += plates[idx].getCenter();
        }
        
        return centroid / continentalPlateIndices.size();
    }
    
    /**
     * @brief Apply forces for the dispersal phase (breaking up supercontinent)
     */
    void applyDispersalForces(const std::vector<int>& continentalPlateIndices) {
        // Calculate continental centroid
        Vec3f centroid = calculateContinentalCentroid(continentalPlateIndices);
        float forceMagnitude = 0.4f;
        
        // Push continents away from centroid (breaking up the supercontinent)
        auto& plates = const_cast<std::vector<TectonicPlate>&>(tectonics.getPlates());
        for (int idx : continentalPlateIndices) {
            Vec3f platePos = plates[idx].getCenter();
            Vec3f direction = (platePos - centroid).normalized();
            Vec3f force = direction * forceMagnitude;
            
            // Add dispersal force to current movement
            Vec3f currentMovement = plates[idx].getMovement();
            plates[idx].setMovement(currentMovement + force);
        }
        
        std::cout << "Applied dispersal forces to " << continentalPlateIndices.size() 
                  << " continental plates" << std::endl;
    }
    
    /**
     * @brief Apply forces for the drifting phase (random movement)
     */
    void applyDriftingForces(const std::vector<int>& continentalPlateIndices) {
        // In drifting phase, apply slight random perturbations to plates
        float forceMagnitude = 0.2f;
        
        auto& plates = const_cast<std::vector<TectonicPlate>&>(tectonics.getPlates());
        for (int idx : continentalPlateIndices) {
            // Generate random direction on surface of sphere
            Vec3f randomDir = Vec3f::Random().normalized();
            // Project onto tangent plane at plate center
            Vec3f plateCenter = plates[idx].getCenter().normalized();
            float dot = randomDir.dot(plateCenter);
            Vec3f tangentDir = (randomDir - plateCenter * dot).normalized();
            
            Vec3f force = tangentDir * forceMagnitude;
            
            // Add drifting force to current movement
            Vec3f currentMovement = plates[idx].getMovement();
            plates[idx].setMovement(currentMovement + force);
        }
        
        std::cout << "Applied drifting forces to " << continentalPlateIndices.size() 
                  << " continental plates" << std::endl;
    }
    
    /**
     * @brief Apply forces for the convergence phase (beginning to gather)
     */
    void applyConvergenceForces(const std::vector<int>& continentalPlateIndices) {
        // Divide continents into two groups moving toward each other
        Vec3f convergencePoint = Vec3f::Zero(); // Center of planet
        float forceMagnitude = 0.3f;
        
        auto& plates = const_cast<std::vector<TectonicPlate>&>(tectonics.getPlates());
        for (int idx : continentalPlateIndices) {
            Vec3f platePos = plates[idx].getCenter();
            
            // Move all continents toward center (will later form supercontinent)
            Vec3f direction = (convergencePoint - platePos).normalized();
            Vec3f force = direction * forceMagnitude;
            
            // Add convergence force to current movement
            Vec3f currentMovement = plates[idx].getMovement();
            plates[idx].setMovement(currentMovement + force);
        }
        
        std::cout << "Applied convergence forces to " << continentalPlateIndices.size() 
                  << " continental plates" << std::endl;
    }
    
    /**
     * @brief Apply forces for the assembly phase (forming supercontinent)
     */
    void applyAssemblyForces(const std::vector<int>& continentalPlateIndices) {
        // Similar to convergence but stronger and more focused
        Vec3f assemblyPoint = Vec3f::Zero(); // Center of planet
        float forceMagnitude = 0.5f;
        
        auto& plates = const_cast<std::vector<TectonicPlate>&>(tectonics.getPlates());
        for (int idx : continentalPlateIndices) {
            Vec3f platePos = plates[idx].getCenter();
            
            // Strong movement toward assembly point
            Vec3f direction = (assemblyPoint - platePos).normalized();
            Vec3f force = direction * forceMagnitude;
            
            // Add assembly force to current movement
            Vec3f currentMovement = plates[idx].getMovement();
            plates[idx].setMovement(currentMovement + force);
        }
        
        std::cout << "Applied assembly forces to " << continentalPlateIndices.size() 
                  << " continental plates" << std::endl;
    }
    
    /**
     * @brief Apply forces for the stability phase (maintaining supercontinent)
     */
    void applyStabilityForces(const std::vector<int>& continentalPlateIndices) {
        // In stability phase, continental plates move very little
        float dampingFactor = 0.8f;
        
        auto& plates = const_cast<std::vector<TectonicPlate>&>(tectonics.getPlates());
        for (int idx : continentalPlateIndices) {
            // Dampen existing movement
            Vec3f currentMovement = plates[idx].getMovement();
            plates[idx].setMovement(currentMovement * (1.0f - dampingFactor));
        }
        
        std::cout << "Applied stability forces to " << continentalPlateIndices.size() 
                  << " continental plates" << std::endl;
    }
};

// Example usage function
void runSupercontinentExample() {
    // Create a planet with resolution level 5 (10,242 vertices)
    Planet planet(6371.0f, 5, "Earth");
    
    // Create tectonic simulation with 7 plates
    TectonicSimulation tectonics(planet, 7);
    
    // Create supercontinent cycle controller
    SupercontinentCycle cycle(tectonics);
    
    // Run simulation for 500 million years (full cycle)
    double timeStep = 10.0; // 10 million years per step
    int steps = 50;         // 50 steps = 500 million years
    
    std::cout << "Starting supercontinent cycle simulation over " 
              << (timeStep * steps) << " million years..." << std::endl;
    
    for (int i = 0; i < steps; ++i) {
        // Update cycle state and influence tectonics
        cycle.update(timeStep);
        
        // Run tectonic simulation step
        tectonics.simulateStep(timeStep);
        
        // Report current status
        double age = planet.getAge();
        std::cout << "Age: " << age << " Myr, Phase: " << cycle.getPhaseDescription() << std::endl;
        
        // Every 100 million years, report continent positions
        if (i % 10 == 0) {
            const auto& plates = tectonics.getPlates();
            std::cout << "Continental positions:" << std::endl;
            for (size_t j = 0; j < plates.size(); ++j) {
                if (!plates[j].isOceanic()) {
                    const Vec3f& pos = plates[j].getCenter();
                    std::cout << "  Continent " << j << ": (" 
                              << pos.x() << ", " << pos.y() << ", " << pos.z() << ")" << std::endl;
                }
            }
        }
    }
    
    std::cout << "Supercontinent cycle simulation complete!" << std::endl;
}

} // namespace aeonterracpp

// Main function for standalone testing
int main() {
    aeonterracpp::runSupercontinentExample();
    return 0;
}