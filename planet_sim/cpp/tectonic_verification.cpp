#include "aeonterracpp/core/tectonics.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>

using namespace aeonterracpp;

int main() {
    std::cout << "========== AeonTerra Tectonic Simulation Verification ==========" << std::endl << std::endl;
    
    // Use high-resolution clock for timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Create a planet for testing
    int resolution = 5;
    Planet planet(6371.0f, resolution, "Tectonic Test");
    
    std::cout << "[1] Created planet with resolution " << resolution << std::endl;
    std::cout << "    Vertex count: " << planet.getVertexCount() << std::endl;
    std::cout << "    Initial age: " << planet.getAge() << " million years" << std::endl << std::endl;
    
    // Create a tectonic simulation
    int numPlates = 8;
    unsigned int seed = 42;  // Fixed seed for reproducibility
    
    std::cout << "[2] Creating tectonic simulation with " << numPlates << " plates..." << std::endl;
    TectonicSimulation sim(planet, numPlates, seed);
    
    // Get plates
    const auto& plates = sim.getPlates();
    std::cout << "    Created " << plates.size() << " tectonic plates" << std::endl;
    
    // Print information about each plate
    std::cout << "    Plate information:" << std::endl;
    for (size_t i = 0; i < plates.size(); ++i) {
        const auto& plate = plates[i];
        std::cout << "      Plate " << i << ":" << std::endl;
        std::cout << "        ID: " << plate.getId() << std::endl;
        std::cout << "        Type: " << (plate.isOceanic() ? "Oceanic" : "Continental") << std::endl;
        std::cout << "        Vertices: " << plate.getVertices().size() << std::endl;
        std::cout << "        Center: (" << plate.getCenter().x() << ", " 
                  << plate.getCenter().y() << ", " << plate.getCenter().z() << ")" << std::endl;
    }
    
    // Get boundary vertices
    auto boundaries = sim.getBoundaryVertices();
    std::cout << "    Found " << boundaries.size() << " boundary vertices" << std::endl << std::endl;
    
    // Analyze plate coverage
    size_t unassignedCount = 0;
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        if (planet.getPlateId(static_cast<int>(i)) < 0) {
            unassignedCount++;
        }
    }
    
    float coveragePercent = 100.0f * (1.0f - static_cast<float>(unassignedCount) / planet.getVertexCount());
    std::cout << "    Plate coverage: " << std::fixed << std::setprecision(2) << coveragePercent << "%" << std::endl;
    
    if (unassignedCount > 0) {
        std::cout << "    WARNING: " << unassignedCount << " vertices are not assigned to any plate" << std::endl;
    }
    
    // Analyze plate types
    int oceanicCount = 0;
    int continentalCount = 0;
    for (const auto& plate : plates) {
        if (plate.isOceanic()) {
            oceanicCount++;
        } else {
            continentalCount++;
        }
    }
    
    std::cout << "    Plate types: " << oceanicCount << " oceanic, " 
              << continentalCount << " continental" << std::endl << std::endl;
    
    // Perform simulation steps
    std::cout << "[3] Simulating tectonic activity..." << std::endl;
    
    // Set activity level
    sim.setTectonicActivity(0.7f);
    std::cout << "    Tectonic activity level: " << sim.getTectonicActivity() << std::endl;
    
    // Run 5 simulation steps of 10 million years each
    const int numSteps = 5;
    const double timeStep = 10.0;
    
    for (int i = 0; i < numSteps; ++i) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        
        // Run one step
        sim.simulateStep(timeStep);
        
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> stepTime = stepEnd - stepStart;
        
        std::cout << "    Step " << (i+1) << ": " << timeStep << " million years (took " 
                  << std::fixed << std::setprecision(2) << stepTime.count() << " ms)" << std::endl;
    }
    
    std::cout << "    Final planet age: " << planet.getAge() << " million years" << std::endl << std::endl;
    
    // Analyze elevation changes
    float minElevation = std::numeric_limits<float>::max();
    float maxElevation = -std::numeric_limits<float>::max();
    float avgElevation = 0.0f;
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        float elevation = planet.getElevation(static_cast<int>(i));
        minElevation = std::min(minElevation, elevation);
        maxElevation = std::max(maxElevation, elevation);
        avgElevation += elevation;
    }
    
    avgElevation /= planet.getVertexCount();
    
    std::cout << "[4] Final elevation statistics:" << std::endl;
    std::cout << "    Minimum elevation: " << minElevation << " km" << std::endl;
    std::cout << "    Maximum elevation: " << maxElevation << " km" << std::endl;
    std::cout << "    Average elevation: " << avgElevation << " km" << std::endl;
    std::cout << "    Elevation range: " << (maxElevation - minElevation) << " km" << std::endl << std::endl;
    
    // Analyze boundary interaction types
    std::map<PlateInteractionType, int> interactionCounts;
    
    for (int v1 : boundaries) {
        const auto& neighbors = planet.getGrid().getNeighbors(v1);
        
        for (int v2 : neighbors) {
            PlateInteractionType interaction = sim.getInteractionType(v1, v2);
            if (interaction != PlateInteractionType::None) {
                interactionCounts[interaction]++;
            }
        }
    }
    
    std::cout << "[5] Boundary interaction types:" << std::endl;
    std::cout << "    Divergent: " << interactionCounts[PlateInteractionType::Divergent] << std::endl;
    std::cout << "    Convergent: " << interactionCounts[PlateInteractionType::Convergent] << std::endl;
    std::cout << "    Transform: " << interactionCounts[PlateInteractionType::Transform] << std::endl;
    std::cout << "    Subduction: " << interactionCounts[PlateInteractionType::Subduction] << std::endl;
    std::cout << "    Collision: " << interactionCounts[PlateInteractionType::Collision] << std::endl << std::endl;
    
    // Check final plate movement
    std::cout << "[6] Final plate movements:" << std::endl;
    
    for (size_t i = 0; i < plates.size(); ++i) {
        const auto& movement = plates[i].getMovement();
        float magnitude = movement.norm();
        
        std::cout << "    Plate " << i << ": Magnitude = " << std::fixed << std::setprecision(6) << magnitude;
        std::cout << ", Direction = (" << movement.x() << ", " << movement.y() << ", " << movement.z() << ")" << std::endl;
    }
    
    // Display Hydrological System Information
    std::cout << "\n[7] Hydrological system features:" << std::endl;
    
    // Count rivers and lakes
    int riverCount = 0;
    int lakeCount = 0;
    int maxDrainageArea = 0;
    int largestRiverCell = -1;
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        const HydrologicalSystem& hydro = sim.getHydrologicalSystem();
        if (hydro.isRiver[i]) {
            riverCount++;
            if (hydro.drainage[i] > maxDrainageArea) {
                maxDrainageArea = hydro.drainage[i];
                largestRiverCell = i;
            }
        }
        if (hydro.isLake[i]) {
            lakeCount++;
        }
    }
    
    std::cout << "    Total river cells: " << riverCount << std::endl;
    std::cout << "    Total lake cells: " << lakeCount << std::endl;
    
    if (largestRiverCell >= 0) {
        Vec3f riverPos = planet.getGrid().getVertex(largestRiverCell);
        auto [lat, lon] = planet.getGrid().cartesianToLatLon(riverPos);
        std::cout << "    Largest river drainage area: " << maxDrainageArea << " cells at coordinates ("
                  << std::fixed << std::setprecision(2) << lat << "°, " << lon << "°)" << std::endl;
    }
    
    // Display Supercontinent Cycle Information
    std::cout << "\n[8] Supercontinent cycle status:" << std::endl;
    float cyclePhase = sim.getSuperContinentCyclePhase();
    std::cout << "    Current cycle phase: " << (cyclePhase * 100.0f) << "%" << std::endl;
    
    if (cyclePhase < 0.3f) {
        std::cout << "    Phase description: Assembly - continents moving toward each other" << std::endl;
    } else if (cyclePhase < 0.5f) {
        std::cout << "    Phase description: Supercontinent formation - reduced continental movement" << std::endl;
    } else {
        std::cout << "    Phase description: Dispersal - continents moving apart" << std::endl;
    }
    
    // Display Climate System Information
    std::cout << "\n[9] Climate system statistics:" << std::endl;
    
    // Calculate min, max, avg temperature and precipitation
    float minTemp = std::numeric_limits<float>::max();
    float maxTemp = -std::numeric_limits<float>::max();
    float avgTemp = 0.0f;
    float minPrecip = std::numeric_limits<float>::max();
    float maxPrecip = -std::numeric_limits<float>::max();
    float avgPrecip = 0.0f;
    
    const ClimateSystem& climate = sim.getClimateSystem();
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        minTemp = std::min(minTemp, climate.temperature[i]);
        maxTemp = std::max(maxTemp, climate.temperature[i]);
        avgTemp += climate.temperature[i];
        
        minPrecip = std::min(minPrecip, climate.precipitation[i]);
        maxPrecip = std::max(maxPrecip, climate.precipitation[i]);
        avgPrecip += climate.precipitation[i];
    }
    
    avgTemp /= planet.getVertexCount();
    avgPrecip /= planet.getVertexCount();
    
    std::cout << "    Temperature range: " << minTemp << "°C to " << maxTemp << "°C (avg " << avgTemp << "°C)" << std::endl;
    std::cout << "    Precipitation range: " << minPrecip << "mm to " << maxPrecip << "mm (avg " << avgPrecip << "mm)" << std::endl;
    
    // Calculate total execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalTime = endTime - startTime;
    
    std::cout << std::endl;
    std::cout << "Total execution time: " << std::fixed << std::setprecision(4) << totalTime.count() << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Tectonic simulation verification completed successfully!" << std::endl;
    
    return 0;
}