#include "aeonterracpp/core/tectonics.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>

using namespace aeonterracpp;
using Catch::Matchers::WithinAbs;

TEST_CASE("TectonicPlate class", "[tectonics]") {
    SECTION("Constructor and basic properties") {
        Vec3f center(0.0f, 0.0f, 1.0f);
        TectonicPlate plate(5, center);
        
        REQUIRE(plate.getId() == 5);
        REQUIRE(plate.getCenter().isApprox(center));
        REQUIRE(plate.getMovement().isZero());
        REQUIRE_FALSE(plate.isOceanic());
    }
    
    SECTION("Movement") {
        TectonicPlate plate(1, Vec3f(1.0f, 0.0f, 0.0f));
        Vec3f movement(0.1f, 0.2f, 0.0f);
        
        plate.setMovement(movement);
        REQUIRE(plate.getMovement().isApprox(movement));
    }
    
    SECTION("Vertex management") {
        TectonicPlate plate(2, Vec3f(0.0f, 1.0f, 0.0f));
        
        // Add vertices
        plate.addVertex(10);
        plate.addVertex(20);
        plate.addVertex(30);
        
        const auto& vertices = plate.getVertices();
        REQUIRE(vertices.size() == 3);
        REQUIRE(vertices[0] == 10);
        REQUIRE(vertices[1] == 20);
        REQUIRE(vertices[2] == 30);
        
        // Calculate area (simple approximation)
        REQUIRE(plate.calculateArea() == 3.0f);
    }
    
    SECTION("Plate type") {
        TectonicPlate plate(3, Vec3f(0.0f, 0.0f, 1.0f));
        REQUIRE_FALSE(plate.isOceanic());
        
        plate.setOceanic(true);
        REQUIRE(plate.isOceanic());
    }
}

TEST_CASE("ConvectionCell class", "[tectonics]") {
    SECTION("Constructor and basic properties") {
        Vec3f center(0.0f, 0.0f, 1.0f);
        float strength = 0.5f;
        
        ConvectionCell cell(center, strength);
        
        REQUIRE(cell.getCenter().isApprox(center));
        REQUIRE(cell.getStrength() == strength);
    }
    
    SECTION("Force calculation") {
        Vec3f center(0.0f, 0.0f, 1.0f);
        float upwellingStrength = 0.5f;
        float downwellingStrength = -0.5f;
        
        // Upwelling cell pushes away from center
        ConvectionCell upwelling(center, upwellingStrength);
        
        // Position directly below center
        Vec3f posBelow(0.0f, 0.0f, 0.9f);
        Vec3f forceBelow = upwelling.calculateForceAt(posBelow);
        
        // Force should be downward (away from center)
        REQUIRE(forceBelow.z() < 0);
        
        // Position directly above center
        Vec3f posAbove(0.0f, 0.0f, 1.1f);
        Vec3f forceAbove = upwelling.calculateForceAt(posAbove);
        
        // Force should be upward (away from center)
        REQUIRE(forceAbove.z() > 0);
        
        // Downwelling cell pulls toward center
        ConvectionCell downwelling(center, downwellingStrength);
        
        // Position below center
        Vec3f forceBelow2 = downwelling.calculateForceAt(posBelow);
        
        // Force should be upward (toward center)
        REQUIRE(forceBelow2.z() > 0);
        
        // Position above center
        Vec3f forceAbove2 = downwelling.calculateForceAt(posAbove);
        
        // Force should be downward (toward center)
        REQUIRE(forceAbove2.z() < 0);
    }
}

TEST_CASE("TectonicSimulation", "[tectonics]") {
    SECTION("Initialization") {
        // Create a planet
        Planet planet(6371.0f, 2); // Small grid for testing
        
        // Create simulation
        TectonicSimulation sim(planet, 4, 42); // 4 plates, fixed seed
        
        // Check if plates were created
        const auto& plates = sim.getPlates();
        REQUIRE(plates.size() == 4);
        
        // Check if vertices were assigned to plates
        size_t assignedCount = 0;
        for (size_t i = 0; i < planet.getVertexCount(); ++i) {
            if (planet.getPlateId(static_cast<int>(i)) >= 0) {
                assignedCount++;
            }
        }
        
        // All vertices should be assigned
        REQUIRE(assignedCount == planet.getVertexCount());
        
        // Check plate properties
        for (const auto& plate : plates) {
            REQUIRE(plate.getVertices().size() > 0);
            REQUIRE(plate.getCenter().norm() > 0);
        }
    }
    
    SECTION("Plate boundaries") {
        // Create a planet with small grid
        Planet planet(6371.0f, 2);
        
        // Create simulation
        TectonicSimulation sim(planet, 4, 42);
        
        // Get boundary vertices
        auto boundaries = sim.getBoundaryVertices();
        
        // There should be some boundaries
        REQUIRE(boundaries.size() > 0);
        
        // Check if all boundary vertices are actually on boundaries
        for (int vertexIdx : boundaries) {
            int plateId = planet.getPlateId(vertexIdx);
            REQUIRE(plateId >= 0);
            
            // Get neighbors
            const auto& neighbors = planet.getGrid().getNeighbors(vertexIdx);
            
            // At least one neighbor should be in a different plate
            bool hasDifferentPlateNeighbor = false;
            for (int neighborIdx : neighbors) {
                int neighborPlateId = planet.getPlateId(neighborIdx);
                if (neighborPlateId >= 0 && neighborPlateId != plateId) {
                    hasDifferentPlateNeighbor = true;
                    break;
                }
            }
            
            REQUIRE(hasDifferentPlateNeighbor);
        }
    }
    
    SECTION("Isostatic adjustment") {
        // Create a planet
        Planet planet(6371.0f, 2);
        
        // Create simulation
        TectonicSimulation sim(planet, 2, 42);
        
        // Test a specific vertex
        int vertexIdx = 0;
        
        // Set specific values to test isostatic equilibrium
        planet.setCrustType(vertexIdx, 1); // Continental
        planet.setCrustThickness(vertexIdx, 40.0f); // Thick crust
        planet.setCrustDensity(vertexIdx, 2.7f); // Continental density
        planet.setElevation(vertexIdx, 0.0f); // Sea level
        
        // Calculate isostatic adjustment
        float adjustment = sim.calculateIsostaticAdjustment(vertexIdx);
        
        // Thicker continental crust should result in positive adjustment (uplift)
        REQUIRE(adjustment > 0.0f);
        
        // Change to oceanic
        planet.setCrustType(vertexIdx, 0); // Oceanic
        planet.setCrustThickness(vertexIdx, 7.0f); // Thin crust
        planet.setCrustDensity(vertexIdx, 3.0f); // Oceanic density
        planet.setElevation(vertexIdx, 0.0f); // Sea level
        
        // Calculate isostatic adjustment
        adjustment = sim.calculateIsostaticAdjustment(vertexIdx);
        
        // Thinner oceanic crust should result in negative adjustment (subsidence)
        REQUIRE(adjustment < 0.0f);
    }
    
    SECTION("Simulation step") {
        // Create a planet
        Planet planet(6371.0f, 2);
        
        // Create simulation
        TectonicSimulation sim(planet, 4, 42);
        
        // Initial state
        float initialAge = planet.getAge();
        
        // Store initial values for a vertex to check changes
        int testVertex = 0;
        float initialElevation = planet.getElevation(testVertex);
        float initialThickness = planet.getCrustThickness(testVertex);
        
        // Run simulation step
        sim.simulateStep(10.0); // 10 million years
        
        // Age should be advanced
        REQUIRE(planet.getAge() == initialAge + 10.0);
        
        // Some properties should change
        bool elevationChanged = planet.getElevation(testVertex) != initialElevation;
        bool thicknessChanged = planet.getCrustThickness(testVertex) != initialThickness;
        // At least one property should change
        REQUIRE((elevationChanged || thicknessChanged));
        REQUIRE(elevationChanged || thicknessChanged);
    }
    
    SECTION("Interaction types") {
        // Create a planet
        Planet planet(6371.0f, 2);
        TectonicSimulation sim(planet, 2, 42); // Just 2 plates for simplicity
        
        // Find a boundary vertex
        auto boundaries = sim.getBoundaryVertices();
        REQUIRE(!boundaries.empty());
        
        int boundaryVertex = boundaries[0];
        int plateId = planet.getPlateId(boundaryVertex);
        
        // Find a neighbor in a different plate
        const auto& neighbors = planet.getGrid().getNeighbors(boundaryVertex);
        int neighborVertex = -1;
        
        for (int neighbor : neighbors) {
            int neighborPlate = planet.getPlateId(neighbor);
            if (neighborPlate >= 0 && neighborPlate != plateId) {
                neighborVertex = neighbor;
                break;
            }
        }
        
        REQUIRE(neighborVertex >= 0);
        
        // Get interaction type
        PlateInteractionType interaction = sim.getInteractionType(boundaryVertex, neighborVertex);
        
        // Should be one of the valid types
        REQUIRE(interaction != PlateInteractionType::None);
    }
}