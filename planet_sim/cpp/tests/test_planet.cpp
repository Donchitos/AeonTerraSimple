#include "aeonterracpp/core/planet.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace aeonterracpp;

TEST_CASE("Planet constructor", "[planet]") {
    SECTION("Default constructor") {
        Planet planet;
        REQUIRE(planet.getRadius() == Approx(6371.0f));
        REQUIRE(planet.getName() == "AeonTerra");
        REQUIRE(planet.getAge() == Approx(0.0));
        REQUIRE(planet.getHistory().size() == 1); // Initial snapshot
    }
    
    SECTION("Constructor with parameters") {
        Planet planet(6000.0f, 3, "TestPlanet");
        REQUIRE(planet.getRadius() == Approx(6000.0f));
        REQUIRE(planet.getName() == "TestPlanet");
        REQUIRE(planet.getAge() == Approx(0.0));
    }
}

TEST_CASE("Planet property access", "[planet]") {
    Planet planet(6371.0f, 3);
    size_t vertexCount = planet.getVertexCount();
    
    REQUIRE(vertexCount > 0);
    
    SECTION("Get/set elevation") {
        // Test a valid vertex index
        planet.setElevation(0, 2.5f);
        REQUIRE(planet.getElevation(0) == Approx(2.5f));
        
        // Test out of range index
        REQUIRE_THROWS_AS(planet.setElevation(-1, 1.0f), std::out_of_range);
        REQUIRE_THROWS_AS(planet.setElevation(vertexCount, 1.0f), std::out_of_range);
        REQUIRE_THROWS_AS(planet.getElevation(-1), std::out_of_range);
        REQUIRE_THROWS_AS(planet.getElevation(vertexCount), std::out_of_range);
    }
    
    SECTION("Get/set crust thickness") {
        planet.setCrustThickness(0, 30.0f);
        REQUIRE(planet.getCrustThickness(0) == Approx(30.0f));
        
        REQUIRE_THROWS_AS(planet.setCrustThickness(-1, 1.0f), std::out_of_range);
        REQUIRE_THROWS_AS(planet.getCrustThickness(-1), std::out_of_range);
    }
    
    SECTION("Get/set crust density") {
        planet.setCrustDensity(0, 2.8f);
        REQUIRE(planet.getCrustDensity(0) == Approx(2.8f));
        
        REQUIRE_THROWS_AS(planet.setCrustDensity(-1, 1.0f), std::out_of_range);
        REQUIRE_THROWS_AS(planet.getCrustDensity(-1), std::out_of_range);
    }
    
    SECTION("Get/set crust type") {
        planet.setCrustType(0, 1);
        REQUIRE(planet.getCrustType(0) == 1);
        
        REQUIRE_THROWS_AS(planet.setCrustType(-1, 1), std::out_of_range);
        REQUIRE_THROWS_AS(planet.getCrustType(-1), std::out_of_range);
    }
    
    SECTION("Get/set plate ID") {
        planet.setPlateId(0, 5);
        REQUIRE(planet.getPlateId(0) == 5);
        
        REQUIRE_THROWS_AS(planet.setPlateId(-1, 1), std::out_of_range);
        REQUIRE_THROWS_AS(planet.getPlateId(-1), std::out_of_range);
    }
    
    SECTION("Grid access") {
        const SphericalGrid& grid = planet.getGrid();
        REQUIRE(grid.getRadius() == Approx(planet.getRadius()));
        REQUIRE(grid.getVertexCount() == planet.getVertexCount());
    }
}

TEST_CASE("Planet simulation", "[planet]") {
    Planet planet(6371.0f, 2);
    
    SECTION("Age advancement") {
        REQUIRE(planet.getAge() == Approx(0.0));
        
        planet.advanceAge(10.0);
        REQUIRE(planet.getAge() == Approx(10.0));
        
        planet.advanceAge(5.5);
        REQUIRE(planet.getAge() == Approx(15.5));
    }
    
    SECTION("History recording") {
        // Initial state should have one snapshot
        REQUIRE(planet.getHistory().size() == 1);
        
        // Advance and record
        planet.advanceAge(10.0);
        planet.recordSnapshot();
        REQUIRE(planet.getHistory().size() == 2);
        REQUIRE(planet.getHistory()[1].age == Approx(10.0));
        
        // Advance and record again
        planet.advanceAge(5.0);
        planet.recordSnapshot();
        REQUIRE(planet.getHistory().size() == 3);
        REQUIRE(planet.getHistory()[2].age == Approx(15.0));
        
        // Clear history
        planet.clearHistory();
        REQUIRE(planet.getHistory().size() == 0);
    }
}

TEST_CASE("Planet serialization", "[planet]") {
    std::string testFile = "test_planet.bin";
    
    SECTION("Save and load") {
        // Create a planet with some non-default values
        Planet original(6000.0f, 2, "TestPlanet");
        original.advanceAge(25.0);
        original.setElevation(0, 3.5f);
        original.setCrustThickness(0, 28.0f);
        original.setCrustDensity(0, 2.9f);
        original.setCrustType(0, 1);
        original.setPlateId(0, 3);
        
        // Record a few snapshots
        original.recordSnapshot();
        original.advanceAge(10.0);
        original.recordSnapshot();
        
        // Save to file
        original.saveToFile(testFile);
        
        // Load from file
        Planet loaded = Planet::loadFromFile(testFile);
        
        // Verify core properties
        REQUIRE(loaded.getRadius() == Approx(original.getRadius()));
        REQUIRE(loaded.getName() == original.getName());
        REQUIRE(loaded.getAge() == Approx(original.getAge()));
        REQUIRE(loaded.getVertexCount() == original.getVertexCount());
        
        // Verify property values
        REQUIRE(loaded.getElevation(0) == Approx(original.getElevation(0)));
        REQUIRE(loaded.getCrustThickness(0) == Approx(original.getCrustThickness(0)));
        REQUIRE(loaded.getCrustDensity(0) == Approx(original.getCrustDensity(0)));
        REQUIRE(loaded.getCrustType(0) == original.getCrustType(0));
        REQUIRE(loaded.getPlateId(0) == original.getPlateId(0));
        
        // Verify history
        REQUIRE(loaded.getHistory().size() == original.getHistory().size());
        REQUIRE(loaded.getHistory()[1].age == Approx(original.getHistory()[1].age));
        REQUIRE(loaded.getHistory()[2].age == Approx(original.getHistory()[2].age));
        
        // Clean up
        std::filesystem::remove(testFile);
    }
}