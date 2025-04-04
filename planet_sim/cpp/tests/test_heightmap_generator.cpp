#include "aeonterracpp/heightmap/heightmap_generator.h"
#include "aeonterracpp/core/planet.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>
#include <memory>
#include <cmath>

using namespace aeonterracpp;
using Catch::Matchers::WithinAbs;
using Catch::Approx;

TEST_CASE("HeightmapData class", "[heightmap]") {
    SECTION("Construction and basic properties") {
        int width = 100;
        int height = 50;
        BoundingBox region(-30.0f, -60.0f, 30.0f, 60.0f);
        
        HeightmapData heightmap(width, height, region);
        
        REQUIRE(heightmap.width == width);
        REQUIRE(heightmap.height == height);
        REQUIRE(heightmap.region.latMin == region.latMin);
        REQUIRE(heightmap.region.lonMin == region.lonMin);
        REQUIRE(heightmap.region.latMax == region.latMax);
        REQUIRE(heightmap.region.lonMax == region.lonMax);
    }
    
    SECTION("Get/set height values") {
        HeightmapData heightmap(10, 5, BoundingBox(0.0f, 0.0f, 1.0f, 1.0f));
        
        // Set test values
        heightmap.set(0, 0, -5.0f);
        heightmap.set(9, 4, 10.0f);
        heightmap.set(5, 2, 3.5f);
        
        // Get values back
        REQUIRE(heightmap.get(0, 0) == -5.0f);
        REQUIRE(heightmap.get(9, 4) == 10.0f);
        REQUIRE(heightmap.get(5, 2) == 3.5f);
        
        // Check bounds
        REQUIRE_THROWS_AS(heightmap.get(-1, 0), std::out_of_range);
        REQUIRE_THROWS_AS(heightmap.get(10, 0), std::out_of_range);
        REQUIRE_THROWS_AS(heightmap.get(0, -1), std::out_of_range);
        REQUIRE_THROWS_AS(heightmap.get(0, 5), std::out_of_range);
        
        REQUIRE_THROWS_AS(heightmap.set(-1, 0, 0.0f), std::out_of_range);
        REQUIRE_THROWS_AS(heightmap.set(10, 0, 0.0f), std::out_of_range);
        
        // Check min/max values
        REQUIRE(heightmap.minValue == -5.0f);
        REQUIRE(heightmap.maxValue == 10.0f);
    }
    
    SECTION("Normalization") {
        HeightmapData heightmap(5, 5, BoundingBox(0.0f, 0.0f, 1.0f, 1.0f));
        
        // Set values with a known range
        heightmap.set(0, 0, -10.0f);
        heightmap.set(2, 2, 0.0f);
        heightmap.set(4, 4, 10.0f);
        
        // Normalize
        heightmap.normalize();
        
        // Check that values are normalized to 0-1 range
        REQUIRE(heightmap.get(0, 0) == 0.0f);
        REQUIRE(heightmap.get(2, 2) == 0.5f);
        REQUIRE(heightmap.get(4, 4) == 1.0f);
        
        REQUIRE(heightmap.minValue == 0.0f);
        REQUIRE(heightmap.maxValue == 1.0f);
    }
    
    SECTION("Blending") {
        HeightmapData heightmap1(3, 3, BoundingBox(0.0f, 0.0f, 1.0f, 1.0f));
        HeightmapData heightmap2(3, 3, BoundingBox(0.0f, 0.0f, 1.0f, 1.0f));
        
        // Set values in both heightmaps
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                heightmap1.set(x, y, 10.0f);
                heightmap2.set(x, y, 20.0f);
            }
        }
        
        // Blend with 0.25 weight
        heightmap1.blend(heightmap2, 0.25f);
        
        // Check blended values: 10 * 0.75 + 20 * 0.25 = 12.5
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                REQUIRE(heightmap1.get(x, y) == 12.5f);
            }
        }
        
        // Try blending with mismatched dimensions
        HeightmapData heightmap3(4, 3, BoundingBox(0.0f, 0.0f, 1.0f, 1.0f));
        REQUIRE_THROWS_AS(heightmap1.blend(heightmap3, 0.5f), std::invalid_argument);
    }
}

TEST_CASE("HeightmapGenerator", "[heightmap]") {
    SECTION("Basic generation test") {
        // Create a small test planet
        Planet planet(6371.0f, 2);  // Resolution 2 for fast tests
        
        // Set some test elevations
        for (size_t i = 0; i < planet.getVertexCount(); ++i) {
            int vertexIdx = static_cast<int>(i);
            Vec3f pos = planet.getGrid().getVertex(vertexIdx);
            auto [lat, lon] = planet.getGrid().cartesianToLatLon(pos);
            
            // Simple elevation pattern
            float elevation = std::sin(toRadians(lat)) * 2.0f;
            planet.setElevation(vertexIdx, elevation);
        }
        
        // Create heightmap generator
        HeightmapGenerator generator;
        
        // Create a simple config
        HeightmapConfig config;
        config.width = 64;
        config.height = 32;
        config.region = BoundingBox(-90.0f, -180.0f, 90.0f, 180.0f);
        config.enhanceDetail = false;
        
        // Generate heightmap
        HeightmapData heightmap = generator.generateHeightmap(planet, config);
        
        // Check dimensions
        REQUIRE(heightmap.width == config.width);
        REQUIRE(heightmap.height == config.height);
        REQUIRE(heightmap.region.latMin == config.region.latMin);
        REQUIRE(heightmap.region.lonMin == config.region.lonMin);
        
        // Check that heightmap has data
        REQUIRE(heightmap.data.size() == config.width * config.height);
        
        // Check that elevation pattern was captured
        // (North pole should be higher than south pole due to sin(lat) pattern)
        float northPoleElevation = heightmap.get(config.width / 2, 0);  // Top middle
        float southPoleElevation = heightmap.get(config.width / 2, config.height - 1);  // Bottom middle
        REQUIRE(northPoleElevation > southPoleElevation);
    }
    
    SECTION("Config factory") {
        // Test global config
        HeightmapConfig globalConfig = HeightmapConfigFactory::createGlobalConfig(4096, 2048);
        REQUIRE(globalConfig.width == 4096);
        REQUIRE(globalConfig.height == 2048);
        REQUIRE(globalConfig.region.latMin == -90.0f);
        REQUIRE(globalConfig.region.latMax == 90.0f);
        REQUIRE(globalConfig.region.lonMin == -180.0f);
        REQUIRE(globalConfig.region.lonMax == 180.0f);
        
        // Test regional config
        float centerLat = 40.0f;
        float centerLon = -75.0f;
        float width = 10.0f;
        float height = 5.0f;
        int resolution = 20;
        
        HeightmapConfig regionalConfig = HeightmapConfigFactory::createRegionalConfig(
            centerLat, centerLon, width, height, resolution);
        
        REQUIRE(regionalConfig.region.latMin == Approx(centerLat - height / 2.0f));
        REQUIRE(regionalConfig.region.latMax == Approx(centerLat + height / 2.0f));
        REQUIRE(regionalConfig.region.lonMin == Approx(centerLon - width / 2.0f));
        REQUIRE(regionalConfig.region.lonMax == Approx(centerLon + width / 2.0f));
        REQUIRE(regionalConfig.width == width * resolution);
        REQUIRE(regionalConfig.height == height * resolution);
        REQUIRE(regionalConfig.enhanceDetail == true);  // Regional maps have detail enabled
    }
    
    SECTION("Detail enhancement") {
        // Create a test heightmap
        HeightmapData heightmap(64, 64, BoundingBox(-90.0f, -180.0f, 90.0f, 180.0f));
        
        // Set a simple gradient
        for (int y = 0; y < heightmap.height; ++y) {
            float t = static_cast<float>(y) / heightmap.height;
            for (int x = 0; x < heightmap.width; ++x) {
                heightmap.set(x, y, t);
            }
        }
        
        // Store min/max before enhancement
        float minBefore = heightmap.minValue;
        float maxBefore = heightmap.maxValue;
        
        // Create generator and enhance
        HeightmapGenerator generator;
        generator.enhanceDetail(heightmap, 0.5f, 4, 0.2f);
        
        // Check that detail was added (values should have changed)
        bool valuesChanged = false;
        for (int y = 0; y < heightmap.height; ++y) {
            float t = static_cast<float>(y) / heightmap.height;
            for (int x = 0; x < heightmap.width; ++x) {
                if (heightmap.get(x, y) != t) {
                    valuesChanged = true;
                    break;
                }
            }
            if (valuesChanged) break;
        }
        
        REQUIRE(valuesChanged);
        
        // Detail should have expanded the range but not drastically
        REQUIRE(heightmap.minValue <= minBefore);
        REQUIRE(heightmap.maxValue >= maxBefore);
    }
}