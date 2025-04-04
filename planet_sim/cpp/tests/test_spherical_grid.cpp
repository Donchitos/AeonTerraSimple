#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "aeonterracpp/core/spherical_grid.h"

using namespace aeonterracpp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("SphericalGrid Construction", "[spherical_grid]") {
    SECTION("Default construction") {
        SphericalGrid grid;
        REQUIRE(grid.getRadius() == 6371.0f);
        REQUIRE(grid.getResolution() == 5);
        
        // At resolution 5, we expect 10242 vertices (12 * 4^5)
        REQUIRE(grid.getVertexCount() == 10242);
        
        // At resolution 5, we expect 20480 faces (20 * 4^5)
        REQUIRE(grid.getFaceCount() == 20480);
    }
    
    SECTION("Custom parameters") {
        SphericalGrid grid(1000.0f, 2);
        REQUIRE(grid.getRadius() == 1000.0f);
        REQUIRE(grid.getResolution() == 2);
        
        // At resolution 2, we expect 162 vertices (12 * 4^2)
        REQUIRE(grid.getVertexCount() == 162);
        
        // At resolution 2, we expect 320 faces (20 * 4^2)
        REQUIRE(grid.getFaceCount() == 320);
    }
}

TEST_CASE("SphericalGrid Subdivision", "[spherical_grid]") {
    SphericalGrid grid(6371.0f, 1);
    REQUIRE(grid.getVertexCount() == 42);
    REQUIRE(grid.getFaceCount() == 80);
    
    SECTION("Single level subdivision") {
        grid.subdivide(1);
        REQUIRE(grid.getResolution() == 2);
        REQUIRE(grid.getVertexCount() == 162);
        REQUIRE(grid.getFaceCount() == 320);
    }
    
    SECTION("Multi-level subdivision") {
        grid.subdivide(2);
        REQUIRE(grid.getResolution() == 3);
        REQUIRE(grid.getVertexCount() == 642);
        REQUIRE(grid.getFaceCount() == 1280);
    }
}

TEST_CASE("SphericalGrid Geometry", "[spherical_grid]") {
    SphericalGrid grid(6371.0f, 3);
    
    SECTION("All vertices should be on the sphere") {
        for (size_t i = 0; i < grid.getVertexCount(); ++i) {
            const Vec3f& vertex = grid.getVertex(i);
            REQUIRE_THAT(vertex.norm(), WithinRel(grid.getRadius(), 1e-6f));
        }
    }
    
    SECTION("All faces should be valid") {
        for (size_t i = 0; i < grid.getFaceCount(); ++i) {
            const Vec3i& face = grid.getFace(i);
            REQUIRE(face.x() >= 0);
            REQUIRE(face.x() < static_cast<int>(grid.getVertexCount()));
            REQUIRE(face.y() >= 0);
            REQUIRE(face.y() < static_cast<int>(grid.getVertexCount()));
            REQUIRE(face.z() >= 0);
            REQUIRE(face.z() < static_cast<int>(grid.getVertexCount()));
        }
    }
    
    SECTION("All vertices should have neighbors") {
        for (size_t i = 0; i < grid.getVertexCount(); ++i) {
            const auto& neighbors = grid.getNeighbors(static_cast<int>(i));
            REQUIRE(!neighbors.empty());
            
            // Each vertex in an icosphere should have 5 or 6 neighbors
            REQUIRE((neighbors.size() == 5 || neighbors.size() == 6));
            
            // All neighbors should be valid
            for (int neighbor : neighbors) {
                REQUIRE(neighbor >= 0);
                REQUIRE(neighbor < static_cast<int>(grid.getVertexCount()));
            }
        }
    }
}

TEST_CASE("SphericalGrid Coordinate Conversion", "[spherical_grid]") {
    SphericalGrid grid(6371.0f, 4);
    
    SECTION("Lat/Lon to Cartesian") {
        // North pole
        Vec3f north = grid.latLonToCartesian(90.0f, 0.0f);
        REQUIRE_THAT(north.x(), WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(north.y(), WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(north.z(), WithinAbs(6371.0f, 1e-5f));
        
        // South pole
        Vec3f south = grid.latLonToCartesian(-90.0f, 0.0f);
        REQUIRE_THAT(south.x(), WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(south.y(), WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(south.z(), WithinAbs(-6371.0f, 1e-5f));
        
        // Equator at prime meridian
        Vec3f equator = grid.latLonToCartesian(0.0f, 0.0f);
        REQUIRE_THAT(equator.x(), WithinAbs(6371.0f, 1e-5f));
        REQUIRE_THAT(equator.y(), WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(equator.z(), WithinAbs(0.0f, 1e-5f));
    }
    
    SECTION("Cartesian to Lat/Lon") {
        // North pole
        auto [lat1, lon1] = grid.cartesianToLatLon(Vec3f(0.0f, 0.0f, 6371.0f));
        REQUIRE_THAT(lat1, WithinAbs(90.0f, 1e-5f));
        
        // South pole
        auto [lat2, lon2] = grid.cartesianToLatLon(Vec3f(0.0f, 0.0f, -6371.0f));
        REQUIRE_THAT(lat2, WithinAbs(-90.0f, 1e-5f));
        
        // Equator at prime meridian
        auto [lat3, lon3] = grid.cartesianToLatLon(Vec3f(6371.0f, 0.0f, 0.0f));
        REQUIRE_THAT(lat3, WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(lon3, WithinAbs(0.0f, 1e-5f));
        
        // Equator at 90 degrees east
        auto [lat4, lon4] = grid.cartesianToLatLon(Vec3f(0.0f, 6371.0f, 0.0f));
        REQUIRE_THAT(lat4, WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(lon4, WithinAbs(90.0f, 1e-5f));
    }
    
    SECTION("Round-trip conversion") {
        for (float lat = -80.0f; lat <= 80.0f; lat += 20.0f) {
            for (float lon = -160.0f; lon <= 160.0f; lon += 40.0f) {
                Vec3f pos = grid.latLonToCartesian(lat, lon);
                auto [newLat, newLon] = grid.cartesianToLatLon(pos);
                
                REQUIRE_THAT(newLat, WithinAbs(lat, 1e-4f));
                
                // Longitude requires special handling around the -180/180 boundary
                float lonDiff = std::abs(newLon - lon);
                if (lonDiff > 180.0f) {
                    lonDiff = 360.0f - lonDiff;
                }
                REQUIRE_THAT(lonDiff, WithinAbs(0.0f, 1e-4f));
            }
        }
    }
}

TEST_CASE("SphericalGrid Distance Calculations", "[spherical_grid]") {
    SphericalGrid grid(6371.0f, 3);
    
    SECTION("Distance from vertex to itself should be zero") {
        REQUIRE_THAT(grid.calculateDistance(0, 0), WithinAbs(0.0f, 1e-5f));
    }
    
    SECTION("Distance between vertices") {
        // Test a few vertex pairs with expected distances
        // This is more of a sanity check since we don't know exact vertex positions
        
        // Check a few neighbor distances
        for (size_t i = 0; i < 10; ++i) {
            int v = static_cast<int>(i);
            const auto& neighbors = grid.getNeighbors(v);
            
            for (int n : neighbors) {
                float dist = grid.calculateDistance(v, n);
                
                // Distance should be positive
                REQUIRE(dist > 0.0f);
                
                // At resolution 3, neighboring vertices should be roughly 1000-1500km apart
                // This is just a rough estimate for Earth-sized planet
                REQUIRE(dist < grid.getRadius() / 2.0f);
            }
        }
    }
}

TEST_CASE("SphericalGrid Region Queries", "[spherical_grid]") {
    SphericalGrid grid(6371.0f, 4);
    
    SECTION("Query for North Pole region") {
        BoundingBox region(60.0f, -180.0f, 90.0f, 180.0f);
        auto vertices = grid.getVerticesInRegion(region);
        
        REQUIRE(!vertices.empty());
        
        // Check that all returned vertices are actually in the region
        for (int idx : vertices) {
            auto [lat, lon] = grid.cartesianToLatLon(grid.getVertex(idx));
            REQUIRE(lat >= region.latMin);
            REQUIRE(lat <= region.latMax);
        }
    }
    
    SECTION("Query for Equatorial region") {
        BoundingBox region(-10.0f, -10.0f, 10.0f, 10.0f);
        auto vertices = grid.getVerticesInRegion(region);
        
        REQUIRE(!vertices.empty());
        
        // Check that all returned vertices are actually in the region
        for (int idx : vertices) {
            auto [lat, lon] = grid.cartesianToLatLon(grid.getVertex(idx));
            
            // Handle longitude wraparound
            if (lon < region.lonMin && region.lonMax > 160.0f) {
                lon += 360.0f;
            } else if (lon > region.lonMax && region.lonMin < -160.0f) {
                lon -= 360.0f;
            }
            
            REQUIRE(lat >= region.latMin);
            REQUIRE(lat <= region.latMax);
            REQUIRE(lon >= region.lonMin);
            REQUIRE(lon <= region.lonMax);
        }
    }
    
    SECTION("Find nearest vertex") {
        float testLat = 45.0f;
        float testLon = 45.0f;
        
        int nearest = grid.findNearestVertex(testLat, testLon);
        REQUIRE(nearest >= 0);
        REQUIRE(nearest < static_cast<int>(grid.getVertexCount()));
        
        Vec3f targetPos = grid.latLonToCartesian(testLat, testLon);
        Vec3f nearestPos = grid.getVertex(nearest);
        
        // The nearest position should be the closest to our target
        float nearestDist = (targetPos - nearestPos).norm();
        
        // Check 10 random vertices to make sure none are closer
        for (int i = 0; i < 10; ++i) {
            int randIdx = std::rand() % grid.getVertexCount();
            if (randIdx == nearest) continue;
            
            Vec3f randPos = grid.getVertex(randIdx);
            float randDist = (targetPos - randPos).norm();
            
            REQUIRE(randDist >= nearestDist);
        }
    }
}