#include "aeonterracpp/core/spherical_grid.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>

int main() {
    try {
        std::cout << "=== Step 1: Creating basic grid (resolution 0) ===" << std::endl;
        // Start with resolution 0 (just the base icosahedron)
        aeonterracpp::SphericalGrid grid(6371.0f, 0);
        std::cout << "Basic grid created with " << grid.getVertexCount() << " vertices" << std::endl;
        
        std::cout << "\n=== Step 2: Testing basic grid functionality ===" << std::endl;
        // Test basic functionality on the initial grid
        std::cout << "Vertex count: " << grid.getVertexCount() << std::endl;
        std::cout << "Face count: " << grid.getFaceCount() << std::endl;
        std::cout << "First vertex position: (" << grid.getVertex(0).x() << ", "
                 << grid.getVertex(0).y() << ", " << grid.getVertex(0).z() << ")" << std::endl;
                
        std::cout << "\n=== Step 3: Incrementally subdividing grid ===" << std::endl;
        // Now try to subdivide one level at a time
        std::cout << "Subdividing 1 level..." << std::endl;
        grid.subdivide(1);
        std::cout << "Grid after 1 subdivision: " << grid.getVertexCount() << " vertices, "
                 << grid.getFaceCount() << " faces" << std::endl;
                
        std::cout << "Subdividing 1 more level..." << std::endl;
        grid.subdivide(1);
        std::cout << "Grid after 2 subdivisions: " << grid.getVertexCount() << " vertices, "
                 << grid.getFaceCount() << " faces" << std::endl;
                 
        std::cout << "\n=== Step 4: Testing coordinate conversions ===" << std::endl;
        
        // Print some basic information
        std::cout << "SphericalGrid created successfully!" << std::endl;
        std::cout << "------------------------------" << std::endl;
        std::cout << "Radius: " << grid.getRadius() << " km" << std::endl;
        std::cout << "Resolution: " << grid.getResolution() << std::endl;
        std::cout << "Vertex count: " << grid.getVertexCount() << std::endl;
        std::cout << "Face count: " << grid.getFaceCount() << std::endl;
        std::cout << "------------------------------" << std::endl;
        
        // Print information about a few vertices
        std::cout << "Sample vertices:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), grid.getVertexCount()); ++i) {
            aeonterracpp::Vec3f v = grid.getVertex(i);
            auto [lat, lon] = grid.cartesianToLatLon(v);
            std::cout << "Vertex " << i << ": (" << std::fixed << std::setprecision(3)
                      << v.x() << ", " << v.y() << ", " << v.z() << ") - Lat/Lon: ("
                      << lat << "°, " << lon << "°)" << std::endl;
        }
        std::cout << "------------------------------" << std::endl;
        
        // Try some coordinate conversions
        std::cout << "Testing coordinate conversions:" << std::endl;
        
        // Test North Pole
        aeonterracpp::Vec3f northPole = grid.latLonToCartesian(90.0f, 0.0f);
        std::cout << "North Pole position: (" << northPole.x() << ", " << northPole.y() << ", " << northPole.z() << ")" << std::endl;
        
        // Test round-trip conversion
        auto [lat, lon] = grid.cartesianToLatLon(northPole);
        std::cout << "North Pole round-trip: (" << lat << "°, " << lon << "°)" << std::endl;
        
        // Test nearest vertex
        std::cout << "Testing nearest vertex functionality:" << std::endl;
        for (float testLat : {0.0f, 45.0f, 90.0f}) {
            for (float testLon : {0.0f, 90.0f, 180.0f}) {
                int vertexIdx = grid.findNearestVertex(testLat, testLon);
                if (vertexIdx >= 0) {
                    aeonterracpp::Vec3f vertex = grid.getVertex(vertexIdx);
                    auto [vLat, vLon] = grid.cartesianToLatLon(vertex);
                    std::cout << "Nearest to (" << testLat << "°, " << testLon << "°): vertex "
                              << vertexIdx << " at (" << vLat << "°, " << vLon << "°)" << std::endl;
                } else {
                    std::cout << "No nearest vertex found for (" << testLat << "°, " << testLon << "°)" << std::endl;
                }
            }
        }
        std::cout << "------------------------------" << std::endl;
        
        // Test neighbor queries
        std::cout << "Testing neighbor queries:" << std::endl;
        int testVertex = 0;
        const auto& neighbors = grid.getNeighbors(testVertex);
        std::cout << "Vertex " << testVertex << " has " << neighbors.size() << " neighbors: ";
        for (int n : neighbors) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Simple test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception occurred" << std::endl;
        return 2;
    }
}