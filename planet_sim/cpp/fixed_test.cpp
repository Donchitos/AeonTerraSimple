#include "aeonterracpp/core/spherical_grid.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>

int main() {
    try {
        std::cout << "====== SphericalGrid Test ======" << std::endl;
        
        // Create grid with resolution 0 (just icosahedron)
        std::cout << "\n[1] Creating grid with resolution 0..." << std::endl;
        aeonterracpp::SphericalGrid grid(6371.0f, 0);
        
        // Basic info
        std::cout << "Resolution: " << grid.getResolution() << std::endl;
        std::cout << "Vertex count: " << grid.getVertexCount() << std::endl;
        std::cout << "Face count: " << grid.getFaceCount() << std::endl;
        
        // Test one vertex
        if (grid.getVertexCount() > 0) {
            aeonterracpp::Vec3f v = grid.getVertex(0);
            std::cout << "First vertex: (" << v.x() << ", " << v.y() << ", " << v.z() << ")" << std::endl;
        }
        
        // Test neighbors
        std::cout << "\n[2] Testing neighbor queries..." << std::endl;
        if (grid.getVertexCount() > 0) {
            const auto& neighbors = grid.getNeighbors(0);
            std::cout << "Vertex 0 has " << neighbors.size() << " neighbors" << std::endl;
        }
        
        // Test coordinate conversion
        std::cout << "\n[3] Testing coordinate conversion..." << std::endl;
        aeonterracpp::Vec3f northPole = grid.latLonToCartesian(90.0f, 0.0f);
        std::cout << "North Pole position: (" << northPole.x() << ", " 
                 << northPole.y() << ", " << northPole.z() << ")" << std::endl;
        
        // Test round-trip conversion
        auto [lat, lon] = grid.cartesianToLatLon(northPole);
        std::cout << "North Pole round-trip: (" << lat << "°, " << lon << "°)" << std::endl;
        
        // Test subdividing once
        std::cout << "\n[4] Testing subdivision (level 1)..." << std::endl;
        grid.subdivide(1);
        std::cout << "Resolution: " << grid.getResolution() << std::endl;
        std::cout << "Vertex count: " << grid.getVertexCount() << std::endl;
        std::cout << "Face count: " << grid.getFaceCount() << std::endl;
        
        // Test findNearestVertex
        std::cout << "\n[5] Testing findNearestVertex..." << std::endl;
        int equatorIdx = grid.findNearestVertex(0.0f, 0.0f);
        if (equatorIdx >= 0) {
            std::cout << "Found nearest vertex to equator at index " << equatorIdx << std::endl;
            aeonterracpp::Vec3f v = grid.getVertex(equatorIdx);
            std::cout << "Position: (" << v.x() << ", " << v.y() << ", " << v.z() << ")" << std::endl;
        } else {
            std::cout << "Warning: No nearest vertex found for equator" << std::endl;
        }
        
        // Test vertex in region
        std::cout << "\n[6] Testing getVerticesInRegion..." << std::endl;
        aeonterracpp::BoundingBox region(0.0f, 0.0f, 45.0f, 45.0f);
        auto vertices = grid.getVerticesInRegion(region);
        std::cout << "Found " << vertices.size() << " vertices in region" << std::endl;
        
        std::cout << "\nTest completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception occurred" << std::endl;
        return 2;
    }
}