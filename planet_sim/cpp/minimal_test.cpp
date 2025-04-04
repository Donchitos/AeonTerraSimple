#include "aeonterracpp/core/spherical_grid.h"
#include <iostream>

int main() {
    try {
        std::cout << "Starting minimal test..." << std::endl;
        
        // Create a minimal spherical grid with resolution 0 (just the icosahedron)
        std::cout << "Creating minimal spherical grid (resolution 0)..." << std::endl;
        aeonterracpp::SphericalGrid grid(6371.0f, 0);
        
        // Print basic info
        std::cout << "Basic info:" << std::endl;
        std::cout << "Radius: " << grid.getRadius() << " km" << std::endl;
        std::cout << "Resolution: " << grid.getResolution() << std::endl;
        std::cout << "Vertex count: " << grid.getVertexCount() << std::endl;
        std::cout << "Face count: " << grid.getFaceCount() << std::endl;
        
        // Print the first vertex
        if (grid.getVertexCount() > 0) {
            std::cout << "First vertex: " << grid.getVertex(0).x() << ", " 
                     << grid.getVertex(0).y() << ", " << grid.getVertex(0).z() << std::endl;
        }
        
        std::cout << "Minimal test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return 2;
    }
}