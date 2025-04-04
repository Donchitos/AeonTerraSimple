#include "aeonterracpp/core/spherical_grid.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace aeonterracpp;

// Helper function to measure execution time
template<typename Func>
double measureExecutionTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    try {
        std::cout << "========== AeonTerra C++ Implementation Verification ==========" << std::endl;
        
        // 1. Create a spherical grid with resolution 4 (reasonable size for testing)
        std::cout << "\n[1] Creating spherical grid (resolution 4)..." << std::endl;
        double createTime = measureExecutionTime([&]() {
            SphericalGrid grid(6371.0f, 4);
            
            // Print basic information
            std::cout << "Grid created successfully!" << std::endl;
            std::cout << "Resolution: " << grid.getResolution() << std::endl;
            std::cout << "Vertex count: " << grid.getVertexCount() << std::endl;
            std::cout << "Face count: " << grid.getFaceCount() << std::endl;
            
            // 2. Test coordinate conversions
            std::cout << "\n[2] Testing coordinate conversions..." << std::endl;
            
            // Test some key points
            std::vector<std::pair<std::string, std::pair<float, float>>> testPoints = {
                {"North Pole", {90.0f, 0.0f}},
                {"South Pole", {-90.0f, 0.0f}},
                {"Prime Meridian Equator", {0.0f, 0.0f}},
                {"Date Line Equator", {0.0f, 180.0f}},
                {"Tokyo", {35.6895f, 139.6917f}},
                {"New York", {40.7128f, -74.0060f}}
            };
            
            for (const auto& [name, coords] : testPoints) {
                float lat = coords.first;
                float lon = coords.second;
                
                // Convert to 3D coordinates
                Vec3f pos = grid.latLonToCartesian(lat, lon);
                
                // Convert back to lat/lon
                auto [newLat, newLon] = grid.cartesianToLatLon(pos);
                
                // Calculate error
                float latError = std::abs(lat - newLat);
                float lonError = std::abs(lon - newLon);
                if (lonError > 180.0f) lonError = 360.0f - lonError;
                
                std::cout << name << ": "
                          << std::fixed << std::setprecision(4)
                          << "(" << lat << "°, " << lon << "°) -> "
                          << "(" << newLat << "°, " << newLon << "°) "
                          << "Error: " << latError << "°, " << lonError << "°" << std::endl;
            }
            
            // 3. Test region queries
            std::cout << "\n[3] Testing region queries..." << std::endl;
            
            // Define some test regions
            std::vector<std::pair<std::string, BoundingBox>> regions = {
                {"North America", BoundingBox(25.0f, -130.0f, 50.0f, -60.0f)},
                {"Europe", BoundingBox(35.0f, -10.0f, 60.0f, 30.0f)},
                {"Australia", BoundingBox(-40.0f, 110.0f, -10.0f, 155.0f)},
                {"Small Area", BoundingBox(40.0f, -75.0f, 41.0f, -74.0f)}
            };
            
            for (const auto& [name, region] : regions) {
                auto vertices = grid.getVerticesInRegion(region);
                std::cout << name << ": Found " << vertices.size() << " vertices" << std::endl;
                
                // Print a couple of sample vertices if available
                if (!vertices.empty()) {
                    std::cout << "  Sample vertices:" << std::endl;
                    for (size_t i = 0; i < std::min(size_t(2), vertices.size()); ++i) {
                        Vec3f pos = grid.getVertex(vertices[i]);
                        auto [lat, lon] = grid.cartesianToLatLon(pos);
                        std::cout << "  - Vertex " << vertices[i] << ": (" 
                                  << lat << "°, " << lon << "°)" << std::endl;
                    }
                }
            }
            
            // 4. Performance test with higher resolution subdivision
            std::cout << "\n[4] Performance test: Subdividing to higher resolution..." << std::endl;
            
            // Create a small grid for subdivision testing
            SphericalGrid testGrid(6371.0f, 2);
            std::cout << "Starting with resolution 2: " 
                      << testGrid.getVertexCount() << " vertices, "
                      << testGrid.getFaceCount() << " faces" << std::endl;
            
            // Subdivide once and measure time
            double subdivideTime = measureExecutionTime([&]() {
                testGrid.subdivide(1);
            });
            
            std::cout << "After subdivision: "
                      << testGrid.getVertexCount() << " vertices, "
                      << testGrid.getFaceCount() << " faces" << std::endl;
            std::cout << "Subdivision time: " << subdivideTime << " ms" << std::endl;
        });
        
        std::cout << "\nTotal execution time: " << createTime << " ms" << std::endl;
        std::cout << "\nVerification completed successfully!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception occurred" << std::endl;
        return 2;
    }
}