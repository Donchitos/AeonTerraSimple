#include "aeonterracpp/core/biome_system.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <fstream>

using namespace aeonterracpp;

// Helper function to verify value is in reasonable range
template<typename T>
bool check_range(const T& value, const T& min_value, const T& max_value, const std::string& name) {
    if (value < min_value || value > max_value) {
        std::cerr << "ERROR: " << name << " value " << value << " is outside range [" 
                  << min_value << "," << max_value << "]" << std::endl;
        return false;
    }
    return true;
}

// Helper function to export a 2D array of values to a CSV file for analysis
void export_biomes_to_csv(const std::string& filename, const std::vector<BiomeType>& data, 
                         const BiomeSystem& biomeSystem, int width, int height, 
                         const SphericalGrid& grid) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "latitude,longitude,biome_id,biome_name" << std::endl;
    
    // Write data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Calculate lat/lon for this cell
            float lon = (static_cast<float>(x) / width) * 360.0f - 180.0f;
            float lat = 90.0f - (static_cast<float>(y) / height) * 180.0f;
            
            // Find the closest vertex
            Vec3f latLonPos;
            float latRad = toRadians(lat);
            float lonRad = toRadians(lon);
            latLonPos.x() = std::cos(latRad) * std::cos(lonRad);
            latLonPos.y() = std::cos(latRad) * std::sin(lonRad);
            latLonPos.z() = std::sin(latRad);
            
            int vertex = grid.findClosestVertex(latLonPos);
            
            if (vertex >= 0 && vertex < static_cast<int>(data.size())) {
                BiomeType biome = data[vertex];
                std::string biomeName = biomeSystem.getBiomeName(biome);
                
                file << lat << "," << lon << "," << static_cast<int>(biome) << "," << biomeName << std::endl;
            } else {
                file << lat << "," << lon << ",0,Unknown" << std::endl;
            }
        }
    }
    
    file.close();
    std::cout << "Exported biome data to " << filename << std::endl;
}

// Helper function to export vegetation and soil data to CSV
void export_float_to_csv(const std::string& filename, const std::vector<float>& data, 
                       int width, int height, const SphericalGrid& grid) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "latitude,longitude,value" << std::endl;
    
    // Write data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Calculate lat/lon for this cell
            float lon = (static_cast<float>(x) / width) * 360.0f - 180.0f;
            float lat = 90.0f - (static_cast<float>(y) / height) * 180.0f;
            
            // Find the closest vertex
            Vec3f latLonPos;
            float latRad = toRadians(lat);
            float lonRad = toRadians(lon);
            latLonPos.x() = std::cos(latRad) * std::cos(lonRad);
            latLonPos.y() = std::cos(latRad) * std::sin(lonRad);
            latLonPos.z() = std::sin(latRad);
            
            int vertex = grid.findClosestVertex(latLonPos);
            
            if (vertex >= 0 && vertex < static_cast<int>(data.size())) {
                file << lat << "," << lon << "," << data[vertex] << std::endl;
            } else {
                file << lat << "," << lon << ",0" << std::endl;
            }
        }
    }
    
    file.close();
    std::cout << "Exported data to " << filename << std::endl;
}

// Test the BiomeSystem
bool test_biome_system(Planet& planet, const ClimateSystem& climate) {
    std::cout << "Testing BiomeSystem..." << std::endl;
    
    // Create biome system
    auto start = std::chrono::high_resolution_clock::now();
    BiomeSystem biomes(planet, climate);
    biomes.initialize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "  Initialized biome system in " << elapsed.count() << " seconds" << std::endl;
    
    // Test a simulation step
    start = std::chrono::high_resolution_clock::now();
    biomes.update(1.0); // 1 million year step
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    std::cout << "  Updated biome system in " << elapsed.count() << " seconds" << std::endl;
    
    // Check some values
    bool success = true;
    const SphericalGrid& grid = planet.getGrid();
    
    // Sample at different locations
    std::cout << "  Sampling biomes at different locations:" << std::endl;
    
    // Find a land cell
    int landVertex = -1;
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        if (planet.getElevation(i) > 0.0f) {
            landVertex = i;
            break;
        }
    }
    
    if (landVertex >= 0) {
        Vec3f landPos = grid.getVertex(landVertex);
        
        // Get biome type and properties
        BiomeType biomeType = biomes.getBiomeType(landPos);
        const BiomeProperties& props = biomes.getBiomeProperties(landPos);
        
        std::cout << "    Land position: Biome = " << props.name << std::endl;
        std::cout << "      Vegetation density: " << biomes.getVegetationDensity(landPos) << std::endl;
        std::cout << "      Erosion rate: " << props.baseErosionRate << std::endl;
        std::cout << "      Soil depth: " << props.soilDepth << std::endl;
        
        // Basic checks
        success &= check_range(biomes.getVegetationDensity(landPos), 0.0f, 1.0f, "vegetation density");
        success &= check_range(props.baseErosionRate, 0.0f, 1.0f, "erosion rate");
        success &= check_range(props.soilDepth, 0.0f, 5.0f, "soil depth");
    } else {
        std::cerr << "  WARNING: No land cells found for testing" << std::endl;
    }
    
    // Find an ocean cell
    int oceanVertex = -1;
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        if (planet.getElevation(i) < 0.0f) {
            oceanVertex = i;
            break;
        }
    }
    
    if (oceanVertex >= 0) {
        Vec3f oceanPos = grid.getVertex(oceanVertex);
        
        // Get biome type and properties
        BiomeType biomeType = biomes.getBiomeType(oceanPos);
        const BiomeProperties& props = biomes.getBiomeProperties(oceanPos);
        
        std::cout << "    Ocean position: Biome = " << props.name << std::endl;
        std::cout << "      Vegetation density: " << biomes.getVegetationDensity(oceanPos) << std::endl;
        
        // Basic checks
        success &= (props.name == "Ocean" || props.name == "Deep Ocean" || 
                   props.name == "Coastal Waters" || props.name == "Ocean Trench");
    } else {
        std::cerr << "  WARNING: No ocean cells found for testing" << std::endl;
    }
    
    // Find a mountain cell if possible
    int mountainVertex = -1;
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        if (planet.getElevation(i) > 1.5f) {
            mountainVertex = i;
            break;
        }
    }
    
    if (mountainVertex >= 0) {
        Vec3f mountainPos = grid.getVertex(mountainVertex);
        
        // Get biome type and properties
        BiomeType biomeType = biomes.getBiomeType(mountainPos);
        const BiomeProperties& props = biomes.getBiomeProperties(mountainPos);
        
        std::cout << "    Mountain position: Biome = " << props.name << std::endl;
        std::cout << "      Vegetation density: " << biomes.getVegetationDensity(mountainPos) << std::endl;
        
        // Basic checks
        success &= (props.name == "Mountain" || props.name == "High Mountain");
    } else {
        std::cerr << "  WARNING: No mountain cells found for testing" << std::endl;
    }
    
    // Export data for analysis
    export_biomes_to_csv("biomes.csv", biomes.getBiomeData(), biomes, 72, 36, grid);
    export_float_to_csv("vegetation.csv", biomes.getVegetationData(), 72, 36, grid);
    export_float_to_csv("soil_depth.csv", biomes.getSoilDepthData(), 72, 36, grid);
    
    return success;
}

// Run a stress test with multiple simulation steps
void stress_test(Planet& planet, ClimateSystem& climate) {
    std::cout << "Running stress test..." << std::endl;
    
    // Create biome system
    BiomeSystem biomes(planet, climate);
    biomes.initialize();
    
    // Run multiple simulation steps
    const int steps = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < steps; ++i) {
        // Update climate first (it affects biomes)
        float dayOfYear = (365.0f / steps) * i;
        climate.simulate(0.1, dayOfYear);
        
        // Then update biomes
        biomes.update(0.1);
        
        std::cout << "  Completed step " << (i+1) << " of " << steps 
                  << " (day " << dayOfYear << ")" << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Stress test completed in " << elapsed.count() << " seconds ("
              << (elapsed.count() / steps) << " seconds per step)" << std::endl;
}

int main() {
    std::cout << "==== Biome System Test Suite ====" << std::endl;
    
    // Create a test planet
    float radius = 6371.0f;
    int resolution = 5; // Resolution level (12,962 vertices)
    std::cout << "Creating test planet with resolution " << resolution << "..." << std::endl;
    Planet planet(radius, resolution, "TestPlanet");
    
    // Initialize simple elevation pattern (basic continents and oceans)
    std::cout << "Initializing elevation pattern..." << std::endl;
    const SphericalGrid& grid = planet.getGrid();
    
    for (size_t i = 0; i < grid.getVertexCount(); ++i) {
        Vec3f pos = grid.getVertex(i);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Simple elevation pattern - continents at mid latitudes
        float elevation;
        if (std::abs(lat) > 10.0f && std::abs(lat) < 60.0f) {
            // Land
            elevation = 0.5f;
            
            // Add some mountains
            if (std::abs(lon) > 30.0f && std::abs(lon) < 60.0f) {
                elevation = 2.0f;
            }
        } else {
            // Ocean
            elevation = -4.0f;
        }
        
        planet.setElevation(i, elevation);
        
        // Set crust type based on elevation
        if (elevation >= 0.0f) {
            planet.setCrustType(i, 1); // Continental
            planet.setCrustThickness(i, 30.0f);
            planet.setCrustDensity(i, 2.7f);
        } else {
            planet.setCrustType(i, 0); // Oceanic
            planet.setCrustThickness(i, 7.0f);
            planet.setCrustDensity(i, 3.0f);
        }
    }
    
    // Create climate system (required for biome system)
    std::cout << "Creating climate system..." << std::endl;
    ClimateSystem climate(planet);
    climate.initialize();
    climate.simulate(1.0, 182.0f); // Run a step with mid-year conditions
    
    // Test biome system
    bool biomeSuccess = test_biome_system(planet, climate);
    std::cout << "Biome system test " 
              << (biomeSuccess ? "PASSED" : "FAILED") << std::endl << std::endl;
    
    // Run stress test
    stress_test(planet, climate);
    
    // Overall result
    std::cout << std::endl << "==== Test Suite " 
              << (biomeSuccess ? "PASSED" : "FAILED") << " ====" << std::endl;
    
    return biomeSuccess ? 0 : 1;
}