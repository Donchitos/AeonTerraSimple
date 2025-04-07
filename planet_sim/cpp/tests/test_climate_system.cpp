#include "aeonterracpp/core/climate_system.h"
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
void export_to_csv(const std::string& filename, const std::vector<float>& data, int width, int height, 
                  const SphericalGrid& grid) {
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

// Test the Atmospheric Circulation component
bool test_atmospheric_circulation(const Planet& planet) {
    std::cout << "Testing AtmosphericCirculation..." << std::endl;
    
    // Create parameters with Earth-like values
    ClimateParams params;
    
    // Create atmosphere
    auto start = std::chrono::high_resolution_clock::now();
    AtmosphericCirculation atmosphere(planet, params);
    atmosphere.initialize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "  Initialized atmosphere in " << elapsed.count() << " seconds" << std::endl;
    
    // Test a simulation step
    start = std::chrono::high_resolution_clock::now();
    atmosphere.simulate(1.0, 182.0f); // 1 million year step, mid-year
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    std::cout << "  Simulated atmosphere step in " << elapsed.count() << " seconds" << std::endl;
    
    // Check some values
    bool success = true;
    const SphericalGrid& grid = planet.getGrid();
    
    // Sample at different latitudes
    std::vector<float> latitudes = {0.0f, 30.0f, 60.0f, 90.0f};
    
    for (float lat : latitudes) {
        // Create position at this latitude
        Vec3f pos;
        float latRad = toRadians(lat);
        pos.x() = std::cos(latRad);
        pos.y() = 0.0f;
        pos.z() = std::sin(latRad);
        pos *= grid.getRadius();
        
        // Check temperature
        float temp = atmosphere.getTemperature(pos);
        std::cout << "  Temperature at " << lat << "° latitude: " << temp << " °C" << std::endl;
        success &= check_range(temp, -80.0f, 50.0f, "temperature");
        
        // Check pressure
        float pressure = atmosphere.getPressure(pos);
        std::cout << "  Pressure at " << lat << "° latitude: " << pressure << " hPa" << std::endl;
        success &= check_range(pressure, 500.0f, 1100.0f, "pressure");
        
        // Check wind
        Vec3f wind = atmosphere.getWindVector(pos);
        float windSpeed = wind.norm();
        std::cout << "  Wind speed at " << lat << "° latitude: " << windSpeed << " m/s" << std::endl;
        success &= check_range(windSpeed, 0.0f, 50.0f, "wind speed");
    }
    
    // Export data for analysis
    export_to_csv("atmosphere_temperature.csv", atmosphere.getTemperatureData(), 72, 36, grid);
    export_to_csv("atmosphere_pressure.csv", atmosphere.getPressureData(), 72, 36, grid);
    
    return success;
}

// Test the Oceanic Circulation component
bool test_oceanic_circulation(const Planet& planet, const AtmosphericCirculation& atmosphere) {
    std::cout << "Testing OceanicCirculation..." << std::endl;
    
    // Create parameters with Earth-like values
    ClimateParams params;
    
    // Create ocean
    auto start = std::chrono::high_resolution_clock::now();
    OceanicCirculation ocean(planet, atmosphere, params);
    ocean.initialize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "  Initialized ocean in " << elapsed.count() << " seconds" << std::endl;
    
    // Test a simulation step
    start = std::chrono::high_resolution_clock::now();
    ocean.simulate(1.0, 182.0f); // 1 million year step, mid-year
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    std::cout << "  Simulated ocean step in " << elapsed.count() << " seconds" << std::endl;
    
    // Check some values
    bool success = true;
    const SphericalGrid& grid = planet.getGrid();
    
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
        
        // Check SST
        float sst = ocean.getSeaSurfaceTemperature(oceanPos);
        std::cout << "  Sea surface temperature at sample ocean cell: " << sst << " °C" << std::endl;
        success &= check_range(sst, -2.0f, 35.0f, "sea surface temperature");
        
        // Check current
        Vec3f current = ocean.getCurrentVector(oceanPos);
        float currentSpeed = current.norm();
        std::cout << "  Current speed at sample ocean cell: " << currentSpeed << " units" << std::endl;
        success &= check_range(currentSpeed, 0.0f, 2.0f, "current speed");
    } else {
        std::cerr << "  WARNING: No ocean cells found for testing" << std::endl;
    }
    
    // Export data for analysis
    export_to_csv("ocean_sst.csv", ocean.getSeaSurfaceTemperatureData(), 72, 36, grid);
    
    return success;
}

// Test the Hydrological System component
bool test_hydrological_system(const Planet& planet, const AtmosphericCirculation& atmosphere, 
                            const OceanicCirculation& ocean) {
    std::cout << "Testing HydrologicalSystem..." << std::endl;
    
    // Create parameters with Earth-like values
    ClimateParams params;
    
    // Create hydrology
    auto start = std::chrono::high_resolution_clock::now();
    HydrologicalSystem hydrology(planet, atmosphere, ocean, params);
    hydrology.initialize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "  Initialized hydrology in " << elapsed.count() << " seconds" << std::endl;
    
    // Test a simulation step
    start = std::chrono::high_resolution_clock::now();
    hydrology.simulate(1.0, 182.0f); // 1 million year step, mid-year
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    std::cout << "  Simulated hydrology step in " << elapsed.count() << " seconds" << std::endl;
    
    // Check some values
    bool success = true;
    const SphericalGrid& grid = planet.getGrid();
    
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
        
        // Check precipitation
        float precip = hydrology.getPrecipitation(landPos);
        std::cout << "  Precipitation at sample land cell: " << precip << " mm/year" << std::endl;
        success &= check_range(precip, 0.0f, 5000.0f, "precipitation");
        
        // Check evaporation
        float evap = hydrology.getEvaporation(landPos);
        std::cout << "  Evaporation at sample land cell: " << evap << " mm/year" << std::endl;
        success &= check_range(evap, 0.0f, 3000.0f, "evaporation");
        
        // Check river flow
        float flow = hydrology.getRiverFlow(landPos);
        std::cout << "  River flow at sample land cell: " << flow << " m³/s" << std::endl;
        success &= check_range(flow, 0.0f, 200000.0f, "river flow");
    } else {
        std::cerr << "  WARNING: No land cells found for testing" << std::endl;
    }
    
    // Export data for analysis
    export_to_csv("precipitation.csv", hydrology.getPrecipitationData(), 72, 36, grid);
    export_to_csv("river_flow.csv", hydrology.getRiverFlowData(), 72, 36, grid);
    
    return success;
}

// Test the full Climate System
bool test_climate_system(Planet& planet) {
    std::cout << "Testing ClimateSystem..." << std::endl;
    
    // Create climate system
    auto start = std::chrono::high_resolution_clock::now();
    ClimateSystem climate(planet);
    climate.initialize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "  Initialized climate system in " << elapsed.count() << " seconds" << std::endl;
    
    // Test a simulation step
    start = std::chrono::high_resolution_clock::now();
    climate.simulate(1.0, 182.0f); // 1 million year step, mid-year
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    std::cout << "  Simulated climate step in " << elapsed.count() << " seconds" << std::endl;
    
    // Verify we can access the components
    const AtmosphericCirculation& atmosphere = climate.getAtmosphere();
    const OceanicCirculation& ocean = climate.getOcean();
    const HydrologicalSystem& hydrology = climate.getHydrology();
    
    // Check if we can get the data
    bool success = true;
    success &= !atmosphere.getTemperatureData().empty();
    success &= !ocean.getSeaSurfaceTemperatureData().empty();
    success &= !hydrology.getPrecipitationData().empty();
    
    if (!success) {
        std::cerr << "  ERROR: One or more climate components returned empty data" << std::endl;
    }
    
    // Test file saving/loading
    climate.saveToFile("climate_test.dat");
    success &= climate.loadFromFile("climate_test.dat");
    if (!success) {
        std::cerr << "  ERROR: Failed to save or load climate data" << std::endl;
    }
    
    return success;
}

// Run a stress test with multiple simulation steps
void stress_test(Planet& planet) {
    std::cout << "Running stress test..." << std::endl;
    
    // Create climate system
    ClimateSystem climate(planet);
    climate.initialize();
    
    // Run multiple simulation steps
    const int steps = 12; // One year with monthly steps
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < steps; ++i) {
        float dayOfYear = (365.0f / steps) * i;
        climate.simulate(0.1, dayOfYear);
        std::cout << "  Completed step " << (i+1) << " of " << steps 
                  << " (day " << dayOfYear << ")" << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Stress test completed in " << elapsed.count() << " seconds ("
              << (elapsed.count() / steps) << " seconds per step)" << std::endl;
}

int main() {
    std::cout << "==== Climate System Test Suite ====" << std::endl;
    
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
    
    // Run individual component tests
    bool atmosphereSuccess = test_atmospheric_circulation(planet);
    std::cout << "Atmospheric circulation test " 
              << (atmosphereSuccess ? "PASSED" : "FAILED") << std::endl << std::endl;
    
    // Create atmosphere for ocean test
    ClimateParams params;
    AtmosphericCirculation atmosphere(planet, params);
    atmosphere.initialize();
    
    bool oceanSuccess = test_oceanic_circulation(planet, atmosphere);
    std::cout << "Oceanic circulation test " 
              << (oceanSuccess ? "PASSED" : "FAILED") << std::endl << std::endl;
    
    // Create ocean for hydrology test
    OceanicCirculation ocean(planet, atmosphere, params);
    ocean.initialize();
    
    bool hydrologySuccess = test_hydrological_system(planet, atmosphere, ocean);
    std::cout << "Hydrological system test " 
              << (hydrologySuccess ? "PASSED" : "FAILED") << std::endl << std::endl;
    
    // Test full climate system
    bool climateSuccess = test_climate_system(planet);
    std::cout << "Climate system test " 
              << (climateSuccess ? "PASSED" : "FAILED") << std::endl << std::endl;
    
    // Run stress test
    stress_test(planet);
    
    // Overall result
    bool overallSuccess = atmosphereSuccess && oceanSuccess && hydrologySuccess && climateSuccess;
    std::cout << std::endl << "==== Test Suite " 
              << (overallSuccess ? "PASSED" : "FAILED") << " ====" << std::endl;
    
    return overallSuccess ? 0 : 1;
}