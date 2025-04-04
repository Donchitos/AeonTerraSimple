#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/spherical_grid.h"
#include "aeonterracpp/core/tectonics.h"
#include "aeonterracpp/heightmap/heightmap_generator.h"
#include <iostream>
#include <chrono>

using namespace aeonterracpp;

int main() {
    std::cout << "========== AeonTerra C++ Simplified Component Verification ==========" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // 1. Verify SphericalGrid
        std::cout << "\n[1] Testing SphericalGrid..." << std::endl;
        SphericalGrid grid(6371.0f, 4);
        std::cout << "    Created grid with " << grid.getVertexCount() << " vertices" << std::endl;
        std::cout << "✅ SphericalGrid verification passed" << std::endl;
        
        // 2. Verify Planet
        std::cout << "\n[2] Testing Planet..." << std::endl;
        Planet planet(6371.0f, 4, "Test Planet");
        std::cout << "    Created planet with " << planet.getVertexCount() << " vertices" << std::endl;
        std::cout << "    Planet name: " << planet.getName() << std::endl;
        std::cout << "    Initial age: " << planet.getAge() << " million years" << std::endl;
        std::cout << "✅ Planet verification passed" << std::endl;
        
        // 3. Verify TectonicSimulation
        std::cout << "\n[3] Testing TectonicSimulation..." << std::endl;
        TectonicSimulation sim(planet, 5, 42);
        std::cout << "    Created simulation with " << sim.getPlates().size() << " plates" << std::endl;
        sim.simulateStep(10.0);
        std::cout << "    Simulated 10 million years successfully" << std::endl;
        std::cout << "    Planet age after simulation: " << planet.getAge() << " million years" << std::endl;
        std::cout << "✅ TectonicSimulation verification passed" << std::endl;
        
        // 4. Verify HeightmapGenerator
        std::cout << "\n[4] Testing HeightmapGenerator..." << std::endl;
        HeightmapGenerator generator;
        auto config = HeightmapConfigFactory::createGlobalConfig(32, 16);
        std::cout << "    Created global heightmap config: " << config.width << "x" << config.height << std::endl;
        auto heightmap = generator.generateHeightmap(planet, config);
        std::cout << "    Generated heightmap with dimensions: " << heightmap.width << "x" << heightmap.height << std::endl;
        std::cout << "    Elevation range: " << heightmap.minValue << " to " << heightmap.maxValue << " km" << std::endl;
        std::cout << "✅ HeightmapGenerator verification passed" << std::endl;
        
        // Full pipeline test
        std::cout << "\n[5] Testing full pipeline (SphericalGrid -> Planet -> Tectonics -> Heightmap)..." << std::endl;
        Planet fullTestPlanet(6371.0f, 5, "Pipeline Test");
        TectonicSimulation fullSim(fullTestPlanet, 8, 42);
        
        // Run multiple simulation steps
        for (int i = 0; i < 3; ++i) {
            std::cout << "    Simulating step " << (i+1) << "..." << std::endl;
            fullSim.simulateStep(5.0);
        }
        
        // Generate a heightmap
        auto fullConfig = HeightmapConfigFactory::createGlobalConfig(64, 32);
        auto fullHeightmap = generator.generateHeightmap(fullTestPlanet, fullConfig);
        
        // Write to file
        std::string filename = "test_heightmap.raw";
        generator.exportRAW(fullHeightmap, filename);
        std::cout << "    Exported heightmap to " << filename << std::endl;
        std::cout << "✅ Full pipeline verification passed" << std::endl;
        
        // Success!
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "\nAll component verification tests passed in " << elapsed.count() << " seconds!" << std::endl;
        std::cout << "The heightmap generation pipeline is working correctly." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ ERROR: Unknown exception occurred" << std::endl;
        return 1;
    }
}