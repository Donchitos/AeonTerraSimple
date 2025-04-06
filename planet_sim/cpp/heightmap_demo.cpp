#include "aeonterracpp/core/planet.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <limits>

using namespace aeonterracpp;

/**
 * @brief Simple class to export heightmap data to different formats
 */
class HeightmapExporter {
public:
    /**
     * @brief Export heightmap as raw grayscale image
     * 
     * @param planet The planet to export
     * @param filename Output filename
     * @param width Image width (pixels)
     * @param height Image height (pixels)
     */
    static void exportGrayscaleRaw(const Planet& planet, const std::string& filename, 
                                int width = 1024, int height = 512) {
        std::cout << "Exporting heightmap to " << filename << "..." << std::endl;
        
        // Allocate pixel buffer
        std::vector<unsigned char> pixels(width * height, 0);
        
        // Min/max elevation for normalization
        float minElevation = std::numeric_limits<float>::max();
        float maxElevation = -std::numeric_limits<float>::max();
        
        // Find min/max elevation
        for (size_t i = 0; i < planet.getVertexCount(); ++i) {
            float elevation = planet.getElevation(static_cast<int>(i));
            minElevation = std::min(minElevation, elevation);
            maxElevation = std::max(maxElevation, elevation);
        }
        
        float elevationRange = maxElevation - minElevation;
        if (elevationRange < 0.001f) elevationRange = 0.001f; // Avoid division by zero
        
        std::cout << "Elevation range: " << minElevation << " to " << maxElevation << " km" << std::endl;
        
        // For each pixel in the output image
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Convert pixel coordinates to lat/lon
                float lon = (static_cast<float>(x) / width * 360.0f) - 180.0f;
                float lat = 90.0f - (static_cast<float>(y) / height * 180.0f);
                
                // Find nearest vertex
                int vertexIdx = planet.getGrid().findNearestVertex(lat, lon);
                
                if (vertexIdx >= 0) {
                    // Get elevation and normalize to 0-255 range
                    float elevation = planet.getElevation(vertexIdx);
                    float normalizedElevation = (elevation - minElevation) / elevationRange;
                    unsigned char pixelValue = static_cast<unsigned char>(normalizedElevation * 255.0f);
                    
                    // Set pixel
                    pixels[y * width + x] = pixelValue;
                }
            }
        }
        
        // Write raw image data to file
        std::ofstream outFile(filename, std::ios::binary);
        outFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
        outFile.close();
        
        std::cout << "Heightmap exported successfully" << std::endl;
    }
    
    /**
     * @brief Export heightmap as PGM image
     * 
     * @param planet The planet to export
     * @param filename Output filename
     * @param width Image width (pixels)
     * @param height Image height (pixels)
     */
    static void exportPGM(const Planet& planet, const std::string& filename, 
                      int width = 1024, int height = 512) {
        std::cout << "Exporting heightmap as PGM to " << filename << "..." << std::endl;
        
        // Allocate pixel buffer
        std::vector<unsigned char> pixels(width * height, 0);
        
        // Min/max elevation for normalization
        float minElevation = std::numeric_limits<float>::max();
        float maxElevation = -std::numeric_limits<float>::max();
        
        // Find min/max elevation
        for (size_t i = 0; i < planet.getVertexCount(); ++i) {
            float elevation = planet.getElevation(static_cast<int>(i));
            minElevation = std::min(minElevation, elevation);
            maxElevation = std::max(maxElevation, elevation);
        }
        
        float elevationRange = maxElevation - minElevation;
        if (elevationRange < 0.001f) elevationRange = 0.001f; // Avoid division by zero
        
        std::cout << "Elevation range: " << minElevation << " to " << maxElevation << " km" << std::endl;
        
        // For each pixel in the output image
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Convert pixel coordinates to lat/lon
                float lon = (static_cast<float>(x) / width * 360.0f) - 180.0f;
                float lat = 90.0f - (static_cast<float>(y) / height * 180.0f);
                
                // Find nearest vertex
                int vertexIdx = planet.getGrid().findNearestVertex(lat, lon);
                
                if (vertexIdx >= 0) {
                    // Get elevation and normalize to 0-255 range
                    float elevation = planet.getElevation(vertexIdx);
                    float normalizedElevation = (elevation - minElevation) / elevationRange;
                    unsigned char pixelValue = static_cast<unsigned char>(normalizedElevation * 255.0f);
                    
                    // Set pixel
                    pixels[y * width + x] = pixelValue;
                }
            }
        }
        
        // Write PGM file
        std::ofstream outFile(filename, std::ios::binary);
        
        // PGM header
        outFile << "P5\n" << width << " " << height << "\n255\n";
        
        // Image data
        outFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
        outFile.close();
        
        std::cout << "PGM file exported successfully" << std::endl;
    }
    
    /**
     * @brief Export heightmap as ASCII visualization to console
     * 
     * @param planet The planet to export
     * @param width Width of ASCII art
     * @param height Height of ASCII art
     */
    static void printAsciiVisualization(const Planet& planet, int width = 80, int height = 40) {
        std::cout << "ASCII Visualization of Planet " << planet.getName() << std::endl;
        std::cout << "Age: " << planet.getAge() << " million years" << std::endl;
        
        // Allocate character buffer
        std::vector<char> chars(width * height, ' ');
        
        // Min/max elevation for normalization
        float minElevation = std::numeric_limits<float>::max();
        float maxElevation = -std::numeric_limits<float>::max();
        
        // Find min/max elevation
        for (size_t i = 0; i < planet.getVertexCount(); ++i) {
            float elevation = planet.getElevation(static_cast<int>(i));
            minElevation = std::min(minElevation, elevation);
            maxElevation = std::max(maxElevation, elevation);
        }
        
        float elevationRange = maxElevation - minElevation;
        if (elevationRange < 0.001f) elevationRange = 0.001f; // Avoid division by zero
        
        // For each character in the output 
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Convert coordinates to lat/lon
                float lon = (static_cast<float>(x) / width * 360.0f) - 180.0f;
                float lat = 90.0f - (static_cast<float>(y) / height * 180.0f);
                
                // Find nearest vertex
                int vertexIdx = planet.getGrid().findNearestVertex(lat, lon);
                
                if (vertexIdx >= 0) {
                    // Get elevation and normalize
                    float elevation = planet.getElevation(vertexIdx);
                    float normalizedElevation = (elevation - minElevation) / elevationRange;
                    
                    // Convert to ASCII character
                    char c;
                    if (elevation < 0.0f) {
                        // Ocean depths
                        float depth = -elevation;
                        if (depth > 6.0f) c = '=';
                        else if (depth > 3.0f) c = '-';
                        else c = '~';
                    } else {
                        // Land heights
                        if (elevation > 5.0f) c = '^';
                        else if (elevation > 2.0f) c = 'n';
                        else if (elevation > 1.0f) c = '.';
                        else c = ' ';
                    }
                    
                    // Set character
                    chars[y * width + x] = c;
                }
            }
        }
        
        // Print the ASCII art
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::cout << chars[y * width + x];
            }
            std::cout << std::endl;
        }
    }
};

/**
 * @brief Generate interesting terrain patterns manually
 */
void generateTerrainPatterns(Planet& planet) {
    std::cout << "Generating terrain patterns..." << std::endl;
    
    // Get grid information
    const SphericalGrid& grid = planet.getGrid();
    size_t vertexCount = grid.getVertexCount();
    
    // Create a fractal mountain range pattern
    for (size_t i = 0; i < vertexCount; ++i) {
        // Get position
        Vec3f position = grid.getVertex(static_cast<int>(i));
        auto [lat, lon] = grid.cartesianToLatLon(position);
        
        // Default to ocean
        float elevation = -3.0f; // Basic ocean depth
        
        // Create several mountain ranges and continents
        
        // 1. Main continent
        float distFromCenter1 = std::sqrt(std::pow(lat - 20, 2) + std::pow(lon - 0, 2));
        if (distFromCenter1 < 40) {
            elevation = 0.5f + 3.0f * std::exp(-distFromCenter1 / 30.0f);
        }
        
        // 2. Island chain
        float distFromCenter2 = std::sqrt(std::pow(lat + 10, 2) + std::pow(lon - 100, 2));
        if (distFromCenter2 < 50) {
            float islandElevation = 0.2f + 1.5f * std::exp(-distFromCenter2 / 10.0f);
            elevation = std::max(elevation, islandElevation);
        }
        
        // 3. Mountain range
        float distFromLine = std::abs(lat - 0.5f * lon - 10);
        if (lon > -60 && lon < 0 && lat > -30 && lat < 30 && distFromLine < 15) {
            float mountainHeight = 3.0f * (1.0f - distFromLine / 15.0f);
            elevation = std::max(elevation, mountainHeight);
        }
        
        // 4. Circular island
        float distFromCenter3 = std::sqrt(std::pow(lat + 40, 2) + std::pow(lon + 120, 2));
        if (distFromCenter3 < 30) {
            float islandElevation = 0.1f + 2.0f * std::exp(-distFromCenter3 / 15.0f);
            elevation = std::max(elevation, islandElevation);
        }
        
        // 5. Add some noise for texture
        float noiseValue = 0.3f * std::sin(lat * 0.2f) * std::cos(lon * 0.3f);
        noiseValue += 0.2f * std::sin(lat * 0.5f + lon * 0.7f);
        elevation += noiseValue;
        
        // Set the elevation
        planet.setElevation(static_cast<int>(i), elevation);
        
        // Set crust properties based on elevation
        if (elevation > 0.0f) {
            // Continental crust
            planet.setCrustType(static_cast<int>(i), 1);
            planet.setCrustThickness(static_cast<int>(i), 30.0f + elevation * 2.0f);
            planet.setCrustDensity(static_cast<int>(i), 2.7f);
        } else {
            // Oceanic crust
            planet.setCrustType(static_cast<int>(i), 0);
            planet.setCrustThickness(static_cast<int>(i), 7.0f);
            planet.setCrustDensity(static_cast<int>(i), 3.0f);
        }
    }
    
    std::cout << "Terrain generation complete" << std::endl;
}

/**
 * @brief Apply simple erosion algorithm
 */
void applySimpleErosion(Planet& planet, float amount) {
    std::cout << "Applying erosion..." << std::endl;
    
    // Get grid information
    const SphericalGrid& grid = planet.getGrid();
    size_t vertexCount = grid.getVertexCount();
    
    // Create a copy of current elevations
    std::vector<float> oldElevation(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        oldElevation[i] = planet.getElevation(static_cast<int>(i));
    }
    
    // Apply simple diffusion-based erosion
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = oldElevation[i];
        
        // Skip underwater points
        if (elevation < 0) continue;
        
        // Get neighbors
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        
        // Calculate average elevation of neighbors
        float avgNeighborElevation = 0;
        for (int neighborIdx : neighbors) {
            avgNeighborElevation += oldElevation[neighborIdx];
        }
        avgNeighborElevation /= neighbors.size();
        
        // Erode toward average of neighbors
        float newElevation = elevation * (1.0f - amount) + avgNeighborElevation * amount;
        
        // Apply stronger erosion to mountain peaks
        if (elevation > 2.0f) {
            float extraErosion = (elevation - 2.0f) * 0.1f * amount;
            newElevation -= extraErosion;
        }
        
        // Set new elevation
        planet.setElevation(vertexIdx, newElevation);
    }
    
    std::cout << "Erosion complete" << std::endl;
}

/**
 * @brief Simple demo showing heightmap generation
 */
int main() {
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "AeonTerra Heightmap Generation Demo" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    // Create planet with resolution level 6 (40,962 vertices)
    Planet planet(6371.0f, 6, "DemoWorld");
    std::cout << "Created planet with " << planet.getVertexCount() << " vertices" << std::endl;
    
    // Generate custom terrain patterns
    generateTerrainPatterns(planet);
    
    // Export initial state heightmap
    HeightmapExporter::exportPGM(planet, "initial_heightmap.pgm");
    HeightmapExporter::printAsciiVisualization(planet);
    
    // Apply erosion in multiple passes
    for (int i = 0; i < 5; i++) {
        applySimpleErosion(planet, 0.2f);
        
        // Export intermediate stage
        if (i == 2) {
            HeightmapExporter::exportPGM(planet, "mid_erosion_heightmap.pgm");
        }
    }
    
    // Advance planet age
    planet.advanceAge(100.0);
    
    // Record the final state
    planet.recordSnapshot();
    
    // Export final heightmap
    HeightmapExporter::exportPGM(planet, "final_heightmap.pgm");
    HeightmapExporter::printAsciiVisualization(planet);
    
    // End timer and print elapsed time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Generation completed in " << (duration / 1000.0) << " seconds" << std::endl;
    
    std::cout << "Final exported files:" << std::endl;
    std::cout << "  - initial_heightmap.pgm" << std::endl;
    std::cout << "  - mid_erosion_heightmap.pgm" << std::endl;
    std::cout << "  - final_heightmap.pgm" << std::endl;
    
    std::cout << "Instructions to view heightmaps:" << std::endl;
    std::cout << "1. PGM files can be viewed with most image viewers" << std::endl;
    std::cout << "2. You can also convert them to other formats with tools like ImageMagick" << std::endl;
    std::cout << "3. The ASCII visualization above shows the terrain pattern" << std::endl;
    
    return 0;
}