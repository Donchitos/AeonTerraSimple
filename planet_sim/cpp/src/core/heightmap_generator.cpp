#include "aeonterracpp/core/heightmap_generator.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace aeonterracpp {

HeightmapGenerator::HeightmapGenerator(const Planet& planet, const TectonicSimulation* tectonics)
    : planet(planet), tectonics(tectonics), width(1024), height(512) {
    
    // Initialize with default parameters
    params = HeightmapParams();
    
    // Initialize random number generator with default seed
    rng = std::mt19937(params.noiseSeed != 0 ? static_cast<unsigned int>(params.noiseSeed) : std::random_device()());
    
    // Default terrain modifier does nothing
    terrainModifier = [](float lat, float lon, float height) { return height; };
}

void HeightmapGenerator::setParameters(const HeightmapParams& params) {
    this->params = params;
    this->width = params.width;
    this->height = params.height;
    
    // Update random number generator seed if changed
    if (params.noiseSeed != 0) {
        rng = std::mt19937(static_cast<unsigned int>(params.noiseSeed));
    }
}

const HeightmapParams& HeightmapGenerator::getParameters() const {
    return params;
}

std::vector<float> HeightmapGenerator::generate(bool equirectangular) {
    // Resize heightmap data to match requested dimensions
    heightmapData.resize(width * height);
    
    // Step 1: Sample planet elevation data and convert to heightmap projection
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Convert pixel coordinates to latitude/longitude
            float lon = (static_cast<float>(x) / width) * 360.0f - 180.0f;
            float lat;
            
            if (equirectangular) {
                // Simple equirectangular projection
                lat = 90.0f - (static_cast<float>(y) / height) * 180.0f;
            } else {
                // Equal-area projection (sinusoidal)
                float normalizedY = static_cast<float>(y) / height;
                lat = 90.0f - normalizedY * 180.0f;
                
                // Adjust longitude based on latitude to preserve area
                float factor = std::cos(toRadians(lat));
                if (factor != 0) {
                    lon = (static_cast<float>(x) / width - 0.5f) * 360.0f / factor;
                }
                
                // Clamp longitude to valid range
                if (lon < -180.0f) lon = -180.0f;
                if (lon > 180.0f) lon = 180.0f;
            }
            
            // Sample elevation from planet
            float elevation = samplePlanetElevation(lat, lon);
            
            // Store in heightmap
            heightmapData[y * width + x] = elevation;
        }
    }
    
    // Step 2: Apply various terrain features
    applyTerrainFeatures();
    
    // Step 3: Apply tectonic features if available
    if (tectonics) {
        applyTectonicFeatures();
    }
    
    // Step 4: Apply climate influence
    applyClimateInfluence();
    
    // Step 5: Apply erosion simulation
    applyErosion();
    
    // Step 6: Apply custom terrain modifier if set
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float lon = (static_cast<float>(x) / width) * 360.0f - 180.0f;
            float lat = 90.0f - (static_cast<float>(y) / height) * 180.0f;
            int idx = y * width + x;
            
            heightmapData[idx] = terrainModifier(lat, lon, heightmapData[idx]);
        }
    }
    
    // Step 7: Normalize the terrain
    normalizeTerrain();
    
    return heightmapData;
}

float HeightmapGenerator::samplePlanetElevation(float lat, float lon) const {
    // Convert lat/lon to 3D position
    const SphericalGrid& grid = planet.getGrid();
    Vec3f position;
    
    // Convert from lat/lon to 3D coordinates
    float latRad = toRadians(lat);
    float lonRad = toRadians(lon);
    
    position.x() = std::cos(latRad) * std::cos(lonRad);
    position.y() = std::cos(latRad) * std::sin(lonRad);
    position.z() = std::sin(latRad);
    
    position *= grid.getRadius();
    
    // Find closest vertex on the sphere grid
    int closestVertexIdx = grid.findClosestVertex(position);
    
    // Return elevation
    if (closestVertexIdx >= 0) {
        return planet.getElevation(closestVertexIdx);
    }
    
    return 0.0f; // Default if no vertex found
}

float HeightmapGenerator::generateNoise(float x, float y) const {
    // Simple Perlin-like noise implementation
    // In a real implementation, you'd want to use a proper noise library like libnoise or FastNoise
    
    // For this example, we'll use a very simplified noise function
    float noise = 0.0f;
    float amplitude = 1.0f;
    float frequency = params.noiseScale;
    
    for (int i = 0; i < params.noiseOctaves; ++i) {
        float nx = x * frequency;
        float ny = y * frequency;
        
        // Simple hash function to get pseudo-random value
        int ix = static_cast<int>(nx) & 255;
        int iy = static_cast<int>(ny) & 255;
        float fx = nx - static_cast<int>(nx);
        float fy = ny - static_cast<int>(ny);
        
        float n = (ix * 1299721 + iy * 7919) % 1000 / 1000.0f;
        
        // Bilinear interpolation
        float sx = fx * fx * (3.0f - 2.0f * fx);
        float sy = fy * fy * (3.0f - 2.0f * fy);
        noise += n * amplitude;
        
        amplitude *= params.noisePersistence;
        frequency *= params.noiseLacunarity;
    }
    
    return noise * params.noiseStrength;
}

void HeightmapGenerator::applyTerrainFeatures() {
    // Apply additional terrain features to the base heightmap
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float elevation = heightmapData[idx];
            
            // Apply noise based on current elevation
            float nx = static_cast<float>(x) / width * 100.0f;
            float ny = static_cast<float>(y) / height * 100.0f;
            float noise = generateNoise(nx, ny);
            
            // Mountains are more rugged
            if (elevation > 1.0f) {
                float mountainFactor = (elevation - 1.0f) / 9.0f; // Scale from 0-1 for elevations 1-10km
                noise *= (1.0f + mountainFactor * params.mountainRoughness * 2.0f);
            }
            
            // Plateaus are flatter
            if (elevation > 0.5f && elevation < 2.0f) {
                float plateauFactor = 1.0f - std::abs(elevation - 1.25f) / 0.75f;
                if (plateauFactor > 0) {
                    noise *= (1.0f - plateauFactor * params.plateauFlatness);
                }
            }
            
            // Continental shelves have distinct drop-offs
            if (elevation < 0.0f && elevation > -0.5f) {
                float shelfFactor = 1.0f - std::abs(elevation - params.continentalShelfDepth) / 0.5f;
                if (shelfFactor > 0) {
                    noise *= 0.3f; // Less noise on continental shelves
                }
            }
            
            // Apply the noise
            heightmapData[idx] += noise;
        }
    }
}

void HeightmapGenerator::applyTectonicFeatures() {
    if (!tectonics) return;
    
    // Get plate boundaries
    std::vector<int> boundaryVertices = tectonics->getBoundaryVertices();
    const SphericalGrid& grid = planet.getGrid();
    
    // Build a mask of plate boundaries
    std::vector<bool> isBoundary(planet.getVertexCount(), false);
    for (int vertexIdx : boundaryVertices) {
        isBoundary[vertexIdx] = true;
    }
    
    // Apply tectonic features
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Convert to lat/lon
            float lon = (static_cast<float>(x) / width) * 360.0f - 180.0f;
            float lat = 90.0f - (static_cast<float>(y) / height) * 180.0f;
            
            // Convert to 3D position
            Vec3f position;
            float latRad = toRadians(lat);
            float lonRad = toRadians(lon);
            
            position.x() = std::cos(latRad) * std::cos(lonRad);
            position.y() = std::cos(latRad) * std::sin(lonRad);
            position.z() = std::sin(latRad);
            
            position *= grid.getRadius();
            
            // Find closest vertex
            int closestVertexIdx = grid.findClosestVertex(position);
            int idx = y * width + x;
            
            if (closestVertexIdx >= 0) {
                // Check if this is near a plate boundary
                if (isBoundary[closestVertexIdx]) {
                    // Find the neighboring vertex on a different plate
                    const auto& neighbors = grid.getNeighbors(closestVertexIdx);
                    for (int neighborIdx : neighbors) {
                        if (planet.getPlateId(neighborIdx) != planet.getPlateId(closestVertexIdx)) {
                            // Determine interaction type
                            PlateInteractionType interaction = tectonics->getInteractionType(
                                closestVertexIdx, neighborIdx);
                            
                            // Apply different effects based on interaction type
                            switch (interaction) {
                                case PlateInteractionType::Divergent:
                                    // Rift valleys or mid-ocean ridges
                                    if (heightmapData[idx] < 0.0f) {
                                        // Mid-ocean ridge
                                        heightmapData[idx] += 0.5f * params.tectonicInfluence;
                                    } else {
                                        // Rift valley
                                        heightmapData[idx] -= 0.7f * params.tectonicInfluence;
                                    }
                                    break;
                                    
                                case PlateInteractionType::Convergent:
                                case PlateInteractionType::Collision:
                                    // Mountain building
                                    heightmapData[idx] += 1.5f * params.tectonicInfluence;
                                    break;
                                    
                                case PlateInteractionType::Subduction:
                                    // Deep ocean trenches next to mountains
                                    if (heightmapData[idx] < 0.0f) {
                                        // Ocean trench
                                        heightmapData[idx] -= 2.0f * params.tectonicInfluence;
                                    } else {
                                        // Volcanic arc
                                        heightmapData[idx] += 1.2f * params.tectonicInfluence;
                                    }
                                    break;
                                    
                                case PlateInteractionType::Transform:
                                    // Strike-slip fault features
                                    float offset = (static_cast<float>(x + y) / (width + height));
                                    offset = std::sin(offset * 10.0f) * params.faultDisplacement;
                                    heightmapData[idx] += offset * params.tectonicInfluence;
                                    break;
                            }
                            
                            break; // Found a boundary, no need to check other neighbors
                        }
                    }
                }
            }
        }
    }
}

void HeightmapGenerator::applyClimateInfluence() {
    if (!tectonics) return; // Need tectonics for climate data
    
    const SphericalGrid& grid = planet.getGrid();
    
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Convert to lat/lon
            float lon = (static_cast<float>(x) / width) * 360.0f - 180.0f;
            float lat = 90.0f - (static_cast<float>(y) / height) * 180.0f;
            
            // Find closest vertex
            Vec3f position;
            float latRad = toRadians(lat);
            float lonRad = toRadians(lon);
            
            position.x() = std::cos(latRad) * std::cos(lonRad);
            position.y() = std::cos(latRad) * std::sin(lonRad);
            position.z() = std::sin(latRad);
            
            position *= grid.getRadius();
            
            int closestVertexIdx = grid.findClosestVertex(position);
            int idx = y * width + x;
            
            if (closestVertexIdx >= 0) {
                // Access climate data from the vertex
                const auto& climate = tectonics->getClimateSystem();
                float precipitation = climate.precipitation[closestVertexIdx];
                float weatheringRate = climate.weatheringRate[closestVertexIdx];
                
                // Adjust elevation based on climate factors
                if (heightmapData[idx] > 0.0f) {
                    // Higher precipitation causes more erosion on land
                    float erosionFactor = (precipitation / 2000.0f) * weatheringRate;
                    heightmapData[idx] -= erosionFactor * params.precipitationErosion * params.climateInfluence;
                }
                
                // Check if this is a river
                const auto& hydro = tectonics->getHydrologicalSystem();
                if (hydro.isRiver[closestVertexIdx]) {
                    // Carve river valleys
                    float riverDepth = 0.2f * params.riverErosionFactor * std::min(1.0f, hydro.riverFlow[closestVertexIdx] / 100.0f);
                    heightmapData[idx] -= riverDepth * params.climateInfluence;
                }
                
                // Lakes flatten terrain
                if (hydro.isLake[closestVertexIdx] && heightmapData[idx] > 0.0f) {
                    // Make lake basins flatter
                    heightmapData[idx] = std::min(heightmapData[idx], 0.1f);
                }
            }
        }
    }
}

void HeightmapGenerator::applyErosion() {
    // Simple thermal and hydraulic erosion simulation
    std::vector<float> tempHeightmap = heightmapData;
    
    // Run multiple iterations
    for (int iteration = 0; iteration < params.erosionIterations; ++iteration) {
        // Thermal erosion (material slumps down steep slopes)
        #pragma omp parallel for
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;
                float currentHeight = tempHeightmap[idx];
                
                // Check all 8 neighbors
                float lowestNeighbor = currentHeight;
                int lowestIdx = -1;
                
                for (int ny = -1; ny <= 1; ++ny) {
                    for (int nx = -1; nx <= 1; ++nx) {
                        if (nx == 0 && ny == 0) continue; // Skip current cell
                        
                        int neighborIdx = (y + ny) * width + (x + nx);
                        float neighborHeight = tempHeightmap[neighborIdx];
                        
                        if (neighborHeight < lowestNeighbor) {
                            lowestNeighbor = neighborHeight;
                            lowestIdx = neighborIdx;
                        }
                    }
                }
                
                // If there's a lower neighbor, move some material
                if (lowestIdx >= 0) {
                    float slope = currentHeight - lowestNeighbor;
                    float threshold = 0.01f + 0.05f * params.erosionStrength;
                    
                    if (slope > threshold) {
                        float transferAmount = (slope - threshold) * 0.5f * params.erosionStrength;
                        
                        // Apply in the temp heightmap for the next iteration
                        heightmapData[idx] -= transferAmount;
                        heightmapData[lowestIdx] += transferAmount;
                    }
                }
            }
        }
        
        // Update temp heightmap for next iteration
        tempHeightmap = heightmapData;
    }
}

void HeightmapGenerator::normalizeTerrain() {
    // Find current min/max elevation
    float minElevation = std::numeric_limits<float>::max();
    float maxElevation = std::numeric_limits<float>::lowest();
    
    for (float height : heightmapData) {
        minElevation = std::min(minElevation, height);
        maxElevation = std::max(maxElevation, height);
    }
    
    // Adjust to target range while preserving sea level
    float seaLevel = params.oceanLevel;
    
    if (maxElevation > seaLevel && minElevation < seaLevel) {
        float landScale = (params.maxElevation - seaLevel) / (maxElevation - seaLevel);
        float oceanScale = (seaLevel - params.minElevation) / (seaLevel - minElevation);
        
        #pragma omp parallel for
        for (size_t i = 0; i < heightmapData.size(); ++i) {
            if (heightmapData[i] >= seaLevel) {
                // Scale land
                heightmapData[i] = seaLevel + (heightmapData[i] - seaLevel) * landScale;
            } else {
                // Scale ocean
                heightmapData[i] = seaLevel - (seaLevel - heightmapData[i]) * oceanScale;
            }
        }
    } else {
        // No sea level in the data, do a simple rescale
        float scale = (params.maxElevation - params.minElevation) / (maxElevation - minElevation);
        float offset = params.minElevation - minElevation * scale;
        
        #pragma omp parallel for
        for (size_t i = 0; i < heightmapData.size(); ++i) {
            heightmapData[i] = heightmapData[i] * scale + offset;
        }
    }
}

std::vector<Vec3f> HeightmapGenerator::generateNormalMap() {
    std::vector<Vec3f> normalMap(width * height);
    
    #pragma omp parallel for
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            
            // Sample surrounding heights
            float hL = heightmapData[y * width + (x - 1)];      // left
            float hR = heightmapData[y * width + (x + 1)];      // right
            float hU = heightmapData[(y - 1) * width + x];      // up
            float hD = heightmapData[(y + 1) * width + x];      // down
            
            // Calculate the normal vector
            Vec3f dx(2.0f, 0.0f, (hR - hL));
            Vec3f dy(0.0f, 2.0f, (hD - hU));
            
            // Cross product
            Vec3f normal = dx.cross(dy);
            normal.normalize();
            
            normalMap[idx] = normal;
        }
    }
    
    // Handle edges
    for (int y = 0; y < height; ++y) {
        normalMap[y * width] = normalMap[y * width + 1];
        normalMap[y * width + width - 1] = normalMap[y * width + width - 2];
    }
    
    for (int x = 0; x < width; ++x) {
        normalMap[x] = normalMap[width + x];
        normalMap[(height - 1) * width + x] = normalMap[(height - 2) * width + x];
    }
    
    return normalMap;
}

std::vector<float> HeightmapGenerator::generateSlopeMap() {
    std::vector<float> slopeMap(width * height);
    
    #pragma omp parallel for
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            
            // Sample surrounding heights
            float hL = heightmapData[y * width + (x - 1)];      // left
            float hR = heightmapData[y * width + (x + 1)];      // right
            float hU = heightmapData[(y - 1) * width + x];      // up
            float hD = heightmapData[(y + 1) * width + x];      // down
            
            // Calculate gradient
            float dx = (hR - hL) / 2.0f;
            float dy = (hD - hU) / 2.0f;
            
            // Calculate slope (0-1 range)
            float slope = std::sqrt(dx * dx + dy * dy);
            slope = std::min(1.0f, slope * 5.0f); // Scale to make more visible
            
            slopeMap[idx] = slope;
        }
    }
    
    // Handle edges
    for (int y = 0; y < height; ++y) {
        slopeMap[y * width] = slopeMap[y * width + 1];
        slopeMap[y * width + width - 1] = slopeMap[y * width + width - 2];
    }
    
    for (int x = 0; x < width; ++x) {
        slopeMap[x] = slopeMap[width + x];
        slopeMap[(height - 1) * width + x] = slopeMap[(height - 2) * width + x];
    }
    
    return slopeMap;
}

bool HeightmapGenerator::saveToFile(const std::string& filename, const std::string& format) {
    // This is a placeholder for file saving functionality
    // In a real implementation, you'd use a proper image library like stb_image or libpng
    
    if (format == "raw") {
        // Save as raw 32-bit float data
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        file.write(reinterpret_cast<const char*>(heightmapData.data()), 
                  heightmapData.size() * sizeof(float));
        
        return file.good();
    } else {
        std::cerr << "Unsupported format: " << format << std::endl;
        std::cerr << "Only 'raw' format is implemented in this example." << std::endl;
        return false;
    }
}

int HeightmapGenerator::getWidth() const {
    return width;
}

int HeightmapGenerator::getHeight() const {
    return height;
}

const std::vector<float>& HeightmapGenerator::getData() const {
    return heightmapData;
}

void HeightmapGenerator::setTerrainModifier(std::function<float(float, float, float)> modifier) {
    terrainModifier = modifier;
}

} // namespace aeonterracpp