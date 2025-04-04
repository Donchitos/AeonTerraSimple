#include "aeonterracpp/core/tectonics.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_set>

namespace aeonterracpp {

//-----------------------------------------------------------------------------
// TectonicPlate implementation
//-----------------------------------------------------------------------------

TectonicPlate::TectonicPlate(int id, const Vec3f& center)
    : id(id), center(center), movement(Vec3f::Zero()), oceanic(false) {
}

int TectonicPlate::getId() const {
    return id;
}

const Vec3f& TectonicPlate::getCenter() const {
    return center;
}

const Vec3f& TectonicPlate::getMovement() const {
    return movement;
}

void TectonicPlate::setMovement(const Vec3f& movement) {
    this->movement = movement;
}

const std::vector<int>& TectonicPlate::getVertices() const {
    return vertices;
}

void TectonicPlate::addVertex(int vertexIdx) {
    vertices.push_back(vertexIdx);
}

float TectonicPlate::calculateArea() const {
    return static_cast<float>(vertices.size());  // Simple approximation
}

bool TectonicPlate::isOceanic() const {
    return oceanic;
}

void TectonicPlate::setOceanic(bool oceanic) {
    this->oceanic = oceanic;
}

BoundingBox TectonicPlate::calculateBoundingBox() const {
    // Default values
    float latMin = 90.0f;
    float latMax = -90.0f;
    float lonMin = 180.0f;
    float lonMax = -180.0f;
    
    // We would need to access vertex positions, which requires planet grid
    // For now, return an empty bounding box
    return BoundingBox(latMin, lonMin, latMax, lonMax);
}

//-----------------------------------------------------------------------------
// ConvectionCell implementation
//-----------------------------------------------------------------------------

ConvectionCell::ConvectionCell(const Vec3f& center, float strength)
    : center(center), strength(strength), radius(M_PI / 4.0f) {  // ~45 degree radius
}

const Vec3f& ConvectionCell::getCenter() const {
    return center;
}

float ConvectionCell::getStrength() const {
    return strength;
}

Vec3f ConvectionCell::calculateForceAt(const Vec3f& position) const {
    // Calculate vector from position to cell center
    Vec3f toCenter = center - position;
    
    // Calculate distance
    float distance = toCenter.norm();
    
    // Calculate direction
    Vec3f direction = toCenter.normalized();
    
    // Calculate force magnitude based on distance
    // Force decreases with square of distance and is limited to the cell radius
    float forceMagnitude = 0.0f;
    if (distance < radius) {
        // Smoothly interpolate force from center to edge of cell
        // Maximum at center, zero at radius
        float falloff = 1.0f - (distance / radius) * (distance / radius);
        
        // Force direction depends on cell type (upwelling or downwelling)
        forceMagnitude = strength * falloff;
        
        // For upwelling (positive strength), force is away from center
        // For downwelling (negative strength), force is toward center
        if (strength > 0) {
            direction = -direction; // Reverse direction for upwelling
        }
    }
    
    return direction * forceMagnitude;
}

//-----------------------------------------------------------------------------
// TectonicSimulation implementation
//-----------------------------------------------------------------------------

TectonicSimulation::TectonicSimulation(Planet& planet, int numPlates, unsigned int seed)
    : planet(planet), rng(seed != 0 ? seed : std::random_device()()) {
    
    // Set default parameters
    params.mantleDensity = 3.3f;         // g/cm³
    params.oceanicDensity = 3.0f;        // g/cm³
    params.continentalDensity = 2.7f;    // g/cm³
    params.tectonicActivityScale = 0.5f; // Medium activity
    params.erosionRate = 0.05f;          // km/Myr
    params.sedimentationRate = 0.02f;    // km/Myr
    
    // Initialize new parameters
    params.superContinentCyclePhase = 0.0f;    // Start at beginning of cycle
    params.activityVariabilityPeriod = 100.0f; // 100 million year activity cycle
    params.riverFlowRate = 1.0f;               // Default river flow rate
    params.weatheringRate = 0.02f;             // Chemical weathering rate
    params.enableClimateEffects = true;        // Enable climate effects by default
    
    // Initialize plates
    initializePlates();
    
    // Initialize mantle
    initializeMantle(numPlates);
    
    // Initialize mass balance tracking
    initializeMassBalance();
    
    // Initialize climate system
    climate.initialize(planet.getVertexCount());
    
    // Initialize hydrological system
    hydrology.initialize(planet.getVertexCount());
    
    // Calculate initial climate conditions
    if (params.enableClimateEffects) {
        climate.update(planet.getGrid(), planet.getVertexCount());
    }
    
    std::cout << "Initialized enhanced tectonic simulation with climate and hydrological systems" << std::endl;
}

// Initialize mass balance tracking
void TectonicSimulation::initializeMassBalance() {
    massBalance.reset();
    
    // Calculate initial crust volumes
    const SphericalGrid& grid = planet.getGrid();
    float cellArea = 4.0f * M_PI * planet.getRadius() * planet.getRadius() / grid.getVertexCount();
    
    for (size_t i = 0; i < grid.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        bool isOceanic = planet.getCrustType(vertexIdx) == 0;
        float thickness = planet.getCrustThickness(vertexIdx);
        
        // Calculate volume of this cell's crust
        float cellVolume = cellArea * thickness;
        
        // Add to appropriate total
        if (isOceanic) {
            massBalance.initialOceanicVolume += cellVolume;
        } else {
            massBalance.initialContinentalVolume += cellVolume;
        }
    }
    
    std::cout << "Initial continental crust volume: " << massBalance.initialContinentalVolume << " km³" << std::endl;
    std::cout << "Initial oceanic crust volume: " << massBalance.initialOceanicVolume << " km³" << std::endl;
}

// Update the climate system based on elevation and latitude
void TectonicSimulation::ClimateSystem::update(const SphericalGrid& grid, size_t vertexCount) {
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Get position and convert to latitude/longitude
        Vec3f pos = grid.getVertex(vertexIdx);
        auto [lat, lon] = grid.cartesianToLatLon(pos);
        
        // Temperature varies with latitude (simplified model)
        // Highest at equator, lowest at poles
        float tempLatFactor = std::cos(toRadians(lat));
        temperature[i] = 30.0f * tempLatFactor * tempLatFactor - 5.0f;
        
        // Elevation affects temperature (-6.5°C per km)
        float elevation = 0.0f; // Need to get this from planet
        temperature[i] -= elevation * 6.5f;
        
        // Precipitation varies with latitude (simplistic model)
        // Highest at equator and mid-latitudes, lower at horse latitudes and poles
        float latAbs = std::abs(lat);
        float precipFactor = 0.0f;
        
        if (latAbs < 10.0f) {
            // Equatorial (ITCZ)
            precipFactor = 1.0f;
        } else if (latAbs < 30.0f) {
            // Horse latitudes (dry)
            precipFactor = 0.3f;
        } else if (latAbs < 60.0f) {
            // Mid-latitudes (moderately wet)
            precipFactor = 0.7f;
        } else {
            // Polar (dry)
            precipFactor = 0.2f;
        }
        
        precipitation[i] = 2000.0f * precipFactor;
        
        // Calculate weathering rate based on temperature and precipitation
        // Chemical weathering increases with temperature and precipitation
        weatheringRate[i] = 0.01f +
                         (temperature[i] + 20.0f) / 50.0f * 0.5f +
                         precipitation[i] / 2000.0f * 0.5f;
    }
}

void TectonicSimulation::initializePlates() {
    // Clear existing plates
    plates.clear();
    
    // Number of plates to create
    int numPlates = 7;  // Default number
    
    // Get planet grid to access vertex positions
    const SphericalGrid& grid = planet.getGrid();
    size_t vertexCount = grid.getVertexCount();
    
    // Create seed points for plates using Fibonacci sphere method
    std::vector<Vec3f> seedPoints;
    
    // Create evenly distributed points on the sphere
    float goldenRatio = (1.0f + std::sqrt(5.0f)) / 2.0f;
    float angleIncrement = 2.0f * M_PI / goldenRatio;
    
    for (int i = 0; i < numPlates; ++i) {
        float t = static_cast<float>(i) / numPlates;
        float inclination = std::acos(1.0f - 2.0f * t);
        float azimuth = angleIncrement * i;
        
        // Convert to Cartesian coordinates
        float x = std::sin(inclination) * std::cos(azimuth);
        float y = std::sin(inclination) * std::sin(azimuth);
        float z = std::cos(inclination);
        
        Vec3f point(x, y, z);
        point *= grid.getRadius();  // Scale to planet radius
        
        seedPoints.push_back(point);
    }
    
    // Create plates from seed points
    for (int i = 0; i < numPlates; ++i) {
        TectonicPlate plate(i, seedPoints[i]);
        
        // Determine plate type (oceanic or continental)
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        bool isOceanic = dist(rng) < 0.7f;  // 70% chance of oceanic plates
        plate.setOceanic(isOceanic);
        
        plates.push_back(plate);
    }
    
    // Assign vertices to nearest plate
    std::vector<int> unassignedVertices;
    for (size_t i = 0; i < vertexCount; ++i) {
        unassignedVertices.push_back(static_cast<int>(i));
    }
    
    // Shuffle to avoid bias
    std::shuffle(unassignedVertices.begin(), unassignedVertices.end(), rng);
    
    // Assign a few seed vertices to each plate
    const int initialSeedsPerPlate = 5;
    std::vector<std::vector<int>> plateSeeds(numPlates);
    
    for (int i = 0; i < numPlates * initialSeedsPerPlate && !unassignedVertices.empty(); ++i) {
        int plateIdx = i % numPlates;
        int vertexIdx = unassignedVertices.back();
        unassignedVertices.pop_back();
        
        // Add vertex to plate
        plates[plateIdx].addVertex(vertexIdx);
        planet.setPlateId(vertexIdx, plateIdx);
        
        // Add to seeds for growth
        plateSeeds[plateIdx].push_back(vertexIdx);
    }
    
    // Grow plates simultaneously by assigning neighbors of assigned vertices
    bool continueGrowing = true;
    while (continueGrowing && !unassignedVertices.empty()) {
        continueGrowing = false;
        
        // Process one layer of neighbors for each plate
        for (int plateIdx = 0; plateIdx < numPlates; ++plateIdx) {
            std::vector<int> newSeeds;
            
            // Process all current seeds for this plate
            for (int seed : plateSeeds[plateIdx]) {
                // Get neighbors of this vertex
                auto neighbors = grid.getNeighbors(seed);
                
                // Check each neighbor
                for (int neighbor : neighbors) {
                    // Check if already assigned to a plate
                    if (planet.getPlateId(neighbor) == -1) {
                        // Assign to current plate
                        plates[plateIdx].addVertex(neighbor);
                        planet.setPlateId(neighbor, plateIdx);
                        
                        // Add as new seed for next iteration
                        newSeeds.push_back(neighbor);
                        
                        // Remove from unassigned list
                        auto it = std::find(unassignedVertices.begin(), unassignedVertices.end(), neighbor);
                        if (it != unassignedVertices.end()) {
                            unassignedVertices.erase(it);
                        }
                        
                        continueGrowing = true;
                    }
                }
            }
            
            // Update seeds for next iteration
            plateSeeds[plateIdx] = newSeeds;
        }
    }
    
    // If any vertices remain unassigned, assign them to the closest plate
    for (int vertexIdx : unassignedVertices) {
        int plateId = identifyClosestPlate(vertexIdx);
        if (plateId >= 0) {
            plates[plateId].addVertex(vertexIdx);
            planet.setPlateId(vertexIdx, plateId);
        }
    }
    
    // Set initial plate properties based on type
    for (const auto& plate : plates) {
        for (int vertexIdx : plate.getVertices()) {
            if (plate.isOceanic()) {
                // Oceanic crust is thinner and denser
                planet.setCrustType(vertexIdx, 0);
                planet.setCrustThickness(vertexIdx, 7.0f);
                planet.setCrustDensity(vertexIdx, params.oceanicDensity);
                
                // Oceanic plates are below sea level
                planet.setElevation(vertexIdx, -4.0f);
            } else {
                // Continental crust is thicker and less dense
                planet.setCrustType(vertexIdx, 1);
                planet.setCrustThickness(vertexIdx, 35.0f);
                planet.setCrustDensity(vertexIdx, params.continentalDensity);
                
                // Continental plates are above sea level
                planet.setElevation(vertexIdx, 0.5f);
            }
        }
    }
    
    // Apply initial isostatic adjustments
    applyIsostaticAdjustments();
    
    std::cout << "Initialized " << numPlates << " tectonic plates" << std::endl;
}

void TectonicSimulation::initializeMantle(int numCells) {
    // Clear existing cells
    convectionCells.clear();
    
    // Ensure reasonable number of cells
    if (numCells <= 0) {
        numCells = plates.size() + 2; // More cells than plates for dynamic flow
    }
    
    // Get planet radius
    float radius = planet.getRadius();
    
    // Create pairs of upwelling/downwelling cells for more realistic convection patterns
    int numPairs = numCells / 2;
    std::uniform_real_distribution<float> strengthDist(0.5f, 1.5f); // Stronger forces
    std::uniform_real_distribution<float> offsetDist(-0.2f, 0.2f);  // Random offset
    
    // Setup convection cell pairs similar to Earth's mantle plumes
    for (int i = 0; i < numPairs; ++i) {
        // Create a pair of cells - one upwelling and one downwelling
        // Distributed evenly but with some randomness
        
        // Primary position - upwelling (hot spot)
        float t1 = static_cast<float>(i) / numPairs;
        float inclination1 = std::acos(1.0f - 2.0f * t1) + offsetDist(rng);
        float azimuth1 = 2.0f * M_PI * t1 + offsetDist(rng);
        
        // Secondary position - downwelling (subduction zones)
        // Roughly opposite the upwelling
        float inclination2 = M_PI - inclination1 + offsetDist(rng);
        float azimuth2 = azimuth1 + M_PI + offsetDist(rng);
        
        // Convert to Cartesian coordinates - upwelling
        float x1 = std::sin(inclination1) * std::cos(azimuth1);
        float y1 = std::sin(inclination1) * std::sin(azimuth1);
        float z1 = std::cos(inclination1);
        Vec3f position1(x1, y1, z1);
        position1 *= radius;
        
        // Convert to Cartesian coordinates - downwelling
        float x2 = std::sin(inclination2) * std::cos(azimuth2);
        float y2 = std::sin(inclination2) * std::sin(azimuth2);
        float z2 = std::cos(inclination2);
        Vec3f position2(x2, y2, z2);
        position2 *= radius;
        
        // Random strength - upwelling is positive, downwelling is negative
        float strength1 = strengthDist(rng);
        float strength2 = -strengthDist(rng);
        
        // Create cells at both positions
        convectionCells.emplace_back(position1, strength1);
        convectionCells.emplace_back(position2, strength2);
    }
    
    // Create additional cells to break symmetry if needed
    if (numCells % 2 != 0) {
        // Random position for the odd cell
        std::uniform_real_distribution<float> posDist(-1.0f, 1.0f);
        Vec3f randomPos(posDist(rng), posDist(rng), posDist(rng));
        randomPos.normalize();
        randomPos *= radius;
        
        // Random strength
        float strength = strengthDist(rng) * (posDist(rng) > 0 ? 1 : -1);
        
        convectionCells.emplace_back(randomPos, strength);
    }
    
    std::cout << "Initialized " << convectionCells.size() << " mantle convection cells" << std::endl;
    std::cout << "  Includes " << numPairs << " upwelling/downwelling pairs" << std::endl;
}

void TectonicSimulation::simulateStep(double timeStep) {
    // Update supercontinent cycle phase (complete cycle is ~500 million years)
    params.superContinentCyclePhase += static_cast<float>(timeStep) / 500.0f;
    if (params.superContinentCyclePhase > 1.0f) {
        params.superContinentCyclePhase -= 1.0f;
        std::cout << "Completed a full supercontinent cycle" << std::endl;
    }
    
    // Calculate tectonic activity based on cycle phase and variability period
    float baseActivity = params.tectonicActivityScale;
    float ageInCycle = static_cast<float>(std::fmod(planet.getAge(), params.activityVariabilityPeriod));
    float cycleFactor = std::sin(2.0f * static_cast<float>(M_PI) * ageInCycle / params.activityVariabilityPeriod);
    float adjustedActivity = baseActivity * (1.0f + 0.2f * cycleFactor);
    
    // Temporarily adjust tectonic activity
    float originalActivity = params.tectonicActivityScale;
    params.tectonicActivityScale = adjustedActivity;
    
    // Modify plate motion tendencies based on supercontinent cycle
    applySuperContinentCycleForces();
    
    // Calculate plate movements based on convection forces
    calculatePlateMovements(timeStep);
    
    // Process interactions between plates
    processPlateInteractions();
    
    // Update climate system
    if (params.enableClimateEffects) {
        climate.update(planet.getGrid(), planet.getVertexCount());
    }
    
    // Calculate hydrological processes (river networks)
    calculateHydrologicalProcesses(timeStep);
    
    // Apply erosion and sedimentation (enhanced with climate effects)
    applyErosionAndSedimentation(timeStep);
    
    // Apply isostatic adjustments
    applyIsostaticAdjustments();
    
    // Restore original activity level
    params.tectonicActivityScale = originalActivity;
    
    // Advance planet age
    planet.advanceAge(timeStep);
    
    // Report on continental growth
    printContinentalGrowthStats();
    
    std::cout << "Simulated " << timeStep << " million years of tectonic activity" << std::endl;
    std::cout << "Planet age: " << planet.getAge() << " million years" << std::endl;
    std::cout << "Supercontinent cycle phase: " << (params.superContinentCyclePhase * 100.0f) << "%" << std::endl;
}

void TectonicSimulation::calculateHydrologicalProcesses(double timeStep) {
    const SphericalGrid& grid = planet.getGrid();
    size_t vertexCount = grid.getVertexCount();
    
    // Reset drainage and flow data
    for (size_t i = 0; i < vertexCount; ++i) {
        hydrology.drainage[i] = 0.0f;
        hydrology.riverFlow[i] = 0.0f;
        hydrology.isRiver[i] = false;
        hydrology.isLake[i] = false;
    }
    
    // Calculate flow directions - water flows to lowest neighbor
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        
        // Find lowest neighbor
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        hydrology.flowDirection[i] = -1; // Default: no outflow (sink)
        float lowestElevation = elevation;
        
        for (int neighborIdx : neighbors) {
            float neighborElevation = planet.getElevation(neighborIdx);
            if (neighborElevation < lowestElevation) {
                lowestElevation = neighborElevation;
                hydrology.flowDirection[i] = neighborIdx;
            }
        }
        
        // Calculate initial drainage area (base unit is one cell)
        hydrology.drainage[i] = 1.0f;
    }
    
    // Calculate drainage area by propagating flow downstream
    // Use topological sort approach - process from highest to lowest
    struct DrainageVertex {
        int index;
        float elevation;
        
        bool operator<(const DrainageVertex& other) const {
            return elevation > other.elevation; // Higher elevations first
        }
    };
    
    std::vector<DrainageVertex> sortedVertices;
    sortedVertices.reserve(vertexCount);
    
    for (size_t i = 0; i < vertexCount; ++i) {
        sortedVertices.push_back({static_cast<int>(i), planet.getElevation(static_cast<int>(i))});
    }
    
    std::sort(sortedVertices.begin(), sortedVertices.end());
    
    // Propagate drainage area downstream
    for (const auto& vertex : sortedVertices) {
        int vertexIdx = vertex.index;
        int flowTarget = hydrology.flowDirection[vertexIdx];
        
        if (flowTarget >= 0) {
            // Add this cell's drainage to the downstream cell
            hydrology.drainage[flowTarget] += hydrology.drainage[vertexIdx];
        }
    }
    
    // Calculate river flows and identify rivers
    float riverThreshold = 10.0f; // Cells with drainage > 10 become rivers
    
    for (size_t i = 0; i < vertexCount; ++i) {
        // River flow proportional to drainage area and precipitation
        if (params.enableClimateEffects) {
            hydrology.riverFlow[i] = hydrology.drainage[i] * climate.precipitation[i] / 1000.0f * params.riverFlowRate;
        } else {
            hydrology.riverFlow[i] = hydrology.drainage[i] * params.riverFlowRate;
        }
        
        // Mark as river if flow exceeds threshold
        if (hydrology.riverFlow[i] > riverThreshold) {
            hydrology.isRiver[i] = true;
            
            // Rivers enhance erosion where they flow
            float elevation = planet.getElevation(static_cast<int>(i));
            if (elevation > 0.0f) {
                // Enhanced fluvial erosion - rivers cut valleys
                float erosionFactor = std::min(5.0f, std::log10(hydrology.riverFlow[i]));
                float additionalErosion = params.erosionRate * erosionFactor * 0.1f * static_cast<float>(timeStep);
                planet.setElevation(static_cast<int>(i), elevation - additionalErosion);
            }
        }
    }
    
    // Identify lakes (depressions with no outflow)
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        
        // Check if this is a sink (no outflow) above sea level
        if (hydrology.flowDirection[i] == -1 && elevation > 0.0f) {
            hydrology.isLake[i] = true;
            
            // Lakes have flat elevation
            const auto& neighbors = grid.getNeighbors(vertexIdx);
            for (int neighborIdx : neighbors) {
                if (planet.getElevation(neighborIdx) > elevation) {
                    // Set lake level
                    planet.setElevation(neighborIdx, elevation);
                }
            }
        }
    }
    
    // Count rivers and lakes for reporting
    int riverCount = std::count(hydrology.isRiver.begin(), hydrology.isRiver.end(), true);
    int lakeCount = std::count(hydrology.isLake.begin(), hydrology.isLake.end(), true);
    
    if (riverCount > 0 || lakeCount > 0) {
        std::cout << "Hydrological features: " << riverCount << " river cells, " << lakeCount << " lake cells" << std::endl;
    }
}

void TectonicSimulation::printContinentalGrowthStats() {
    // Get current crustal volumes
    const SphericalGrid& grid = planet.getGrid();
    double currentOceanicVolume = 0.0;
    double currentContinentalVolume = 0.0;
    float cellArea = 4.0f * M_PI * planet.getRadius() * planet.getRadius() / grid.getVertexCount();
    
    for (size_t i = 0; i < grid.getVertexCount(); ++i) {
        int vertexIdx = static_cast<int>(i);
        bool isOceanic = planet.getCrustType(vertexIdx) == 0;
        float thickness = planet.getCrustThickness(vertexIdx);
        
        // Calculate volume of this cell's crust
        float cellVolume = cellArea * thickness;
        
        // Add to appropriate total
        if (isOceanic) {
            currentOceanicVolume += cellVolume;
        } else {
            currentContinentalVolume += cellVolume;
        }
    }
    
    // Calculate growth percentages
    double oceanicGrowthPct = (currentOceanicVolume - massBalance.initialOceanicVolume) /
                              massBalance.initialOceanicVolume * 100.0;
    double continentalGrowthPct = (currentContinentalVolume - massBalance.initialContinentalVolume) /
                                   massBalance.initialContinentalVolume * 100.0;
    
    // Print stats periodically (every 50 million years)
    double age = planet.getAge();
    if (static_cast<int>(age) % 50 == 0 && age > 0.0) {
        std::cout << "--------- Crustal Balance at " << age << " Myr ---------" << std::endl;
        std::cout << "Continental crust: " << currentContinentalVolume << " km³ ("
                  << (continentalGrowthPct >= 0 ? "+" : "") << continentalGrowthPct << "%)" << std::endl;
        std::cout << "Oceanic crust: " << currentOceanicVolume << " km³ ("
                  << (oceanicGrowthPct >= 0 ? "+" : "") << oceanicGrowthPct << "%)" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }
}
    
    // Calculate hydrological processes (river networks)
    calculateHydrologicalProcesses(timeStep);
    
    // Apply erosion and sedimentation (enhanced with climate effects)
    applyErosionAndSedimentation(timeStep);
    
    // Apply isostatic adjustments
    applyIsostaticAdjustments();
    
    // Restore original activity level
    params.tectonicActivityScale = originalActivity;
    
    // Advance planet age
    planet.advanceAge(timeStep);
    
    // Report on continental growth
    printContinentalGrowthStats();
    
    std::cout << "Simulated " << timeStep << " million years of tectonic activity" << std::endl;
    std::cout << "Planet age: " << planet.getAge() << " million years" << std::endl;
    std::cout << "Supercontinent cycle phase: " << (params.superContinentCyclePhase * 100.0f) << "%" << std::endl;
}

std::vector<int> TectonicSimulation::getBoundaryVertices() const {
    std::vector<int> boundaryVertices;
    
    const SphericalGrid& grid = planet.getGrid();
    size_t vertexCount = grid.getVertexCount();
    
    // Check each vertex to see if any of its neighbors belong to a different plate
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        int plateId = planet.getPlateId(vertexIdx);
        
        // Skip unassigned vertices
        if (plateId < 0) continue;
        
        // Get neighbors
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        
        // Check if any neighbor belongs to a different plate
        bool isBoundary = false;
        for (int neighborIdx : neighbors) {
            int neighborPlateId = planet.getPlateId(neighborIdx);
            if (neighborPlateId >= 0 && neighborPlateId != plateId) {
                isBoundary = true;
                break;
            }
        }
        
        if (isBoundary) {
            boundaryVertices.push_back(vertexIdx);
        }
    }
    
    return boundaryVertices;
}

PlateInteractionType TectonicSimulation::getInteractionType(int v1, int v2) const {
    int p1 = planet.getPlateId(v1);
    int p2 = planet.getPlateId(v2);
    
    // Check if both vertices are assigned to plates
    if (p1 < 0 || p2 < 0 || p1 == p2) {
        return PlateInteractionType::None;
    }
    
    const SphericalGrid& grid = planet.getGrid();
    Vec3f pos1 = grid.getVertex(v1);
    Vec3f pos2 = grid.getVertex(v2);
    
    // Get plate movement vectors
    const Vec3f& move1 = plates[p1].getMovement();
    const Vec3f& move2 = plates[p2].getMovement();
    
    // Calculate relative motion
    Vec3f boundary = pos2 - pos1;
    boundary.normalize();
    
    // Project movement vectors onto boundary
    float v1Proj = move1.dot(boundary);
    float v2Proj = move2.dot(boundary);
    
    // Calculate relative velocity
    float relVelocity = v2Proj - v1Proj;
    
    // Get plate types
    bool p1Oceanic = plates[p1].isOceanic();
    bool p2Oceanic = plates[p2].isOceanic();
    
    // Determine interaction type based on relative motion and plate types
    if (std::abs(relVelocity) < 0.01f) {
        return PlateInteractionType::Transform;
    } else if (relVelocity > 0) {
        // Divergent boundary
        return PlateInteractionType::Divergent;
    } else {
        // Convergent boundary - determine type based on plate compositions
        if (p1Oceanic && p2Oceanic) {
            return PlateInteractionType::Subduction;
        } else if (!p1Oceanic && !p2Oceanic) {
            return PlateInteractionType::Collision;
        } else {
            return PlateInteractionType::Subduction;
        }
    }
}

const std::vector<TectonicPlate>& TectonicSimulation::getPlates() const {
    return plates;
}

void TectonicSimulation::setTectonicActivity(float value) {
    params.tectonicActivityScale = std::max(0.0f, std::min(1.0f, value));
}

float TectonicSimulation::getTectonicActivity() const {
    return params.tectonicActivityScale;
}

float TectonicSimulation::calculateIsostaticAdjustment(int vertexIdx) const {
    // Get crust properties
    float thickness = planet.getCrustThickness(vertexIdx);
    float density = planet.getCrustDensity(vertexIdx);
    
    // Calculate isostatic equilibrium height
    // Using simplified Airy isostasy model:
    // Height above or below sea level is proportional to thickness and density difference
    float densityDiff = params.mantleDensity - density;
    float equilibriumHeight = thickness * (densityDiff / params.mantleDensity);
    
    // Current height
    float currentHeight = planet.getElevation(vertexIdx);
    
    // Calculate adjustment (move partially toward equilibrium)
    float adjustment = (equilibriumHeight - currentHeight) * 0.2f;
    
    return adjustment;
}

Vec3f TectonicSimulation::calculateConvectionForce(const TectonicPlate& plate) const {
    Vec3f totalForce = Vec3f::Zero();
    
    // Get plate center and properties
    const Vec3f& plateCenter = plate.getCenter();
    bool isOceanic = plate.isOceanic();
    float plateArea = plate.calculateArea();
    
    // 1. MANTLE CONVECTION FORCE
    // Sum forces from all convection cells
    for (const auto& cell : convectionCells) {
        Vec3f force = cell.calculateForceAt(plateCenter);
        // Amplify force for more dynamic movement
        force *= 2.5f;
        totalForce += force;
    }
    
    // 2. SLAB PULL FORCE - major driver of plate tectonics
    // Find plate boundaries
    std::vector<int> boundaryVertices;
    for (int vertexIdx : plate.getVertices()) {
        const auto& neighbors = planet.getGrid().getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            int neighborPlateId = planet.getPlateId(neighborIdx);
            if (neighborPlateId >= 0 && neighborPlateId != plate.getId()) {
                // This is a boundary vertex
                PlateInteractionType interaction = getInteractionType(vertexIdx, neighborIdx);
                
                // Add slab pull force at subduction zones
                if (interaction == PlateInteractionType::Subduction) {
                    // Calculate direction from boundary to plate center
                    Vec3f boundaryPos = planet.getGrid().getVertex(vertexIdx);
                    Vec3f pullDirection = (plateCenter - boundaryPos).normalized();
                    
                    // Stronger pull for oceanic plates
                    float pullStrength = isOceanic ? 0.5f : 0.2f;
                    totalForce += pullDirection * pullStrength;
                }
                
                boundaryVertices.push_back(vertexIdx);
                break;
            }
        }
    }
    
    // 3. RIDGE PUSH FORCE - secondary driver of plate tectonics
    for (int vertexIdx : boundaryVertices) {
        const auto& neighbors = planet.getGrid().getNeighbors(vertexIdx);
        for (int neighborIdx : neighbors) {
            int neighborPlateId = planet.getPlateId(neighborIdx);
            if (neighborPlateId >= 0 && neighborPlateId != plate.getId()) {
                PlateInteractionType interaction = getInteractionType(vertexIdx, neighborIdx);
                
                // Add ridge push at divergent boundaries
                if (interaction == PlateInteractionType::Divergent) {
                    Vec3f boundaryPos = planet.getGrid().getVertex(vertexIdx);
                    Vec3f pushDirection = (boundaryPos - plateCenter).normalized();
                    
                    float pushStrength = 0.3f;
                    totalForce += pushDirection * pushStrength;
                }
            }
        }
    }
    
    // Scale by plate size and tectonic activity
    float areaNormalized = std::min(1.0f, plateArea / 1000.0f);
    totalForce *= areaNormalized * params.tectonicActivityScale;
    
    return totalForce;
}

void TectonicSimulation::processPlateInteractions() {
    // Get all boundary vertices
    std::vector<int> boundaries = getBoundaryVertices();
    
    // Track convergent zones for later processing (continental accretion)
    std::vector<std::pair<int, int>> accretionPairs;
    
    // Process each boundary vertex
    for (int vertexIdx : boundaries) {
        int plateId = planet.getPlateId(vertexIdx);
        if (plateId < 0) continue;
        
        const SphericalGrid& grid = planet.getGrid();
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        
        // Check interaction with each neighboring vertex from a different plate
        for (int neighborIdx : neighbors) {
            int neighborPlateId = planet.getPlateId(neighborIdx);
            if (neighborPlateId < 0 || neighborPlateId == plateId) continue;
            
            // Determine interaction type between plates
            PlateInteractionType interaction = getInteractionType(vertexIdx, neighborIdx);
            
            // Get plate properties
            bool vertexOceanic = plates[plateId].isOceanic();
            bool neighborOceanic = plates[neighborPlateId].isOceanic();
            
            // Apply effects based on interaction type
            switch (interaction) {
                case PlateInteractionType::Divergent: {
                    // SEAFLOOR SPREADING
                    // This simulates mid-ocean ridges and continental rifting
                    
                    // Get the current properties
                    float thickness = planet.getCrustThickness(vertexIdx);
                    float elevation = planet.getElevation(vertexIdx);
                    
                    // Thin crust at divergent boundaries (rifting)
                    planet.setCrustThickness(vertexIdx, thickness * 0.95f);
                    
                    // Create seafloor spreading centers (mid-ocean ridges)
                    if (vertexOceanic) {
                        // Seafloor is younger at spreading centers
                        planet.setElevation(vertexIdx, -1.5f); // Elevated seafloor at ridge
                    } else {
                        // Continental rifting can create rift valleys
                        // Eventual continental breakup
                        float randomChance = std::uniform_real_distribution<float>(0, 1)(rng);
                        if (randomChance < 0.01f * params.tectonicActivityScale) {
                            // Occasionally convert continental to oceanic crust during rifting
                            planet.setCrustType(vertexIdx, 0); // Convert to oceanic crust
                            planet.setCrustThickness(vertexIdx, 7.0f); // Thin oceanic crust
                            planet.setCrustDensity(vertexIdx, params.oceanicDensity);
                            planet.setElevation(vertexIdx, -2.0f); // Below sea level
                        } else {
                            // Normal continental rifting - creates rift valleys
                            planet.setElevation(vertexIdx, elevation - 0.2f);
                        }
                    }
                    break;
                }
                
                case PlateInteractionType::Convergent:
                case PlateInteractionType::Collision: {
                    // CONTINENTAL COLLISION (OROGENESIS)
                    // Creates major mountain ranges like the Himalayas
                    
                    if (!vertexOceanic && !neighborOceanic) {
                        // Continental-Continental Collision - major mountain building
                        float thickness = planet.getCrustThickness(vertexIdx);
                        
                        // Significant crustal thickening during continental collision
                        planet.setCrustThickness(vertexIdx, thickness * 1.1f);
                        
                        // Calculate mountain elevation based on crustal thickness
                        // More mature collisions create higher mountains
                        float elevationChange = 0.3f * params.tectonicActivityScale;
                        float elevation = planet.getElevation(vertexIdx);
                        
                        // Himalaya-like mountains can reach 8+ km
                        float maxMountainHeight = 8.0f;
                        float newElevation = std::min(maxMountainHeight, elevation + elevationChange);
                        planet.setElevation(vertexIdx, newElevation);
                    }
                    
                    // Track for accretion
                    accretionPairs.push_back(std::make_pair(vertexIdx, neighborIdx));
                    break;
                }
                
                case PlateInteractionType::Subduction: {
                    // SUBDUCTION PROCESSES
                    // Creates deep ocean trenches, volcanic arcs, new continental crust
                    
                    if (vertexOceanic && !neighborOceanic) {
                        // Oceanic plate subducting under continental plate
                        // Creates deep ocean trenches
                        float elevation = planet.getElevation(vertexIdx);
                        planet.setElevation(vertexIdx, elevation - 0.4f);
                        
                        // Ocean trench can reach -11 km (Mariana Trench)
                        if (elevation < -7.0f) {
                            planet.setElevation(vertexIdx, -7.0f);
                        }
                    }
                    else if (!vertexOceanic && neighborOceanic) {
                        // Continental plate above subduction zone (volcanic arc)
                        // This is how new continental crust forms
                        float thickness = planet.getCrustThickness(vertexIdx);
                        planet.setCrustThickness(vertexIdx, thickness * 1.08f);
                        
                        // Volcanic mountains form inland from subduction zone
                        float elevation = planet.getElevation(vertexIdx);
                        planet.setElevation(vertexIdx, elevation + 0.25f);
                        
                        // Arc volcanism - chance to create new continental material
                        float randomChance = std::uniform_real_distribution<float>(0, 1)(rng);
                        if (randomChance < 0.05f * params.tectonicActivityScale) {
                            // Make neighboring oceanic vertices continental (island arc formation)
                            for (int secondNeighborIdx : grid.getNeighbors(neighborIdx)) {
                                if (planet.getPlateId(secondNeighborIdx) == neighborPlateId) {
                                    // Transform oceanic crust to continental crust
                                    planet.setCrustType(secondNeighborIdx, 1); // Continental
                                    planet.setCrustThickness(secondNeighborIdx, 15.0f);
                                    planet.setCrustDensity(secondNeighborIdx, params.continentalDensity);
                                    
                                    // Elevate above sea level
                                    float secondNeighborElevation = planet.getElevation(secondNeighborIdx);
                                    if (secondNeighborElevation < 0.0f) {
                                        planet.setElevation(secondNeighborIdx, 0.2f);
                                    }
                                    
                                    break; // Just convert one neighbor for now
                                }
                            }
                        }
                    }
                    else if (vertexOceanic && neighborOceanic) {
                        // Oceanic-Oceanic subduction creates island arcs
                        float randomChance = std::uniform_real_distribution<float>(0, 1)(rng);
                        if (randomChance < 0.03f * params.tectonicActivityScale) {
                            // Create island arc (new continental crust)
                            planet.setCrustType(vertexIdx, 1); // Continental
                            planet.setCrustThickness(vertexIdx, 15.0f);
                            planet.setCrustDensity(vertexIdx, params.continentalDensity);
                            planet.setElevation(vertexIdx, 0.5f); // Above sea level
                        } else {
                            // Deep oceanic trench
                            float elevation = planet.getElevation(vertexIdx);
                            planet.setElevation(vertexIdx, elevation - 0.3f);
                        }
                    }
                    
                    // Track for accretion
                    accretionPairs.push_back(std::make_pair(vertexIdx, neighborIdx));
                    break;
                }
                
                case PlateInteractionType::Transform: {
                    // TRANSFORM BOUNDARIES
                    // Mostly horizontal motion, but can create features like San Andreas fault
                    
                    // Small chance of creating pull-apart basins or transpressional ridges
                    float randomChance = std::uniform_real_distribution<float>(0, 1)(rng);
                    if (randomChance < 0.02f * params.tectonicActivityScale) {
                        float elevation = planet.getElevation(vertexIdx);
                        if (randomChance < 0.01f) {
                            // Pull-apart basin
                            planet.setElevation(vertexIdx, elevation - 0.1f);
                        } else {
                            // Transpressional ridge
                            planet.setElevation(vertexIdx, elevation + 0.1f);
                        }
                    }
                    break;
                }
                
                case PlateInteractionType::None:
                    // No interaction
                    break;
            }
        }
    }
    
    // Process continental accretion and terrane addition
    // This is how continents grow over time
    for (const auto& pair : accretionPairs) {
        int vertexIdx = pair.first;
        int neighborIdx = pair.second;
        int plateId = planet.getPlateId(vertexIdx);
        int neighborPlateId = planet.getPlateId(neighborIdx);
        
        // Chance for plate material to be accreted onto another plate
        float randomChance = std::uniform_real_distribution<float>(0, 1)(rng);
        if (randomChance < 0.01f * params.tectonicActivityScale) {
            // Transfer vertex to neighboring plate (accretion)
            if (!plates[plateId].isOceanic() && plates[neighborPlateId].isOceanic()) {
                // Continental crust "captures" oceanic crust
                planet.setPlateId(neighborIdx, plateId);
                
                // Remove from original plate and add to new plate
                // (In a real implementation, we would update the plate.vertices collection as well)
                
                // Convert to continental crust
                planet.setCrustType(neighborIdx, 1); // Continental
                planet.setCrustThickness(neighborIdx, 20.0f);
                planet.setCrustDensity(neighborIdx, params.continentalDensity);
            }
        }
    }
}

void TectonicSimulation::calculatePlateMovements(double timeStep) {
    // Random jitter to break symmetry
    std::uniform_real_distribution<float> jitterDist(-0.05f, 0.05f);
    
    // Calculate movement for each plate based on all force components
    for (auto& plate : plates) {
        // Calculate forces from convection cells, slab pull, and ridge push
        Vec3f force = calculateConvectionForce(plate);
        
        // Add random jitter to break symmetric patterns
        Vec3f jitter(jitterDist(rng), jitterDist(rng), jitterDist(rng));
        force += jitter * params.tectonicActivityScale;
        
        // Apply force to get movement vector (simplified physics)
        // Larger timeStep value means greater movement per step
        Vec3f movement = force * static_cast<float>(timeStep * 2.0);
        
        // Ensure movement is tangent to surface at plate center
        // Project onto tangent plane at plate center
        const Vec3f& center = plate.getCenter();
        Vec3f normalizedCenter = center.normalized();
        float dotProduct = movement.dot(normalizedCenter);
        movement -= normalizedCenter * dotProduct;
        
        // Vary max movement by plate type - oceanic plates generally move faster
        float maxMovement = plate.isOceanic() ? 0.3f : 0.15f;
        
        // Scale by plate size - smaller plates can move faster
        float sizeFactor = 1.0f - std::min(0.5f, plate.calculateArea() / 2000.0f);
        maxMovement *= (1.0f + sizeFactor);
        
        // Apply maximum movement constraint
        float magnitude = movement.norm();
        if (magnitude > maxMovement) {
            movement *= maxMovement / magnitude;
        }
        
        // Preserve some momentum from previous movement (inertia)
        Vec3f prevMovement = plate.getMovement();
        if (prevMovement.norm() > 0) {
            movement = movement * 0.8f + prevMovement * 0.2f;
        }
        
        // Set plate movement
        plate.setMovement(movement);
        
        // Debug output for significant movement
        if (movement.norm() > 0.01f) {
            std::cout << "Plate " << plate.getId() << " movement: "
                      << movement.norm() << " direction: ("
                      << movement.x() << ", " << movement.y() << ", " << movement.z() << ")"
                      << std::endl;
        }
    }
}

void TectonicSimulation::applyErosionAndSedimentation(double timeStep) {
    const SphericalGrid& grid = planet.getGrid();
    size_t vertexCount = grid.getVertexCount();
    
    // Temporary arrays to store changes
    std::vector<float> elevationChanges(vertexCount, 0.0f);
    std::vector<float> sedimentTransport(vertexCount, 0.0f);
    
    // STEP 1: Calculate weathering and erosion based on elevation, slope, and crust type
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        bool isOceanic = planet.getCrustType(vertexIdx) == 0;
        
        // Get neighbors
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        
        // Calculate slope (average elevation difference)
        float maxElevationDiff = 0.0f;
        float avgElevationDiff = 0.0f;
        float lowestNeighborElevation = elevation;
        int lowestNeighborIdx = -1;
        
        for (int neighborIdx : neighbors) {
            float neighborElevation = planet.getElevation(neighborIdx);
            float diff = elevation - neighborElevation;
            
            // Track maximum slope for erosion calculation
            if (diff > maxElevationDiff) {
                maxElevationDiff = diff;
            }
            
            // Track lowest neighbor for flow direction
            if (neighborElevation < lowestNeighborElevation) {
                lowestNeighborElevation = neighborElevation;
                lowestNeighborIdx = neighborIdx;
            }
            
            avgElevationDiff += std::abs(diff);
        }
        avgElevationDiff /= neighbors.size();
        
        // Base erosion rate varies by elevation, slope, and land type
        float baseErosionRate = 0.0f;
        
        if (elevation > 0.0f) {
            // Land erosion - increases with elevation and slope
            // Mountains erode faster than lowlands
            float elevationFactor = std::min(1.0f, elevation / 5.0f);
            float slopeFactor = std::min(1.0f, avgElevationDiff * 5.0f);
            
            // Continental weathering
            if (!isOceanic) {
                // Continental rock erodes faster at high elevations (mechanical weathering)
                baseErosionRate = params.erosionRate * (0.5f + elevationFactor) * (0.2f + slopeFactor);
                
                // High mountains experience more rapid erosion (freeze-thaw cycles)
                if (elevation > 3.0f) {
                    baseErosionRate *= 1.5f;
                }
            } else {
                // Exposed oceanic crust (rare) erodes more slowly
                baseErosionRate = params.erosionRate * 0.3f * (0.5f + slopeFactor);
            }
        } else {
            // Submarine erosion - minimal except on steep slopes
            float slopeFactor = std::min(1.0f, avgElevationDiff * 3.0f);
            baseErosionRate = params.erosionRate * 0.1f * slopeFactor;
        }
        
        // Calculate actual erosion - steep slopes erode more
        float erosion = 0.0f;
        if (maxElevationDiff > 0.0f) {
            erosion = baseErosionRate * maxElevationDiff * static_cast<float>(timeStep);
            elevationChanges[i] -= erosion;
            
            // Track sediment transport
            sedimentTransport[i] += erosion;
        }
    }
    
    // STEP 2: Transport sediment along flow paths (rivers, submarine currents)
    for (size_t i = 0; i < vertexCount; ++i) {
        if (sedimentTransport[i] <= 0.0f) continue;
        
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        const auto& neighbors = grid.getNeighbors(vertexIdx);
        
        // Find flow direction (lowest neighbor)
        int lowestNeighborIdx = -1;
        float lowestElevation = elevation;
        
        for (int neighborIdx : neighbors) {
            float neighborElevation = planet.getElevation(neighborIdx);
            if (neighborElevation < lowestElevation) {
                lowestElevation = neighborElevation;
                lowestNeighborIdx = neighborIdx;
            }
        }
        
        if (lowestNeighborIdx >= 0 && lowestElevation < elevation) {
            // Transport sediment downhill
            float transportAmount = sedimentTransport[i] * 0.7f; // Some sediment remains local
            
            // Deposit based on slope gradient
            float slope = elevation - lowestElevation;
            float depositRatio = std::min(1.0f, 0.2f / (slope + 0.1f));
            
            // Sediment deposited at current location
            float localDeposit = transportAmount * depositRatio;
            elevationChanges[i] += localDeposit;
            
            // Remaining sediment continues downhill
            float continuingTransport = transportAmount - localDeposit;
            sedimentTransport[i] -= transportAmount;
            sedimentTransport[lowestNeighborIdx] += continuingTransport;
        } else {
            // No downhill path, deposit locally (depressions/basins)
            elevationChanges[i] += sedimentTransport[i] * 0.5f;
            
            // Rest distributes evenly to all neighbors (alluvial fan)
            float neighborShare = sedimentTransport[i] * 0.5f / neighbors.size();
            for (int neighborIdx : neighbors) {
                elevationChanges[neighborIdx] += neighborShare;
            }
            
            sedimentTransport[i] = 0.0f;
        }
    }
    
    // STEP 3: Special coastal processes - form beaches, deltas, etc.
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        float elevation = planet.getElevation(vertexIdx);
        
        // Check if this is a coastal vertex (near sea level)
        if (std::abs(elevation) < 0.2f) {
            const auto& neighbors = grid.getNeighbors(vertexIdx);
            bool hasOcean = false;
            bool hasLand = false;
            
            for (int neighborIdx : neighbors) {
                float neighborElevation = planet.getElevation(neighborIdx);
                if (neighborElevation < -0.1f) hasOcean = true;
                if (neighborElevation > 0.1f) hasLand = true;
            }
            
            // Process coastal features where land meets ocean
            if (hasLand && hasOcean) {
                if (elevation > 0.0f) {
                    // Coastal erosion is stronger
                    elevationChanges[i] -= 0.05f * params.erosionRate * static_cast<float>(timeStep);
                } else {
                    // Coastal deposition (deltas, beaches)
                    elevationChanges[i] += 0.03f * params.sedimentationRate * static_cast<float>(timeStep);
                }
            }
        }
    }
    
    // STEP 4: Apply all calculated elevation changes
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        float newElevation = planet.getElevation(vertexIdx) + elevationChanges[i];
        planet.setElevation(vertexIdx, newElevation);
        
        // Update crust type for significantly eroded or deposited areas
        if (elevationChanges[i] > 0.5f && newElevation > 0.0f) {
            // Significant deposition above sea level - likely continental
            planet.setCrustType(vertexIdx, 1); // Continental
        } else if (elevationChanges[i] < -0.8f && newElevation < 0.0f) {
            // Significantly eroded below sea level - exposed oceanic crust
            planet.setCrustType(vertexIdx, 0); // Oceanic
        }
    }
}

void TectonicSimulation::applyIsostaticAdjustments() {
    const SphericalGrid& grid = planet.getGrid();
    size_t vertexCount = grid.getVertexCount();
    
    // Apply isostatic adjustments to each vertex
    for (size_t i = 0; i < vertexCount; ++i) {
        int vertexIdx = static_cast<int>(i);
        
        // Calculate isostatic adjustment
        float adjustment = calculateIsostaticAdjustment(vertexIdx);
        
        // Apply adjustment
        float elevation = planet.getElevation(vertexIdx);
        planet.setElevation(vertexIdx, elevation + adjustment);
    }
}

int TectonicSimulation::identifyClosestPlate(int vertexIdx) const {
    const SphericalGrid& grid = planet.getGrid();
    Vec3f position = grid.getVertex(vertexIdx);
    
    int closestPlate = -1;
    float minDistance = std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < plates.size(); ++i) {
        int plateId = static_cast<int>(i);
        const Vec3f& plateCenter = plates[i].getCenter();
        
        float distance = (plateCenter - position).norm();
        if (distance < minDistance) {
            minDistance = distance;
            closestPlate = plateId;
        }
    }
    
    return closestPlate;
}

float TectonicSimulation::getSuperContinentCyclePhase() const {
    return params.superContinentCyclePhase;
}

void TectonicSimulation::applySuperContinentCycleForces() {
    // Apply forces based on supercontinent cycle phase
    // 0.0-0.3: Assembly phase - continents coming together
    // 0.3-0.5: Supercontinent phase - stable supercontinent
    // 0.5-1.0: Dispersal phase - continents breaking apart
    
    const float assemblyPhaseEnd = 0.3f;
    const float supercontinentPhaseEnd = 0.5f;
    
    if (params.superContinentCyclePhase < assemblyPhaseEnd) {
        // Assembly phase - continents should move toward each other
        // Find a convergence point (simplified approach: use planet center)
        Vec3f convergencePoint = Vec3f::Zero();
        
        // Push continental plates toward convergence point
        for (auto& plate : plates) {
            if (!plate.isOceanic()) {
                // Get vector from plate center to convergence point
                Vec3f direction = (convergencePoint - plate.getCenter()).normalized();
                
                // Create a weak force in this direction
                Vec3f force = direction * 0.3f;
                
                // Add to plate's movement
                Vec3f currentMovement = plate.getMovement();
                plate.setMovement(currentMovement + force * params.tectonicActivityScale);
            }
        }
        
        std::cout << "Supercontinent cycle: Assembly phase - continents converging" << std::endl;
    }
    else if (params.superContinentCyclePhase < supercontinentPhaseEnd) {
        // Supercontinent phase - reduced movement for continental plates
        for (auto& plate : plates) {
            if (!plate.isOceanic()) {
                // Reduce continental plate movement
                Vec3f currentMovement = plate.getMovement();
                plate.setMovement(currentMovement * 0.5f);
            }
        }
        
        std::cout << "Supercontinent cycle: Supercontinent phase - reduced continental movement" << std::endl;
    }
    else {
        // Dispersal phase - continents should move away from each other
        // Calculate average continental centroid
        Vec3f continentalCentroid = Vec3f::Zero();
        int continentalPlateCount = 0;
        
        for (const auto& plate : plates) {
            if (!plate.isOceanic()) {
                continentalCentroid += plate.getCenter();
                continentalPlateCount++;
            }
        }
        
        if (continentalPlateCount > 0) {
            continentalCentroid /= continentalPlateCount;
            
            // Push continental plates away from centroid
            for (auto& plate : plates) {
                if (!plate.isOceanic()) {
                    // Get vector from centroid to plate center
                    Vec3f direction = (plate.getCenter() - continentalCentroid).normalized();
                    
                    // Create a force in this direction
                    Vec3f force = direction * 0.4f;
                    
                    // Add to plate's movement
                    Vec3f currentMovement = plate.getMovement();
                    plate.setMovement(currentMovement + force * params.tectonicActivityScale);
                }
            }
        }
        
        std::cout << "Supercontinent cycle: Dispersal phase - continents separating" << std::endl;
    }
}

} // namespace aeonterracpp