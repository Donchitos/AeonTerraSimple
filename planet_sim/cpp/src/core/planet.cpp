#include "aeonterracpp/core/planet.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace aeonterracpp {

//-----------------------------------------------------------------------------
// HistoricalSnapshot implementation
//-----------------------------------------------------------------------------

HistoricalSnapshot::HistoricalSnapshot(double _age, size_t vertexCount)
    : age(_age),
      elevation(vertexCount, 0.0f),
      plateId(vertexCount, -1),
      maxElevation(-std::numeric_limits<float>::max()),
      minElevation(std::numeric_limits<float>::max()),
      oceanPercentage(0.0f),
      plateCount(0) {
}

//-----------------------------------------------------------------------------
// Planet implementation
//-----------------------------------------------------------------------------

Planet::Planet(float radius, int resolution, const std::string& name)
    : grid(radius, resolution),
      age(0.0),
      name(name) {
    
    // Initialize property vectors
    initializeProperties();
    
    // Record initial state
    recordSnapshot();
}

Planet::Planet(Planet&& other) noexcept
    : grid(std::move(other.grid)),
      elevation(std::move(other.elevation)),
      crustThickness(std::move(other.crustThickness)),
      crustDensity(std::move(other.crustDensity)),
      crustType(std::move(other.crustType)),
      plateId(std::move(other.plateId)),
      age(other.age),
      name(std::move(other.name)),
      history(std::move(other.history)) {
}

Planet& Planet::operator=(Planet&& other) noexcept {
    if (this != &other) {
        grid = std::move(other.grid);
        elevation = std::move(other.elevation);
        crustThickness = std::move(other.crustThickness);
        crustDensity = std::move(other.crustDensity);
        crustType = std::move(other.crustType);
        plateId = std::move(other.plateId);
        age = other.age;
        name = std::move(other.name);
        history = std::move(other.history);
    }
    return *this;
}

Planet::~Planet() = default;

void Planet::initializeProperties() {
    size_t vertexCount = grid.getVertexCount();
    
    // Resize property arrays
    elevation.resize(vertexCount, 0.0f);
    crustThickness.resize(vertexCount, 0.0f);
    crustDensity.resize(vertexCount, 0.0f);
    crustType.resize(vertexCount, 0);
    plateId.resize(vertexCount, -1);
    
    // Set default values
    float defaultOceanicThickness = 7.0f;   // km
    float defaultContinentalThickness = 35.0f; // km
    float oceanicDensity = 3.0f;     // g/cm³
    float continentalDensity = 2.7f; // g/cm³
    
    // Generate a simple initial state - a single continent
    for (size_t i = 0; i < vertexCount; ++i) {
        Vec3f position = grid.getVertex(i);
        auto [lat, lon] = grid.cartesianToLatLon(position);
        
        // Simple elevation model - a circular continent centered at (20°N, 0°E)
        float latCenter = 20.0f;
        float lonCenter = 0.0f;
        
        // Calculate angular distance from the center point
        float angularDist = std::acos(std::sin(toRadians(lat)) * std::sin(toRadians(latCenter)) +
                                     std::cos(toRadians(lat)) * std::cos(toRadians(latCenter)) *
                                     std::cos(toRadians(lon) - toRadians(lonCenter)));
        angularDist = toDegrees(angularDist);
        
        // Create a continent if within 40 degrees of the center
        bool isContinent = (angularDist < 40.0f);
        
        if (isContinent) {
            // Continental crust (elevated)
            crustType[i] = 1;
            crustThickness[i] = defaultContinentalThickness;
            crustDensity[i] = continentalDensity;
            
            // Simple elevation model - higher in the center, tapering to sea level
            float elevationFactor = 1.0f - (angularDist / 40.0f);
            elevation[i] = 2.0f * elevationFactor * elevationFactor;  // Max height 2km
        } else {
            // Oceanic crust (below sea level)
            crustType[i] = 0;
            crustThickness[i] = defaultOceanicThickness;
            crustDensity[i] = oceanicDensity;
            elevation[i] = -4.0f;  // Average ocean depth around 4km
        }
        
        // Initially no plate assignment
        plateId[i] = -1;
    }
    
    std::cout << "Planet initialized with " << vertexCount << " vertices" << std::endl;
}

float Planet::getElevation(int vertexIdx) const {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(elevation.size())) {
        throw std::out_of_range("Vertex index out of range in getElevation");
    }
    return elevation[vertexIdx];
}

void Planet::setElevation(int vertexIdx, float value) {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(elevation.size())) {
        throw std::out_of_range("Vertex index out of range in setElevation");
    }
    elevation[vertexIdx] = value;
}

float Planet::getCrustThickness(int vertexIdx) const {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(crustThickness.size())) {
        throw std::out_of_range("Vertex index out of range in getCrustThickness");
    }
    return crustThickness[vertexIdx];
}

void Planet::setCrustThickness(int vertexIdx, float value) {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(crustThickness.size())) {
        throw std::out_of_range("Vertex index out of range in setCrustThickness");
    }
    crustThickness[vertexIdx] = value;
}

float Planet::getCrustDensity(int vertexIdx) const {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(crustDensity.size())) {
        throw std::out_of_range("Vertex index out of range in getCrustDensity");
    }
    return crustDensity[vertexIdx];
}

void Planet::setCrustDensity(int vertexIdx, float value) {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(crustDensity.size())) {
        throw std::out_of_range("Vertex index out of range in setCrustDensity");
    }
    crustDensity[vertexIdx] = value;
}

int Planet::getCrustType(int vertexIdx) const {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(crustType.size())) {
        throw std::out_of_range("Vertex index out of range in getCrustType");
    }
    return crustType[vertexIdx];
}

void Planet::setCrustType(int vertexIdx, int value) {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(crustType.size())) {
        throw std::out_of_range("Vertex index out of range in setCrustType");
    }
    crustType[vertexIdx] = value;
}

int Planet::getPlateId(int vertexIdx) const {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(plateId.size())) {
        throw std::out_of_range("Vertex index out of range in getPlateId");
    }
    return plateId[vertexIdx];
}

void Planet::setPlateId(int vertexIdx, int value) {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(plateId.size())) {
        throw std::out_of_range("Vertex index out of range in setPlateId");
    }
    plateId[vertexIdx] = value;
}

const SphericalGrid& Planet::getGrid() const {
    return grid;
}

size_t Planet::getVertexCount() const {
    return grid.getVertexCount();
}

float Planet::getRadius() const {
    return grid.getRadius();
}

void Planet::advanceAge(double timeStep) {
    // Simply advance the age - tectonics simulation will be separate
    age += timeStep;
}

double Planet::getAge() const {
    return age;
}

void Planet::recordSnapshot() {
    // Create a new snapshot
    HistoricalSnapshot snapshot(age, getVertexCount());
    
    // Copy current planet state
    snapshot.elevation = elevation;
    snapshot.plateId = plateId;
    
    // Calculate statistics
    snapshot.maxElevation = -std::numeric_limits<float>::max();
    snapshot.minElevation = std::numeric_limits<float>::max();
    int oceanCount = 0;
    
    // Gather unique plate IDs to count plates
    std::vector<int> uniquePlateIds;
    
    for (size_t i = 0; i < getVertexCount(); ++i) {
        // Track min/max elevation
        snapshot.maxElevation = std::max(snapshot.maxElevation, elevation[i]);
        snapshot.minElevation = std::min(snapshot.minElevation, elevation[i]);
        
        // Count ocean vertices (below sea level)
        if (elevation[i] < 0.0f) {
            oceanCount++;
        }
        
        // Track plate IDs
        if (plateId[i] >= 0) {
            if (std::find(uniquePlateIds.begin(), uniquePlateIds.end(), plateId[i]) == uniquePlateIds.end()) {
                uniquePlateIds.push_back(plateId[i]);
            }
        }
    }
    
    // Calculate ocean percentage
    snapshot.oceanPercentage = static_cast<float>(oceanCount) / getVertexCount();
    
    // Set plate count
    snapshot.plateCount = static_cast<int>(uniquePlateIds.size());
    
    // Add to history
    history.push_back(snapshot);
    
    std::cout << "Recorded snapshot at age " << age << " million years" << std::endl;
    std::cout << "  Elevation range: " << snapshot.minElevation << " to " 
              << snapshot.maxElevation << " km" << std::endl;
    std::cout << "  Ocean coverage: " << (snapshot.oceanPercentage * 100.0f) << "%" << std::endl;
    std::cout << "  Plate count: " << snapshot.plateCount << std::endl;
}

const std::string& Planet::getName() const {
    return name;
}

void Planet::setName(const std::string& newName) {
    name = newName;
}

const std::vector<HistoricalSnapshot>& Planet::getHistory() const {
    return history;
}

void Planet::clearHistory() {
    history.clear();
}

void Planet::saveToFile(const std::string& filename) const {
    std::cout << "Saving planet to " << filename << "..." << std::endl;
    
    try {
        // Open output file
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Write header info
        file.write(reinterpret_cast<const char*>(&age), sizeof(age));
        
        // Write name
        size_t nameLength = name.length();
        file.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
        file.write(name.c_str(), nameLength);
        
        // Write grid information
        float radius = grid.getRadius();
        int resolution = grid.getResolution();
        size_t vertexCount = grid.getVertexCount();
        
        file.write(reinterpret_cast<const char*>(&radius), sizeof(radius));
        file.write(reinterpret_cast<const char*>(&resolution), sizeof(resolution));
        file.write(reinterpret_cast<const char*>(&vertexCount), sizeof(vertexCount));
        
        // Write property arrays
        file.write(reinterpret_cast<const char*>(elevation.data()), vertexCount * sizeof(float));
        file.write(reinterpret_cast<const char*>(crustThickness.data()), vertexCount * sizeof(float));
        file.write(reinterpret_cast<const char*>(crustDensity.data()), vertexCount * sizeof(float));
        file.write(reinterpret_cast<const char*>(crustType.data()), vertexCount * sizeof(int));
        file.write(reinterpret_cast<const char*>(plateId.data()), vertexCount * sizeof(int));
        
        // Write history
        size_t historyCount = history.size();
        file.write(reinterpret_cast<const char*>(&historyCount), sizeof(historyCount));
        
        for (const auto& snapshot : history) {
            file.write(reinterpret_cast<const char*>(&snapshot.age), sizeof(snapshot.age));
            file.write(reinterpret_cast<const char*>(&snapshot.maxElevation), sizeof(snapshot.maxElevation));
            file.write(reinterpret_cast<const char*>(&snapshot.minElevation), sizeof(snapshot.minElevation));
            file.write(reinterpret_cast<const char*>(&snapshot.oceanPercentage), sizeof(snapshot.oceanPercentage));
            file.write(reinterpret_cast<const char*>(&snapshot.plateCount), sizeof(snapshot.plateCount));
            
            // Write snapshot data
            file.write(reinterpret_cast<const char*>(snapshot.elevation.data()), vertexCount * sizeof(float));
            file.write(reinterpret_cast<const char*>(snapshot.plateId.data()), vertexCount * sizeof(int));
        }
        
        std::cout << "Planet saved successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving planet: " << e.what() << std::endl;
        throw;
    }
}

Planet Planet::loadFromFile(const std::string& filename) {
    std::cout << "Loading planet from " << filename << "..." << std::endl;
    
    try {
        // Open input file
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        // Read header info
        double age;
        file.read(reinterpret_cast<char*>(&age), sizeof(age));
        
        // Read name
        size_t nameLength;
        file.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
        std::string name(nameLength, ' ');
        file.read(&name[0], nameLength);
        
        // Read grid information
        float radius;
        int resolution;
        size_t vertexCount;
        
        file.read(reinterpret_cast<char*>(&radius), sizeof(radius));
        file.read(reinterpret_cast<char*>(&resolution), sizeof(resolution));
        file.read(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));
        
        // Create planet with specified parameters
        Planet planet(radius, resolution, name);
        planet.age = age;
        
        // Make sure the vertex count matches
        if (planet.getVertexCount() != vertexCount) {
            throw std::runtime_error("Vertex count mismatch when loading planet");
        }
        
        // Read property arrays
        file.read(reinterpret_cast<char*>(planet.elevation.data()), vertexCount * sizeof(float));
        file.read(reinterpret_cast<char*>(planet.crustThickness.data()), vertexCount * sizeof(float));
        file.read(reinterpret_cast<char*>(planet.crustDensity.data()), vertexCount * sizeof(float));
        file.read(reinterpret_cast<char*>(planet.crustType.data()), vertexCount * sizeof(int));
        file.read(reinterpret_cast<char*>(planet.plateId.data()), vertexCount * sizeof(int));
        
        // Read history
        size_t historyCount;
        file.read(reinterpret_cast<char*>(&historyCount), sizeof(historyCount));
        
        // Clear existing history (should be just the initial state)
        planet.clearHistory();
        
        // Read snapshots
        for (size_t i = 0; i < historyCount; ++i) {
            HistoricalSnapshot snapshot(0.0, vertexCount);
            
            file.read(reinterpret_cast<char*>(&snapshot.age), sizeof(snapshot.age));
            file.read(reinterpret_cast<char*>(&snapshot.maxElevation), sizeof(snapshot.maxElevation));
            file.read(reinterpret_cast<char*>(&snapshot.minElevation), sizeof(snapshot.minElevation));
            file.read(reinterpret_cast<char*>(&snapshot.oceanPercentage), sizeof(snapshot.oceanPercentage));
            file.read(reinterpret_cast<char*>(&snapshot.plateCount), sizeof(snapshot.plateCount));
            
            // Read snapshot data
            file.read(reinterpret_cast<char*>(snapshot.elevation.data()), vertexCount * sizeof(float));
            file.read(reinterpret_cast<char*>(snapshot.plateId.data()), vertexCount * sizeof(int));
            
            planet.history.push_back(snapshot);
        }
        
        std::cout << "Planet loaded successfully:" << std::endl;
        std::cout << "  Name: " << planet.getName() << std::endl;
        std::cout << "  Age: " << planet.getAge() << " million years" << std::endl;
        std::cout << "  Vertex count: " << planet.getVertexCount() << std::endl;
        std::cout << "  History snapshots: " << planet.getHistory().size() << std::endl;
        
        return std::move(planet);  // Use move semantics to avoid copy
    } catch (const std::exception& e) {
        std::cerr << "Error loading planet: " << e.what() << std::endl;
        throw;
    }
}

} // namespace aeonterracpp