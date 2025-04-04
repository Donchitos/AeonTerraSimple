#include "aeonterracpp/core/spherical_grid.h"
#include "aeonterracpp/core/spatial_index.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace aeonterracpp {

SphericalGrid::SphericalGrid(float radius, int resolution)
    : vertices(), faces(), neighbors(), spatialIndex(nullptr), radius(radius), resolution(0) {
    // Create the base icosahedron
    createIcosahedron();
    
    // Initialize neighbor list
    rebuildNeighborList();
    
    // Subdivide to the desired resolution
    if (resolution > 0) {
        subdivide(resolution);
    }
    
    // Build neighbor list and spatial index
    rebuildNeighborList();
    updateSpatialIndex();
    
    this->resolution = resolution;
}

SphericalGrid::SphericalGrid(SphericalGrid&& other) noexcept
    : vertices(std::move(other.vertices)),
      faces(std::move(other.faces)),
      neighbors(std::move(other.neighbors)),
      spatialIndex(std::move(other.spatialIndex)),
      radius(other.radius),
      resolution(other.resolution) {
}

SphericalGrid& SphericalGrid::operator=(SphericalGrid&& other) noexcept {
    if (this != &other) {
        vertices = std::move(other.vertices);
        faces = std::move(other.faces);
        neighbors = std::move(other.neighbors);
        spatialIndex = std::move(other.spatialIndex);
        radius = other.radius;
        resolution = other.resolution;
    }
    return *this;
}

SphericalGrid::~SphericalGrid() = default;

size_t SphericalGrid::getVertexCount() const {
    return vertices.size();
}

size_t SphericalGrid::getFaceCount() const {
    return faces.size();
}

const Vec3f& SphericalGrid::getVertex(size_t index) const {
    if (index >= vertices.size()) {
        throw std::out_of_range("Vertex index out of range");
    }
    return vertices[index];
}

const Vec3i& SphericalGrid::getFace(size_t index) const {
    if (index >= faces.size()) {
        throw std::out_of_range("Face index out of range");
    }
    return faces[index];
}

const std::vector<int>& SphericalGrid::getNeighbors(int vertexIdx) const {
    if (vertexIdx < 0 || vertexIdx >= static_cast<int>(neighbors.size())) {
        throw std::out_of_range("Vertex index out of range");
    }
    return neighbors[vertexIdx];
}

void SphericalGrid::createIcosahedron() {
    // Clear existing data
    vertices.clear();
    faces.clear();
    
    // Golden ratio
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    
    // Create the 12 vertices of the icosahedron
    vertices = {
        Vec3f(-1, t, 0).normalized() * radius,
        Vec3f(1, t, 0).normalized() * radius,
        Vec3f(-1, -t, 0).normalized() * radius,
        Vec3f(1, -t, 0).normalized() * radius,
        Vec3f(0, -1, t).normalized() * radius,
        Vec3f(0, 1, t).normalized() * radius,
        Vec3f(0, -1, -t).normalized() * radius,
        Vec3f(0, 1, -t).normalized() * radius,
        Vec3f(t, 0, -1).normalized() * radius,
        Vec3f(t, 0, 1).normalized() * radius,
        Vec3f(-t, 0, -1).normalized() * radius,
        Vec3f(-t, 0, 1).normalized() * radius
    };
    
    // Create the 20 triangular faces
    faces = {
        Vec3i(0, 11, 5), Vec3i(0, 5, 1), Vec3i(0, 1, 7), Vec3i(0, 7, 10), Vec3i(0, 10, 11),
        Vec3i(1, 5, 9), Vec3i(5, 11, 4), Vec3i(11, 10, 2), Vec3i(10, 7, 6), Vec3i(7, 1, 8),
        Vec3i(3, 9, 4), Vec3i(3, 4, 2), Vec3i(3, 2, 6), Vec3i(3, 6, 8), Vec3i(3, 8, 9),
        Vec3i(4, 9, 5), Vec3i(2, 4, 11), Vec3i(6, 2, 10), Vec3i(8, 6, 7), Vec3i(9, 8, 1)
    };
}

void SphericalGrid::subdivide(int levels) {
    if (levels <= 0) return;
    
    std::cout << "Subdividing grid by " << levels << " levels" << std::endl;
    std::cout << "Initial vertices: " << vertices.size() << ", faces: " << faces.size() << std::endl;
    
    for (int level = 0; level < levels; ++level) {
        // Create copies of the current vertices
        std::vector<Vec3f> newVertices = vertices;
        std::vector<Vec3i> newFaces;
        
        // Edge midpoint cache to avoid duplicate vertices
        std::unordered_map<std::pair<int, int>, int, PairHash> midpointCache;
        
        // Reserve space for new faces (each face becomes 4 new faces)
        newFaces.reserve(faces.size() * 4);
        
        // Process each face
        for (const auto& face : faces) {
            int v0 = face[0], v1 = face[1], v2 = face[2];
            
            // Get or create midpoints
            int m01 = getOrCreateMidpoint(v0, v1, newVertices, midpointCache);
            int m12 = getOrCreateMidpoint(v1, v2, newVertices, midpointCache);
            int m20 = getOrCreateMidpoint(v2, v0, newVertices, midpointCache);
            
            // Create 4 new faces
            newFaces.emplace_back(Vec3i(v0, m01, m20));
            newFaces.emplace_back(Vec3i(v1, m12, m01));
            newFaces.emplace_back(Vec3i(v2, m20, m12));
            newFaces.emplace_back(Vec3i(m01, m12, m20));
        }
        
        // Update data structures
        vertices = std::move(newVertices);
        faces = std::move(newFaces);
        
        // Update resolution for this level
        resolution += 1;
        
        std::cout << "After level " << (level+1) << ": "
                  << vertices.size() << " vertices, "
                  << faces.size() << " faces" << std::endl;
        
        // Rebuild neighbor lists after each level
        rebuildNeighborList();
    }
    
    // Update spatial index only once at the end
    updateSpatialIndex();
}

int SphericalGrid::getOrCreateMidpoint(int v0, int v1,
                                      std::vector<Vec3f>& newVertices,
                                      std::unordered_map<std::pair<int, int>, int, PairHash>& cache) {
    // Ensure consistent ordering of indices for cache
    if (v0 > v1) std::swap(v0, v1);
    
    // Check if midpoint already exists
    auto key = std::make_pair(v0, v1);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    
    // Create a new vertex at the midpoint
    Vec3f midpoint = (vertices[v0] + vertices[v1]) * 0.5f;
    
    // Normalize to the sphere
    midpoint.normalize();
    midpoint *= radius;
    
    // Add to vertices - newVertices is passed by reference in subdivide
    int newIndex = static_cast<int>(newVertices.size());
    newVertices.push_back(midpoint);
    
    // Cache this midpoint
    cache[key] = newIndex;
    
    return newIndex;
}

void SphericalGrid::rebuildNeighborList() {
    size_t numVertices = vertices.size();
    neighbors.clear();
    neighbors.resize(numVertices);
    
    // For each face, add the adjacent vertices to each vertex's neighbor list
    for (const auto& face : faces) {
        int a = face[0], b = face[1], c = face[2];
        
        neighbors[a].push_back(b);
        neighbors[a].push_back(c);
        
        neighbors[b].push_back(a);
        neighbors[b].push_back(c);
        
        neighbors[c].push_back(a);
        neighbors[c].push_back(b);
    }
    
    // Remove duplicates from each neighbor list
    for (auto& neighborList : neighbors) {
        std::sort(neighborList.begin(), neighborList.end());
        auto last = std::unique(neighborList.begin(), neighborList.end());
        neighborList.erase(last, neighborList.end());
    }
}

void SphericalGrid::updateSpatialIndex() {
    // Create new spatial index with bounds slightly larger than the planet
    float boundSize = radius * 2.1f; // Just slightly larger than the diameter
    spatialIndex = std::make_unique<OctreeSpatialIndex>(boundSize);
    
    // Add all vertices to the spatial index
    for (size_t i = 0; i < vertices.size(); ++i) {
        spatialIndex->insert(static_cast<int>(i), vertices[i]);
    }
    
    // Debug output
    std::cout << "Spatial index updated with " << vertices.size() << " vertices" << std::endl;
}

Vec3f SphericalGrid::latLonToCartesian(float lat, float lon) const {
    // Convert to radians
    float latRad = toRadians(lat);
    float lonRad = toRadians(lon);
    
    // Convert spherical to Cartesian coordinates
    float x = radius * std::cos(latRad) * std::cos(lonRad);
    float y = radius * std::cos(latRad) * std::sin(lonRad);
    float z = radius * std::sin(latRad);
    
    return Vec3f(x, y, z);
}

std::pair<float, float> SphericalGrid::cartesianToLatLon(const Vec3f& position) const {
    // Calculate distance from origin
    float r = position.norm();
    
    // Check for zero vector
    if (r < 1e-6f) {
        return std::make_pair(0.0f, 0.0f);
    }
    
    // Convert to lat/lon
    float lat = toDegrees(std::asin(position.z() / r));
    float lon = toDegrees(std::atan2(position.y(), position.x()));
    
    return std::make_pair(lat, lon);
}

int SphericalGrid::findNearestVertex(float lat, float lon) const {
    // Convert lat/lon to 3D position
    Vec3f position = latLonToCartesian(lat, lon);
    
    // Check if spatial index exists
    if (!spatialIndex) {
        std::cerr << "Warning: Spatial index not initialized in findNearestVertex" << std::endl;
        return -1;
    }
    
    // Use spatial index to find nearest vertex
    int nearestIdx = spatialIndex->findNearest(position);
    
    // Debug output for negative indices
    if (nearestIdx < 0) {
        std::cerr << "Warning: No nearest vertex found for position ("
                << position.x() << ", " << position.y() << ", " << position.z() << ")" << std::endl;
    }
    
    return nearestIdx;
}

std::vector<int> SphericalGrid::getVerticesInRegion(const BoundingBox& region) const {
    // Convert region bounds to 3D points
    std::vector<Vec3f> boundaryPoints;
    
    // Add corners
    boundaryPoints.push_back(latLonToCartesian(region.latMin, region.lonMin));
    boundaryPoints.push_back(latLonToCartesian(region.latMin, region.lonMax));
    boundaryPoints.push_back(latLonToCartesian(region.latMax, region.lonMin));
    boundaryPoints.push_back(latLonToCartesian(region.latMax, region.lonMax));
    
    // Add middle points on the edges
    boundaryPoints.push_back(latLonToCartesian(region.latMin, (region.lonMin + region.lonMax) * 0.5f));
    boundaryPoints.push_back(latLonToCartesian(region.latMax, (region.lonMin + region.lonMax) * 0.5f));
    boundaryPoints.push_back(latLonToCartesian((region.latMin + region.latMax) * 0.5f, region.lonMin));
    boundaryPoints.push_back(latLonToCartesian((region.latMin + region.latMax) * 0.5f, region.lonMax));
    
    // Add center point
    boundaryPoints.push_back(latLonToCartesian((region.latMin + region.latMax) * 0.5f, 
                                             (region.lonMin + region.lonMax) * 0.5f));
    
    // Find min/max for the AABB
    Vec3f minPoint = boundaryPoints[0];
    Vec3f maxPoint = boundaryPoints[0];
    
    for (const auto& point : boundaryPoints) {
        minPoint = minPoint.cwiseMin(point);
        maxPoint = maxPoint.cwiseMax(point);
    }
    
    // Add a small margin to ensure we capture all relevant vertices
    Vec3f margin = (maxPoint - minPoint) * 0.1f;
    minPoint -= margin;
    maxPoint += margin;
    // Query the spatial index
    std::vector<int> indices;
    if (spatialIndex) {
        AABB aabb(minPoint, maxPoint);
        indices = spatialIndex->query(aabb);
        std::cout << "Found " << indices.size() << " vertices in spatial query" << std::endl;
    } else {
        std::cerr << "Warning: Spatial index not initialized in getVerticesInRegion" << std::endl;
        return indices; // Return empty vector
    }
    
    
    // Filter vertices that are actually inside the region
    std::vector<int> result;
    for (int idx : indices) {
        auto [lat, lon] = cartesianToLatLon(vertices[idx]);
        
        // Handle wraparound for longitude
        if (lon < region.lonMin && region.lonMax > 160.0f) {
            lon += 360.0f;
        } else if (lon > region.lonMax && region.lonMin < -160.0f) {
            lon -= 360.0f;
        }
        
        if (lat >= region.latMin && lat <= region.latMax && lon >= region.lonMin && lon <= region.lonMax) {
            result.push_back(idx);
        }
    }
    
    return result;
}

void SphericalGrid::subdivideRegion(const BoundingBox& region, int additionalLevels) {
    // Not yet implemented - this is a more complex algorithm that will be added later
    // It would involve selecting faces in the region and subdividing only those,
    // then handling transitions at the boundaries
    
    // For now, just subdivide the entire grid
    subdivide(additionalLevels);
}

int SphericalGrid::getResolution() const {
    return resolution;
}

float SphericalGrid::getRadius() const {
    return radius;
}

float SphericalGrid::calculateDistance(int vertexIdx1, int vertexIdx2) const {
    if (vertexIdx1 < 0 || vertexIdx1 >= static_cast<int>(vertices.size()) ||
        vertexIdx2 < 0 || vertexIdx2 >= static_cast<int>(vertices.size())) {
        throw std::out_of_range("Vertex index out of range");
    }
    
    return sphericalDistance(vertices[vertexIdx1], vertices[vertexIdx2]) * radius;
}

float SphericalGrid::sphericalDistance(const Vec3f& p1, const Vec3f& p2) const {
    // Normalize vectors to unit sphere
    Vec3f v1 = p1.normalized();
    Vec3f v2 = p2.normalized();
    
    // Calculate dot product and clamp to valid range for acos
    float dot = v1.dot(v2);
    dot = std::max(-1.0f, std::min(1.0f, dot));
    
    // Return great circle distance (angle in radians)
    return std::acos(dot);
}

} // namespace aeonterracpp