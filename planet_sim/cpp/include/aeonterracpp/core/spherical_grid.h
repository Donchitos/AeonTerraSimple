#pragma once

#include "aeonterracpp/core/vector_types.h"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aeonterracpp {

/**
 * @brief A hash function for pairs of integers
 * 
 * Used for caching edge midpoints during subdivision.
 */
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Forward declaration
class SpatialIndex;

/**
 * @brief Represents a planet's surface as a spherical mesh
 * 
 * The SphericalGrid is based on a subdivided icosahedron, providing a nearly
 * uniform sampling of the sphere. It manages the geometry of the planet and
 * provides operations for spatial queries and coordinate transformations.
 */
class SphericalGrid {
public:
    /**
     * @brief Construct a new SphericalGrid
     * 
     * @param radius Planet radius in kilometers
     * @param resolution Detail level (0-7), higher = more vertices
     */
    SphericalGrid(float radius = 6371.0f, int resolution = 5);
    
    /**
     * @brief Copy constructor is deleted due to unique_ptr member
     */
    SphericalGrid(const SphericalGrid&) = delete;
    
    /**
     * @brief Copy assignment is deleted due to unique_ptr member
     */
    SphericalGrid& operator=(const SphericalGrid&) = delete;
    
    /**
     * @brief Move constructor
     */
    SphericalGrid(SphericalGrid&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    SphericalGrid& operator=(SphericalGrid&& other) noexcept;
    
    /**
     * @brief Destructor
     */
    ~SphericalGrid();
    
    /**
     * @brief Get the number of vertices in the grid
     */
    size_t getVertexCount() const;
    
    /**
     * @brief Get the number of faces in the grid
     */
    size_t getFaceCount() const;
    
    /**
     * @brief Get a specific vertex by index
     * 
     * @param index Vertex index
     * @return The 3D position of the vertex
     */
    const Vec3f& getVertex(size_t index) const;
    
    /**
     * @brief Get a specific face by index
     * 
     * @param index Face index
     * @return A vector of 3 vertex indices representing the triangular face
     */
    const Vec3i& getFace(size_t index) const;
    
    /**
     * @brief Get the indices of all vertices adjacent to a given vertex
     * 
     * @param vertexIdx Index of the vertex
     * @return Vector of indices of neighboring vertices
     */
    const std::vector<int>& getNeighbors(int vertexIdx) const;
    
    /**
     * @brief Get a list of vertices within a region
     * 
     * @param region Bounding box in latitude/longitude coordinates
     * @return Vector of vertex indices within the region
     */
    std::vector<int> getVerticesInRegion(const BoundingBox& region) const;
    
    /**
     * @brief Subdivide the grid to increase resolution
     * 
     * @param levels Number of subdivision levels to perform
     */
    void subdivide(int levels);
    
    /**
     * @brief Subdivide only a specific region of the grid
     * 
     * @param region The region to subdivide in lat/lon coordinates
     * @param additionalLevels Number of additional subdivision levels
     */
    void subdivideRegion(const BoundingBox& region, int additionalLevels);
    
    /**
     * @brief Convert latitude and longitude to a 3D position
     * 
     * @param lat Latitude in degrees
     * @param lon Longitude in degrees
     * @return 3D position on the sphere
     */
    Vec3f latLonToCartesian(float lat, float lon) const;
    
    /**
     * @brief Convert a 3D position to latitude and longitude
     * 
     * @param position 3D position
     * @return std::pair<float, float> (latitude, longitude) in degrees
     */
    std::pair<float, float> cartesianToLatLon(const Vec3f& position) const;
    
    /**
     * @brief Find the vertex nearest to a given lat/lon position
     * 
     * @param lat Latitude in degrees
     * @param lon Longitude in degrees
     * @return Index of the nearest vertex
     */
    int findNearestVertex(float lat, float lon) const;
    
    /**
     * @brief Get the resolution level of the grid
     */
    int getResolution() const;
    
    /**
     * @brief Get the radius of the planet
     */
    float getRadius() const;
    
    /**
     * @brief Calculate the great-circle distance between two vertices
     * 
     * @param vertexIdx1 Index of first vertex
     * @param vertexIdx2 Index of second vertex
     * @return Distance in kilometers
     */
    float calculateDistance(int vertexIdx1, int vertexIdx2) const;
    
private:
    // Core data
    std::vector<Vec3f> vertices;
    std::vector<Vec3i> faces;
    
    // Neighbor relationships
    std::vector<std::vector<int>> neighbors;
    
    // Spatial acceleration
    std::unique_ptr<SpatialIndex> spatialIndex;
    
    // Properties (order matters for initialization)
    float radius;
    int resolution;
    
    // Private methods
    void createIcosahedron();
    void rebuildNeighborList();
    void updateSpatialIndex();
    int getOrCreateMidpoint(int v0, int v1,
                           std::vector<Vec3f>& newVertices,
                           std::unordered_map<std::pair<int, int>, int, PairHash>& cache);
    float sphericalDistance(const Vec3f& p1, const Vec3f& p2) const;
};

} // namespace aeonterracpp