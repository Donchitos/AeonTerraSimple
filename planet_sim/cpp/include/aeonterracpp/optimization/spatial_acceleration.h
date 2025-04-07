#pragma once

#include "aeonterracpp/core/planet.h"
#include "aeonterracpp/core/vector_types.h"
#include <vector>
#include <array>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>

namespace aeonterracpp {

/**
 * @brief Type of spatial acceleration structure
 */
enum class SpatialAccelerationType {
    None,               // No acceleration (linear search)
    Grid,               // Regular 3D grid
    Octree,             // Octree space partitioning
    KdTree,             // K-d tree space partitioning
    BVH                 // Bounding volume hierarchy
};

/**
 * @brief Parameters for spatial acceleration
 */
struct SpatialAccelerationParams {
    SpatialAccelerationType type = SpatialAccelerationType::Grid;
    int maxDepth = 8;           // Maximum depth for hierarchical structures
    int maxElementsPerCell = 32; // Maximum elements per leaf node
    float gridResolution = 1.0f; // Cell size for grid-based acceleration
    bool parallelConstruction = true; // Build acceleration structure in parallel
    bool enableCaching = true;  // Cache query results
    int numThreads = 0;         // Number of threads (0 = use hardware concurrency)
};

/**
 * @brief Abstract base class for spatial acceleration structures
 * 
 * This class provides a common interface for different spatial
 * acceleration structures used to optimize spatial queries
 * across the planet surface.
 */
class SpatialAcceleration {
public:
    /**
     * @brief Construct a new Spatial Acceleration
     * 
     * @param planet Reference to the planet
     */
    SpatialAcceleration(const Planet& planet);
    
    /**
     * @brief Destroy the Spatial Acceleration object
     */
    virtual ~SpatialAcceleration() = default;
    
    /**
     * @brief Build the acceleration structure
     * 
     * @param params Build parameters
     */
    virtual void build(const SpatialAccelerationParams& params = SpatialAccelerationParams()) = 0;
    
    /**
     * @brief Update the acceleration structure after planet changes
     * 
     * @param changedVertices Indices of vertices that changed
     */
    virtual void update(const std::vector<int>& changedVertices) = 0;
    
    /**
     * @brief Find nearest vertex to a point
     * 
     * @param position Query position
     * @return int Index of nearest vertex
     */
    virtual int findNearestVertex(const Vec3f& position) const = 0;
    
    /**
     * @brief Find vertices within radius
     * 
     * @param position Center position
     * @param radius Search radius
     * @return std::vector<int> Indices of vertices within radius
     */
    virtual std::vector<int> findVerticesInRadius(const Vec3f& position, float radius) const = 0;
    
    /**
     * @brief Find k nearest vertices
     * 
     * @param position Query position
     * @param k Number of nearest neighbors to find
     * @return std::vector<int> Indices of k nearest vertices
     */
    virtual std::vector<int> findKNearestVertices(const Vec3f& position, int k) const = 0;
    
    /**
     * @brief Get acceleration structure type
     * 
     * @return SpatialAccelerationType Type of this acceleration structure
     */
    virtual SpatialAccelerationType getType() const = 0;
    
    /**
     * @brief Get build parameters
     * 
     * @return const SpatialAccelerationParams& Current parameters
     */
    const SpatialAccelerationParams& getParams() const;
    
    /**
     * @brief Get memory usage in bytes
     * 
     * @return size_t Memory usage
     */
    virtual size_t getMemoryUsage() const = 0;
    
    /**
     * @brief Get average query time
     * 
     * @return double Average query time in microseconds
     */
    double getAverageQueryTime() const;
    
    /**
     * @brief Clear query time statistics
     */
    void clearQueryTimeStats();
    
    /**
     * @brief Factory method to create acceleration structure
     * 
     * @param type Type of acceleration structure to create
     * @param planet Reference to the planet
     * @param params Build parameters
     * @return std::shared_ptr<SpatialAcceleration> Created acceleration structure
     */
    static std::shared_ptr<SpatialAcceleration> create(
        SpatialAccelerationType type,
        const Planet& planet,
        const SpatialAccelerationParams& params = SpatialAccelerationParams());
    
protected:
    const Planet& planet;
    SpatialAccelerationParams params;
    
    // Query time statistics
    mutable std::vector<double> queryTimes;
    mutable std::mutex statsMutex;
    
    /**
     * @brief Record query time
     * 
     * @param microseconds Query time in microseconds
     */
    void recordQueryTime(double microseconds) const;
};

/**
 * @brief Grid-based spatial acceleration
 * 
 * This implementation divides 3D space into a regular grid
 * for fast spatial lookups. Suitable for uniformly distributed
 * points across a sphere.
 */
class GridAcceleration : public SpatialAcceleration {
public:
    /**
     * @brief Construct a new Grid Acceleration
     * 
     * @param planet Reference to the planet
     */
    GridAcceleration(const Planet& planet);
    
    /**
     * @brief Build the grid acceleration structure
     * 
     * @param params Build parameters
     */
    void build(const SpatialAccelerationParams& params) override;
    
    /**
     * @brief Update grid after planet changes
     * 
     * @param changedVertices Indices of vertices that changed
     */
    void update(const std::vector<int>& changedVertices) override;
    
    /**
     * @brief Find nearest vertex to a point
     * 
     * @param position Query position
     * @return int Index of nearest vertex
     */
    int findNearestVertex(const Vec3f& position) const override;
    
    /**
     * @brief Find vertices within radius
     * 
     * @param position Center position
     * @param radius Search radius
     * @return std::vector<int> Indices of vertices within radius
     */
    std::vector<int> findVerticesInRadius(const Vec3f& position, float radius) const override;
    
    /**
     * @brief Find k nearest vertices
     * 
     * @param position Query position
     * @param k Number of nearest neighbors to find
     * @return std::vector<int> Indices of k nearest vertices
     */
    std::vector<int> findKNearestVertices(const Vec3f& position, int k) const override;
    
    /**
     * @brief Get acceleration structure type
     * 
     * @return SpatialAccelerationType Type of this acceleration structure
     */
    SpatialAccelerationType getType() const override;
    
    /**
     * @brief Get memory usage in bytes
     * 
     * @return size_t Memory usage
     */
    size_t getMemoryUsage() const override;
    
private:
    // Grid dimensions
    int gridSizeX, gridSizeY, gridSizeZ;
    float cellSize;
    Vec3f gridMin;
    
    // Cells containing vertex indices
    std::vector<std::vector<int>> cells;
    
    // Cell lookup cache
    mutable std::unordered_map<Vec3i, std::vector<int>, Vec3iHash> cellCache;
    
    /**
     * @brief Convert position to grid cell index
     * 
     * @param position 3D position
     * @return Vec3i Grid cell indices
     */
    Vec3i positionToCell(const Vec3f& position) const;
    
    /**
     * @brief Get cells that might contain points within radius
     * 
     * @param position Center position
     * @param radius Search radius
     * @return std::vector<Vec3i> Cell indices
     */
    std::vector<Vec3i> getCellsInRadius(const Vec3f& position, float radius) const;
    
    /**
     * @brief Get vertices in a specific cell
     * 
     * @param cellIdx Cell index
     * @return const std::vector<int>& Vertex indices in cell
     */
    const std::vector<int>& getVerticesInCell(const Vec3i& cellIdx) const;
    
    /**
     * @brief Convert cell index to linear index
     * 
     * @param cell Cell index
     * @return int Linear index
     */
    int cellToLinearIndex(const Vec3i& cell) const;
    
    /**
     * @brief Check if cell is valid
     * 
     * @param cell Cell index
     * @return true Cell is within bounds
     * @return false Cell is out of bounds
     */
    bool isValidCell(const Vec3i& cell) const;
};

/**
 * @brief Octree-based spatial acceleration
 * 
 * This implementation uses an octree for spatial partitioning.
 * Suitable for adaptively handling non-uniform distributions.
 */
class OctreeAcceleration : public SpatialAcceleration {
public:
    /**
     * @brief Construct a new Octree Acceleration
     * 
     * @param planet Reference to the planet
     */
    OctreeAcceleration(const Planet& planet);
    
    /**
     * @brief Build the octree acceleration structure
     * 
     * @param params Build parameters
     */
    void build(const SpatialAccelerationParams& params) override;
    
    /**
     * @brief Update octree after planet changes
     * 
     * @param changedVertices Indices of vertices that changed
     */
    void update(const std::vector<int>& changedVertices) override;
    
    /**
     * @brief Find nearest vertex to a point
     * 
     * @param position Query position
     * @return int Index of nearest vertex
     */
    int findNearestVertex(const Vec3f& position) const override;
    
    /**
     * @brief Find vertices within radius
     * 
     * @param position Center position
     * @param radius Search radius
     * @return std::vector<int> Indices of vertices within radius
     */
    std::vector<int> findVerticesInRadius(const Vec3f& position, float radius) const override;
    
    /**
     * @brief Find k nearest vertices
     * 
     * @param position Query position
     * @param k Number of nearest neighbors to find
     * @return std::vector<int> Indices of k nearest vertices
     */
    std::vector<int> findKNearestVertices(const Vec3f& position, int k) const override;
    
    /**
     * @brief Get acceleration structure type
     * 
     * @return SpatialAccelerationType Type of this acceleration structure
     */
    SpatialAccelerationType getType() const override;
    
    /**
     * @brief Get memory usage in bytes
     * 
     * @return size_t Memory usage
     */
    size_t getMemoryUsage() const override;
    
private:
    struct OctreeNode {
        Vec3f center;
        float halfSize;
        std::vector<int> vertexIndices;
        std::array<std::shared_ptr<OctreeNode>, 8> children;
        
        OctreeNode(const Vec3f& center, float halfSize)
            : center(center), halfSize(halfSize) {}
        
        bool isLeaf() const {
            return children[0] == nullptr;
        }
    };
    
    std::shared_ptr<OctreeNode> root;
    
    /**
     * @brief Build octree node recursively
     * 
     * @param node Node to build
     * @param vertexIndices Vertex indices to add to node
     * @param depth Current depth
     */
    void buildNode(std::shared_ptr<OctreeNode> node, 
                  const std::vector<int>& vertexIndices, 
                  int depth);
    
    /**
     * @brief Find nearest vertex in octree node
     * 
     * @param node Current node
     * @param position Query position
     * @param bestDistance Best distance so far
     * @param bestIdx Best vertex index so far
     */
    void findNearestInNode(const std::shared_ptr<OctreeNode>& node, 
                          const Vec3f& position,
                          float& bestDistance,
                          int& bestIdx) const;
    
    /**
     * @brief Find vertices within radius in octree node
     * 
     * @param node Current node
     * @param position Query position
     * @param radius Search radius
     * @param result Result vector
     */
    void findInRadiusInNode(const std::shared_ptr<OctreeNode>& node,
                           const Vec3f& position,
                           float radius,
                           std::vector<int>& result) const;
    
    /**
     * @brief Calculate memory usage of a node and its children
     * 
     * @param node Node to calculate
     * @return size_t Memory usage in bytes
     */
    size_t calculateNodeMemory(const std::shared_ptr<OctreeNode>& node) const;
};

/**
 * @brief Adaptive Mesh-based acceleration
 * 
 * Optimized for spherical meshes by partitioning the sphere
 * into a hierarchical quadrilateral mesh.
 */
class AdaptiveMeshAcceleration : public SpatialAcceleration {
public:
    /**
     * @brief Construct a new Adaptive Mesh Acceleration
     * 
     * @param planet Reference to the planet
     */
    AdaptiveMeshAcceleration(const Planet& planet);
    
    /**
     * @brief Build the adaptive mesh acceleration structure
     * 
     * @param params Build parameters
     */
    void build(const SpatialAccelerationParams& params) override;
    
    /**
     * @brief Update mesh after planet changes
     * 
     * @param changedVertices Indices of vertices that changed
     */
    void update(const std::vector<int>& changedVertices) override;
    
    /**
     * @brief Find nearest vertex to a point
     * 
     * @param position Query position
     * @return int Index of nearest vertex
     */
    int findNearestVertex(const Vec3f& position) const override;
    
    /**
     * @brief Find vertices within radius
     * 
     * @param position Center position
     * @param radius Search radius
     * @return std::vector<int> Indices of vertices within radius
     */
    std::vector<int> findVerticesInRadius(const Vec3f& position, float radius) const override;
    
    /**
     * @brief Find k nearest vertices
     * 
     * @param position Query position
     * @param k Number of nearest neighbors to find
     * @return std::vector<int> Indices of k nearest vertices
     */
    std::vector<int> findKNearestVertices(const Vec3f& position, int k) const override;
    
    /**
     * @brief Get acceleration structure type
     * 
     * @return SpatialAccelerationType Type of this acceleration structure
     */
    SpatialAccelerationType getType() const override;
    
    /**
     * @brief Get memory usage in bytes
     * 
     * @return size_t Memory usage
     */
    size_t getMemoryUsage() const override;
    
private:
    // Implementation details would go here
    // This would use a hierarchical structure optimized for the sphere geometry
};

} // namespace aeonterracpp