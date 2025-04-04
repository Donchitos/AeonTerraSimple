#pragma once

#include "aeonterracpp/core/vector_types.h"
#include <vector>
#include <array>
#include <memory>

namespace aeonterracpp {

/**
 * @brief Abstract base class for spatial indexing
 */
class SpatialIndex {
public:
    virtual ~SpatialIndex() = default;
    
    /**
     * @brief Insert a vertex into the spatial index
     * 
     * @param index Index of the vertex
     * @param position 3D position of the vertex
     */
    virtual void insert(int index, const Vec3f& position) = 0;
    
    /**
     * @brief Query for vertices within a bounding box
     * 
     * @param aabb Axis-aligned bounding box
     * @return Indices of vertices within the box
     */
    virtual std::vector<int> query(const AABB& aabb) const = 0;
    
    /**
     * @brief Find the nearest vertex to a given position
     * 
     * @param position 3D position to query
     * @return Index of the nearest vertex
     */
    virtual int findNearest(const Vec3f& position) const = 0;
};

/**
 * @brief Octree-based spatial index for efficient queries
 */
class OctreeSpatialIndex : public SpatialIndex {
public:
    /**
     * @brief Construct a new Octree Spatial Index
     * 
     * @param size Size of the root node (should encompass all vertices)
     * @param maxDepth Maximum depth of the octree
     * @param maxVerticesPerNode Maximum vertices before splitting a node
     */
    OctreeSpatialIndex(float size, int maxDepth = 8, int maxVerticesPerNode = 16);
    
    void insert(int index, const Vec3f& position) override;
    std::vector<int> query(const AABB& aabb) const override;
    int findNearest(const Vec3f& position) const override;
    
private:
    struct Node {
        AABB bounds;
        std::vector<std::pair<int, Vec3f>> vertices;
        std::array<std::unique_ptr<Node>, 8> children;
        bool isLeaf = true;
        
        Node(const AABB& _bounds) : bounds(_bounds) {}
    };
    
    std::unique_ptr<Node> root;
    int maxDepth;
    int maxVerticesPerNode;
    
    void insertIntoNode(Node* node, int index, const Vec3f& position, int depth);
    void splitNode(Node* node);
    int getOctant(const Vec3f& position, const Vec3f& center) const;
    void queryNode(const Node* node, const AABB& aabb, std::vector<int>& result) const;
    void findNearestInNode(const Node* node, const Vec3f& position, int& nearestIndex, float& minDistSquared) const;
};

} // namespace aeonterracpp