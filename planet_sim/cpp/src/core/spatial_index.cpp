#include "aeonterracpp/core/spatial_index.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace aeonterracpp {

// OctreeSpatialIndex implementation
OctreeSpatialIndex::OctreeSpatialIndex(float size, int maxDepth, int maxVerticesPerNode)
    : maxDepth(maxDepth), maxVerticesPerNode(maxVerticesPerNode) {
    // Create root node with bounds centered at origin
    Vec3f halfSize(size * 0.5f, size * 0.5f, size * 0.5f);
    AABB bounds(Vec3f(-halfSize), Vec3f(halfSize));
    root = std::make_unique<Node>(bounds);
}

void OctreeSpatialIndex::insert(int index, const Vec3f& position) {
    insertIntoNode(root.get(), index, position, 0);
}

void OctreeSpatialIndex::insertIntoNode(Node* node, int index, const Vec3f& position, int depth) {
    // Check if position is inside node bounds
    if (!node->bounds.contains(position)) {
        return;
    }
    
    // If node is a leaf and not full, or we've reached max depth, add vertex to this node
    if ((node->isLeaf && node->vertices.size() < maxVerticesPerNode) || depth >= maxDepth) {
        node->vertices.emplace_back(index, position);
        return;
    }
    
    // If node is a leaf but full, split it
    if (node->isLeaf) {
        splitNode(node);
    }
    
    // Find appropriate child node and insert there
    Vec3f center = node->bounds.center();
    int octant = getOctant(position, center);
    
    insertIntoNode(node->children[octant].get(), index, position, depth + 1);
}

void OctreeSpatialIndex::splitNode(Node* node) {
    // Mark as not a leaf
    node->isLeaf = false;
    
    // Calculate center of node
    Vec3f center = node->bounds.center();
    Vec3f halfSize = node->bounds.size() * 0.5f;
    
    // Create 8 child nodes
    for (int i = 0; i < 8; ++i) {
        Vec3f minCorner = node->bounds.min;
        Vec3f maxCorner = node->bounds.max;
        
        // Adjust min/max corners based on octant
        if (i & 1) minCorner.x() = center.x(); else maxCorner.x() = center.x();
        if (i & 2) minCorner.y() = center.y(); else maxCorner.y() = center.y();
        if (i & 4) minCorner.z() = center.z(); else maxCorner.z() = center.z();
        
        // Create child node
        node->children[i] = std::make_unique<Node>(AABB(minCorner, maxCorner));
    }
    
    // Redistribute vertices to children
    for (const auto& vertex : node->vertices) {
        int octant = getOctant(vertex.second, center);
        node->children[octant]->vertices.push_back(vertex);
    }
    
    // Clear vertices from this node
    node->vertices.clear();
}

int OctreeSpatialIndex::getOctant(const Vec3f& position, const Vec3f& center) const {
    int octant = 0;
    if (position.x() >= center.x()) octant |= 1;
    if (position.y() >= center.y()) octant |= 2;
    if (position.z() >= center.z()) octant |= 4;
    return octant;
}

std::vector<int> OctreeSpatialIndex::query(const AABB& aabb) const {
    std::vector<int> result;
    queryNode(root.get(), aabb, result);
    return result;
}

void OctreeSpatialIndex::queryNode(const Node* node, const AABB& aabb, std::vector<int>& result) const {
    // Skip if node bounds don't intersect query bounds
    if (!node->bounds.intersects(aabb)) {
        return;
    }
    
    // Check vertices in this node
    for (const auto& vertex : node->vertices) {
        if (aabb.contains(vertex.second)) {
            result.push_back(vertex.first);
        }
    }
    
    // If this is not a leaf, check children
    if (!node->isLeaf) {
        for (const auto& child : node->children) {
            if (child) {
                queryNode(child.get(), aabb, result);
            }
        }
    }
}

int OctreeSpatialIndex::findNearest(const Vec3f& position) const {
    int nearestIndex = -1;
    
    // Check if the tree is empty
    if (!root || root->vertices.empty()) {
        return nearestIndex;
    }
    
    float minDistSquared = std::numeric_limits<float>::max();
    findNearestInNode(root.get(), position, nearestIndex, minDistSquared);
    
    return nearestIndex;
}

void OctreeSpatialIndex::findNearestInNode(const Node* node, const Vec3f& position, 
                                          int& nearestIndex, float& minDistSquared) const {
    // Check vertices in this node
    for (const auto& vertex : node->vertices) {
        float distSquared = (vertex.second - position).squaredNorm();
        if (distSquared < minDistSquared) {
            minDistSquared = distSquared;
            nearestIndex = vertex.first;
        }
    }
    
    // If this is a leaf node, we're done
    if (node->isLeaf) {
        return;
    }
    
    // Find octant that contains position
    Vec3f center = node->bounds.center();
    int octant = getOctant(position, center);
    
    // Check closest child first
    if (node->children[octant]) {
        findNearestInNode(node->children[octant].get(), position, nearestIndex, minDistSquared);
    }
    
    // Check other children if they might contain closer points
    for (int i = 0; i < 8; ++i) {
        if (i != octant && node->children[i]) {
            // Calculate minimum possible squared distance to this child node
            float minPossibleDistSquared = 0.0f;
            
            // Calculate distance to each face of the child node's bounds
            const AABB& childBounds = node->children[i]->bounds;
            
            for (int dim = 0; dim < 3; ++dim) {
                if (position[dim] < childBounds.min[dim]) {
                    minPossibleDistSquared += (childBounds.min[dim] - position[dim]) * (childBounds.min[dim] - position[dim]);
                } else if (position[dim] > childBounds.max[dim]) {
                    minPossibleDistSquared += (position[dim] - childBounds.max[dim]) * (position[dim] - childBounds.max[dim]);
                }
            }
            
            // If this child could contain a closer point, check it
            if (minPossibleDistSquared < minDistSquared) {
                findNearestInNode(node->children[i].get(), position, nearestIndex, minDistSquared);
            }
        }
    }
}

} // namespace aeonterracpp