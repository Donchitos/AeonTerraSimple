#include "aeonterracpp/optimization/spatial_acceleration.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <cmath>
#include <queue>

namespace aeonterracpp {

// Base SpatialAcceleration implementation
SpatialAcceleration::SpatialAcceleration(const Planet& planet)
    : planet(planet) {
}

const SpatialAccelerationParams& SpatialAcceleration::getParams() const {
    return params;
}

double SpatialAcceleration::getAverageQueryTime() const {
    std::lock_guard<std::mutex> lock(statsMutex);
    
    if (queryTimes.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double time : queryTimes) {
        sum += time;
    }
    
    return sum / queryTimes.size();
}

void SpatialAcceleration::clearQueryTimeStats() {
    std::lock_guard<std::mutex> lock(statsMutex);
    queryTimes.clear();
}

void SpatialAcceleration::recordQueryTime(double microseconds) const {
    std::lock_guard<std::mutex> lock(statsMutex);
    queryTimes.push_back(microseconds);
    
    // Limit the number of recorded times to prevent unbounded growth
    if (queryTimes.size() > 1000) {
        queryTimes.erase(queryTimes.begin());
    }
}

std::shared_ptr<SpatialAcceleration> SpatialAcceleration::create(
    SpatialAccelerationType type,
    const Planet& planet,
    const SpatialAccelerationParams& params) {
    
    std::shared_ptr<SpatialAcceleration> acceleration;
    
    switch (type) {
        case SpatialAccelerationType::Grid:
            acceleration = std::make_shared<GridAcceleration>(planet);
            break;
        case SpatialAccelerationType::Octree:
            acceleration = std::make_shared<OctreeAcceleration>(planet);
            break;
        case SpatialAccelerationType::None:
        default:
            // Default to the simplest implementation
            acceleration = std::make_shared<GridAcceleration>(planet);
            break;
    }
    
    acceleration->build(params);
    return acceleration;
}

// Grid-based implementation
GridAcceleration::GridAcceleration(const Planet& planet)
    : SpatialAcceleration(planet), gridSizeX(0), gridSizeY(0), gridSizeZ(0), cellSize(0.0f) {
}

void GridAcceleration::build(const SpatialAccelerationParams& params) {
    this->params = params;
    
    // Determine grid dimensions based on planet size and desired resolution
    float planetRadius = planet.getRadius();
    cellSize = params.gridResolution * planetRadius / 100.0f; // Scale cell size with planet radius
    
    // Create a grid that encompasses the planet with some margin
    float gridExtent = planetRadius * 1.1f; // 10% margin
    
    gridSizeX = static_cast<int>(std::ceil(2.0f * gridExtent / cellSize));
    gridSizeY = static_cast<int>(std::ceil(2.0f * gridExtent / cellSize));
    gridSizeZ = static_cast<int>(std::ceil(2.0f * gridExtent / cellSize));
    
    gridMin = Vec3f(-gridExtent, -gridExtent, -gridExtent);
    
    // Initialize grid cells
    size_t totalCells = static_cast<size_t>(gridSizeX) * 
                         static_cast<size_t>(gridSizeY) * 
                         static_cast<size_t>(gridSizeZ);
    
    cells.resize(totalCells);
    
    // Clear any existing data
    for (auto& cell : cells) {
        cell.clear();
    }
    
    cellCache.clear();
    
    // Build the grid by distributing vertices into cells
    const SphericalGrid& grid = planet.getGrid();
    
    // Use multiple threads if enabled
    bool useParallel = params.parallelConstruction && 
                       params.numThreads != 1; // Only use parallel if not explicitly single-threaded
    
    if (useParallel) {
        // Number of threads to use
        int numThreads = params.numThreads > 0 ? 
                          params.numThreads : 
                          std::thread::hardware_concurrency();
        
        if (numThreads < 1) numThreads = 1;
        
        // Create threads
        std::vector<std::thread> threads;
        std::vector<std::vector<std::vector<int>>> threadCells(numThreads);
        
        for (int t = 0; t < numThreads; ++t) {
            threadCells[t].resize(totalCells);
            
            threads.emplace_back([this, &grid, t, numThreads, &threadCells]() {
                size_t vertexCount = grid.getVertexCount();
                for (size_t i = t; i < vertexCount; i += numThreads) {
                    Vec3f position = grid.getVertex(static_cast<int>(i));
                    Vec3i cellIdx = positionToCell(position);
                    
                    if (isValidCell(cellIdx)) {
                        int linearIdx = cellToLinearIndex(cellIdx);
                        threadCells[t][linearIdx].push_back(static_cast<int>(i));
                    }
                }
            });
        }
        
        // Wait for threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Merge results
        for (int t = 0; t < numThreads; ++t) {
            for (size_t i = 0; i < totalCells; ++i) {
                cells[i].insert(cells[i].end(), 
                                threadCells[t][i].begin(), 
                                threadCells[t][i].end());
            }
        }
    } else {
        // Single-threaded implementation
        for (size_t i = 0; i < grid.getVertexCount(); ++i) {
            Vec3f position = grid.getVertex(static_cast<int>(i));
            Vec3i cellIdx = positionToCell(position);
            
            if (isValidCell(cellIdx)) {
                int linearIdx = cellToLinearIndex(cellIdx);
                cells[linearIdx].push_back(static_cast<int>(i));
            }
        }
    }
    
    std::cout << "Built grid acceleration structure with dimensions " 
              << gridSizeX << "x" << gridSizeY << "x" << gridSizeZ 
              << " (cell size: " << cellSize << ")" << std::endl;
}

void GridAcceleration::update(const std::vector<int>& changedVertices) {
    // For simplicity, we rebuild the entire structure
    // A more efficient approach would be to update only the changed cells
    build(params);
}

int GridAcceleration::findNearestVertex(const Vec3f& position) const {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Get the cell containing the query position
    Vec3i cellIdx = positionToCell(position);
    
    // Initialize with worst-case values
    int bestIdx = -1;
    float bestDistSq = std::numeric_limits<float>::max();
    
    // Search in the cell containing the position
    if (isValidCell(cellIdx)) {
        const auto& verticesInCell = getVerticesInCell(cellIdx);
        
        for (int vertexIdx : verticesInCell) {
            Vec3f vertexPos = planet.getGrid().getVertex(vertexIdx);
            float distSq = (vertexPos - position).squaredNorm();
            
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestIdx = vertexIdx;
            }
        }
    }
    
    // If no vertices in the cell, or the cell is invalid,
    // search in neighboring cells in expanding rings until we find vertices
    if (bestIdx == -1) {
        int maxRings = std::max(std::max(gridSizeX, gridSizeY), gridSizeZ);
        
        for (int ring = 1; ring < maxRings && bestIdx == -1; ++ring) {
            // Get all cells in this ring
            std::vector<Vec3i> ringCells;
            
            // Add all cells in the current ring
            for (int dx = -ring; dx <= ring; ++dx) {
                for (int dy = -ring; dy <= ring; ++dy) {
                    for (int dz = -ring; dz <= ring; ++dz) {
                        // Only consider cells on the surface of the ring
                        if (std::abs(dx) == ring || std::abs(dy) == ring || std::abs(dz) == ring) {
                            Vec3i neighborCell(cellIdx.x() + dx, cellIdx.y() + dy, cellIdx.z() + dz);
                            
                            if (isValidCell(neighborCell)) {
                                ringCells.push_back(neighborCell);
                            }
                        }
                    }
                }
            }
            
            // Check all cells in this ring
            for (const Vec3i& cell : ringCells) {
                const auto& verticesInCell = getVerticesInCell(cell);
                
                for (int vertexIdx : verticesInCell) {
                    Vec3f vertexPos = planet.getGrid().getVertex(vertexIdx);
                    float distSq = (vertexPos - position).squaredNorm();
                    
                    if (distSq < bestDistSq) {
                        bestDistSq = distSq;
                        bestIdx = vertexIdx;
                    }
                }
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = endTime - startTime;
    recordQueryTime(duration.count());
    
    return bestIdx;
}

std::vector<int> GridAcceleration::findVerticesInRadius(const Vec3f& position, float radius) const {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<int> result;
    
    // Get all cells that might contain points within radius
    std::vector<Vec3i> candidateCells = getCellsInRadius(position, radius);
    
    // Check all vertices in these cells
    float radiusSq = radius * radius;
    
    for (const Vec3i& cell : candidateCells) {
        const auto& verticesInCell = getVerticesInCell(cell);
        
        for (int vertexIdx : verticesInCell) {
            Vec3f vertexPos = planet.getGrid().getVertex(vertexIdx);
            float distSq = (vertexPos - position).squaredNorm();
            
            if (distSq <= radiusSq) {
                result.push_back(vertexIdx);
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = endTime - startTime;
    recordQueryTime(duration.count());
    
    return result;
}

std::vector<int> GridAcceleration::findKNearestVertices(const Vec3f& position, int k) const {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Priority queue to keep track of the k nearest vertices
    // We use a max heap, so the furthest vertex is always at the top
    using DistanceVertex = std::pair<float, int>;
    std::priority_queue<DistanceVertex> nearestVertices;
    
    // Get the cell containing the query position
    Vec3i cellIdx = positionToCell(position);
    
    // Start with the cell containing the position
    std::vector<Vec3i> searchCells;
    if (isValidCell(cellIdx)) {
        searchCells.push_back(cellIdx);
    }
    
    // Search in expanding rings until we have k vertices
    int ring = 1;
    int maxRings = std::max(std::max(gridSizeX, gridSizeY), gridSizeZ);
    
    while (static_cast<int>(nearestVertices.size()) < k && ring < maxRings) {
        // Add all cells in the current ring
        for (int dx = -ring; dx <= ring; ++dx) {
            for (int dy = -ring; dy <= ring; ++dy) {
                for (int dz = -ring; dz <= ring; ++dz) {
                    // Only consider cells on the surface of the ring
                    if (std::abs(dx) == ring || std::abs(dy) == ring || std::abs(dz) == ring) {
                        Vec3i neighborCell(cellIdx.x() + dx, cellIdx.y() + dy, cellIdx.z() + dz);
                        
                        if (isValidCell(neighborCell)) {
                            searchCells.push_back(neighborCell);
                        }
                    }
                }
            }
        }
        
        // Process all cells collected so far
        for (const Vec3i& cell : searchCells) {
            const auto& verticesInCell = getVerticesInCell(cell);
            
            for (int vertexIdx : verticesInCell) {
                Vec3f vertexPos = planet.getGrid().getVertex(vertexIdx);
                float distSq = (vertexPos - position).squaredNorm();
                
                // If we haven't found k vertices yet, or this vertex is closer than the furthest
                if (static_cast<int>(nearestVertices.size()) < k) {
                    nearestVertices.emplace(distSq, vertexIdx);
                } else if (distSq < nearestVertices.top().first) {
                    nearestVertices.pop();
                    nearestVertices.emplace(distSq, vertexIdx);
                }
            }
        }
        
        // Clear the search cells for the next ring
        searchCells.clear();
        ++ring;
    }
    
    // Extract results from the priority queue
    std::vector<int> result;
    result.reserve(nearestVertices.size());
    
    while (!nearestVertices.empty()) {
        result.push_back(nearestVertices.top().second);
        nearestVertices.pop();
    }
    
    // Reverse to get closest first
    std::reverse(result.begin(), result.end());
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = endTime - startTime;
    recordQueryTime(duration.count());
    
    return result;
}

SpatialAccelerationType GridAcceleration::getType() const {
    return SpatialAccelerationType::Grid;
}

size_t GridAcceleration::getMemoryUsage() const {
    // Calculate memory usage
    size_t usage = 0;
    
    // Grid structure size
    usage += sizeof(GridAcceleration);
    
    // Cell array size
    usage += cells.capacity() * sizeof(std::vector<int>);
    
    // Vertices in cells
    for (const auto& cell : cells) {
        usage += cell.capacity() * sizeof(int);
    }
    
    // Cache map
    usage += cellCache.size() * (sizeof(Vec3i) + sizeof(std::vector<int>));
    for (const auto& entry : cellCache) {
        usage += entry.second.capacity() * sizeof(int);
    }
    
    return usage;
}

Vec3i GridAcceleration::positionToCell(const Vec3f& position) const {
    Vec3f relativePos = position - gridMin;
    
    int x = static_cast<int>(std::floor(relativePos.x() / cellSize));
    int y = static_cast<int>(std::floor(relativePos.y() / cellSize));
    int z = static_cast<int>(std::floor(relativePos.z() / cellSize));
    
    return Vec3i(x, y, z);
}

std::vector<Vec3i> GridAcceleration::getCellsInRadius(const Vec3f& position, float radius) const {
    std::vector<Vec3i> result;
    
    // Calculate the cell extent for the radius
    int cellRadius = static_cast<int>(std::ceil(radius / cellSize)) + 1;
    
    // Get the center cell
    Vec3i centerCell = positionToCell(position);
    
    // Add all cells within radius
    for (int dx = -cellRadius; dx <= cellRadius; ++dx) {
        for (int dy = -cellRadius; dy <= cellRadius; ++dy) {
            for (int dz = -cellRadius; dz <= cellRadius; ++dz) {
                Vec3i cell(centerCell.x() + dx, centerCell.y() + dy, centerCell.z() + dz);
                
                if (isValidCell(cell)) {
                    result.push_back(cell);
                }
            }
        }
    }
    
    return result;
}

const std::vector<int>& GridAcceleration::getVerticesInCell(const Vec3i& cellIdx) const {
    // Check if the cell data is cached
    if (params.enableCaching) {
        auto it = cellCache.find(cellIdx);
        if (it != cellCache.end()) {
            return it->second;
        }
    }
    
    // Get vertices from the grid
    int linearIdx = cellToLinearIndex(cellIdx);
    
    if (linearIdx >= 0 && linearIdx < static_cast<int>(cells.size())) {
        // Cache the result if caching is enabled
        if (params.enableCaching) {
            cellCache[cellIdx] = cells[linearIdx];
        }
        
        return cells[linearIdx];
    }
    
    // Return an empty vector for invalid cells
    static const std::vector<int> emptyVector;
    return emptyVector;
}

int GridAcceleration::cellToLinearIndex(const Vec3i& cell) const {
    if (!isValidCell(cell)) {
        return -1;
    }
    
    return cell.x() + cell.y() * gridSizeX + cell.z() * gridSizeX * gridSizeY;
}

bool GridAcceleration::isValidCell(const Vec3i& cell) const {
    return cell.x() >= 0 && cell.x() < gridSizeX &&
           cell.y() >= 0 && cell.y() < gridSizeY &&
           cell.z() >= 0 && cell.z() < gridSizeZ;
}

// Octree implementation
OctreeAcceleration::OctreeAcceleration(const Planet& planet)
    : SpatialAcceleration(planet), root(nullptr) {
}

void OctreeAcceleration::build(const SpatialAccelerationParams& params) {
    this->params = params;
    
    // Clear any existing octree
    root.reset();
    
    // Create the root node that encompasses the entire planet
    float planetRadius = planet.getRadius();
    float rootSize = planetRadius * 2.1f; // Add some margin
    
    root = std::make_shared<OctreeNode>(Vec3f(0, 0, 0), rootSize / 2.0f);
    
    // Collect all vertex indices
    std::vector<int> allVertices;
    allVertices.reserve(planet.getVertexCount());
    
    for (size_t i = 0; i < planet.getVertexCount(); ++i) {
        allVertices.push_back(static_cast<int>(i));
    }
    
    // Build the octree recursively
    buildNode(root, allVertices, 0);
    
    std::cout << "Built octree acceleration structure with max depth " 
              << params.maxDepth << std::endl;
}

void OctreeAcceleration::update(const std::vector<int>& changedVertices) {
    // For simplicity, we rebuild the entire structure
    // A more efficient approach would be to update only affected nodes
    build(params);
}

int OctreeAcceleration::findNearestVertex(const Vec3f& position) const {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    int bestIdx = -1;
    float bestDistance = std::numeric_limits<float>::max();
    
    findNearestInNode(root, position, bestDistance, bestIdx);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = endTime - startTime;
    recordQueryTime(duration.count());
    
    return bestIdx;
}

std::vector<int> OctreeAcceleration::findVerticesInRadius(const Vec3f& position, float radius) const {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<int> result;
    findInRadiusInNode(root, position, radius, result);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = endTime - startTime;
    recordQueryTime(duration.count());
    
    return result;
}

std::vector<int> OctreeAcceleration::findKNearestVertices(const Vec3f& position, int k) const {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Priority queue to keep track of the k nearest vertices
    // We use a max heap, so the furthest vertex is always at the top
    using DistanceVertex = std::pair<float, int>;
    std::priority_queue<DistanceVertex> nearestVertices;
    
    // Process the root node first
    std::queue<std::shared_ptr<OctreeNode>> nodeQueue;
    nodeQueue.push(root);
    
    while (!nodeQueue.empty()) {
        auto node = nodeQueue.front();
        nodeQueue.pop();
        
        // Calculate distance from query point to node center
        float distToCenter = (position - node->center).norm();
        
        // If we already have k vertices and the closest possible point in this node
        // is farther than our current k-th nearest, skip this node
        if (static_cast<int>(nearestVertices.size()) == k &&
            distToCenter - node->halfSize > nearestVertices.top().first) {
            continue;
        }
        
        // Process vertices in this node
        for (int vertexIdx : node->vertexIndices) {
            Vec3f vertexPos = planet.getGrid().getVertex(vertexIdx);
            float distance = (vertexPos - position).norm();
            
            if (static_cast<int>(nearestVertices.size()) < k) {
                nearestVertices.emplace(distance, vertexIdx);
            } else if (distance < nearestVertices.top().first) {
                nearestVertices.pop();
                nearestVertices.emplace(distance, vertexIdx);
            }
        }
        
        // Add child nodes to the queue
        if (!node->isLeaf()) {
            // Calculate distances to child nodes and sort them
            std::vector<std::pair<float, int>> childDistances;
            
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    float childDist = (position - node->children[i]->center).norm();
                    childDistances.emplace_back(childDist, i);
                }
            }
            
            // Sort by distance (closest first)
            std::sort(childDistances.begin(), childDistances.end());
            
            // Add children to queue in order of distance
            for (const auto& child : childDistances) {
                nodeQueue.push(node->children[child.second]);
            }
        }
    }
    
    // Extract results from the priority queue
    std::vector<int> result;
    result.reserve(nearestVertices.size());
    
    while (!nearestVertices.empty()) {
        result.push_back(nearestVertices.top().second);
        nearestVertices.pop();
    }
    
    // Reverse to get closest first
    std::reverse(result.begin(), result.end());
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = endTime - startTime;
    recordQueryTime(duration.count());
    
    return result;
}

SpatialAccelerationType OctreeAcceleration::getType() const {
    return SpatialAccelerationType::Octree;
}

size_t OctreeAcceleration::getMemoryUsage() const {
    // Calculate memory usage
    size_t usage = sizeof(OctreeAcceleration);
    
    // Add memory used by the octree
    if (root) {
        usage += calculateNodeMemory(root);
    }
    
    return usage;
}

void OctreeAcceleration::buildNode(std::shared_ptr<OctreeNode> node, 
                                 const std::vector<int>& vertexIndices, 
                                 int depth) {
    // If no vertices in this node, return
    if (vertexIndices.empty()) {
        return;
    }
    
    // If we've reached max depth or have few enough vertices, make this a leaf node
    if (depth >= params.maxDepth || 
        static_cast<int>(vertexIndices.size()) <= params.maxElementsPerCell) {
        node->vertexIndices = vertexIndices;
        return;
    }
    
    // Sort vertices into octants
    std::vector<std::vector<int>> octants(8);
    
    for (int vertexIdx : vertexIndices) {
        Vec3f position = planet.getGrid().getVertex(vertexIdx);
        
        // Determine which octant the vertex belongs to
        int octant = 0;
        if (position.x() >= node->center.x()) octant |= 1;
        if (position.y() >= node->center.y()) octant |= 2;
        if (position.z() >= node->center.z()) octant |= 4;
        
        octants[octant].push_back(vertexIdx);
    }
    
    // Create child nodes for non-empty octants
    float childHalfSize = node->halfSize * 0.5f;
    
    for (int i = 0; i < 8; ++i) {
        if (!octants[i].empty()) {
            // Calculate child center
            Vec3f childCenter = node->center;
            if (i & 1) childCenter.x() += childHalfSize;
            else childCenter.x() -= childHalfSize;
            
            if (i & 2) childCenter.y() += childHalfSize;
            else childCenter.y() -= childHalfSize;
            
            if (i & 4) childCenter.z() += childHalfSize;
            else childCenter.z() -= childHalfSize;
            
            // Create and recurse
            node->children[i] = std::make_shared<OctreeNode>(childCenter, childHalfSize);
            buildNode(node->children[i], octants[i], depth + 1);
        }
    }
}

void OctreeAcceleration::findNearestInNode(const std::shared_ptr<OctreeNode>& node, 
                                         const Vec3f& position,
                                         float& bestDistance,
                                         int& bestIdx) const {
    if (!node) return;
    
    // Check if this node could contain a closer point
    float distToCenter = (position - node->center).norm();
    if (distToCenter - node->halfSize > bestDistance) {
        return; // This node is too far away
    }
    
    // Check vertices in this node
    for (int vertexIdx : node->vertexIndices) {
        Vec3f vertexPos = planet.getGrid().getVertex(vertexIdx);
        float distance = (vertexPos - position).norm();
        
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIdx = vertexIdx;
        }
    }
    
    // If this is a leaf node, we're done
    if (node->isLeaf()) {
        return;
    }
    
    // Calculate distances to child nodes and sort them
    std::vector<std::pair<float, int>> childDistances;
    
    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) {
            float childDist = (position - node->children[i]->center).norm();
            childDistances.emplace_back(childDist, i);
        }
    }
    
    // Sort by distance (closest first)
    std::sort(childDistances.begin(), childDistances.end());
    
    // Recurse into children in order of distance
    for (const auto& child : childDistances) {
        findNearestInNode(node->children[child.second], position, bestDistance, bestIdx);
    }
}

void OctreeAcceleration::findInRadiusInNode(const std::shared_ptr<OctreeNode>& node,
                                          const Vec3f& position,
                                          float radius,
                                          std::vector<int>& result) const {
    if (!node) return;
    
    // Calculate distance from query point to node center
    float distToCenter = (position - node->center).norm();
    
    // If the node is completely outside the radius, skip it
    if (distToCenter - node->halfSize > radius) {
        return;
    }
    
    // If the node is completely inside the radius, add all vertices
    if (distToCenter + node->halfSize <= radius) {
        result.insert(result.end(), node->vertexIndices.begin(), node->vertexIndices.end());
        return;
    }
    
    // Otherwise, check each vertex individually
    for (int vertexIdx : node->vertexIndices) {
        Vec3f vertexPos = planet.getGrid().getVertex(vertexIdx);
        float distance = (vertexPos - position).norm();
        
        if (distance <= radius) {
            result.push_back(vertexIdx);
        }
    }
    
    // Recurse into children
    if (!node->isLeaf()) {
        for (const auto& child : node->children) {
            if (child) {
                findInRadiusInNode(child, position, radius, result);
            }
        }
    }
}

size_t OctreeAcceleration::calculateNodeMemory(const std::shared_ptr<OctreeNode>& node) const {
    if (!node) return 0;
    
    // Memory for this node
    size_t memory = sizeof(OctreeNode);
    
    // Memory for vertex indices
    memory += node->vertexIndices.capacity() * sizeof(int);
    
    // Memory for child nodes
    for (const auto& child : node->children) {
        if (child) {
            memory += calculateNodeMemory(child);
        }
    }
    
    return memory;
}

} // namespace aeonterracpp