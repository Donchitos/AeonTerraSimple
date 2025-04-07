#include "aeonterracpp/optimization/memory_optimization.h"
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <string.h>
#include <fcntl.h>
#include <cmath>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace aeonterracpp {

//------------------------------------------------------------------------------
// MemoryStats implementation
//------------------------------------------------------------------------------

std::string MemoryStats::formatSize(size_t bytes) {
    static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    
    size_t s = 0;
    double count = static_cast<double>(bytes);
    
    while (count >= 1024 && s < 5) {
        s++;
        count /= 1024;
    }
    
    std::ostringstream oss;
    if (count - floor(count) == 0.0) {
        oss << static_cast<size_t>(count) << " " << suffixes[s];
    } else {
        oss << std::fixed << std::setprecision(2) << count << " " << suffixes[s];
    }
    
    return oss.str();
}

void MemoryStats::reset() {
    currentUsage = 0;
    peakUsage = 0;
    totalAllocated = 0;
    totalDeallocated = 0;
    allocationCount = 0;
    deallocationCount = 0;
    largestAllocation = 0;
    categoryUsage.clear();
}

//------------------------------------------------------------------------------
// MemoryManager implementation
//------------------------------------------------------------------------------

MemoryManager& MemoryManager::getInstance() {
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager() {
    // Initialize with default configuration
    MemoryConfig defaultConfig;
    configure(defaultConfig);
}

void MemoryManager::configure(const MemoryConfig& config) {
    std::lock_guard<std::mutex> lock(mutex);
    this->config = config;
    
    // Clear pool allocators to re-create them with new config
    pools.clear();
}

const MemoryConfig& MemoryManager::getConfig() const {
    return config;
}

void* MemoryManager::allocate(size_t size, const std::string& category) {
    // Check if we're near memory limit
    if (isNearLimit()) {
        // Try to free up memory by calling cleanup callbacks
        for (const auto& callback : cleanupCallbacks) {
            callback();
        }
        
        // Check again after cleanup
        if (isNearLimit(98.0f)) {
            std::cerr << "ERROR: Memory limit reached: " 
                      << MemoryStats::formatSize(stats.currentUsage) 
                      << " of " << MemoryStats::formatSize(memoryLimit) 
                      << std::endl;
            return nullptr;
        }
    }
    
    void* ptr = nullptr;
    
    // Use appropriate allocation strategy based on configuration
    switch (config.strategy) {
        case MemoryStrategy::PoolAllocator:
            // Use pool allocator for small, fixed-size allocations
            if (size <= 1024) {
                PoolAllocator* pool = getPool(size);
                ptr = pool->allocate();
            } else {
                // Fall back to default allocator for larger allocations
                ptr = std::malloc(size);
            }
            break;
            
        case MemoryStrategy::Default:
        default:
            ptr = std::malloc(size);
            break;
    }
    
    // Update stats
    if (ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        stats.currentUsage += size;
        stats.totalAllocated += size;
        stats.allocationCount++;
        stats.peakUsage = std::max(stats.peakUsage, stats.currentUsage);
        stats.largestAllocation = std::max(stats.largestAllocation, size);
        
        if (stats.categoryUsage.find(category) == stats.categoryUsage.end()) {
            stats.categoryUsage[category] = size;
        } else {
            stats.categoryUsage[category] += size;
        }
    }
    
    return ptr;
}

void MemoryManager::deallocate(void* ptr, size_t size, const std::string& category) {
    if (!ptr) return;
    
    // Use appropriate deallocation strategy based on configuration
    switch (config.strategy) {
        case MemoryStrategy::PoolAllocator:
            // Try to use pool allocator for small, fixed-size allocations
            if (size <= 1024) {
                PoolAllocator* pool = getPool(size);
                pool->deallocate(ptr);
            } else {
                // Fall back to default deallocator for larger allocations
                std::free(ptr);
            }
            break;
            
        case MemoryStrategy::Default:
        default:
            std::free(ptr);
            break;
    }
    
    // Update stats
    std::lock_guard<std::mutex> lock(mutex);
    stats.currentUsage -= size;
    stats.totalDeallocated += size;
    stats.deallocationCount++;
    
    if (stats.categoryUsage.find(category) != stats.categoryUsage.end()) {
        if (stats.categoryUsage[category] >= size) {
            stats.categoryUsage[category] -= size;
        } else {
            // Shouldn't happen, but protect against underflow
            stats.categoryUsage[category] = 0;
        }
    }
}

void* MemoryManager::allocateAligned(size_t size, size_t alignment, const std::string& category) {
    void* ptr = nullptr;
    
    // Use platform-specific aligned allocation
#if defined(_MSC_VER)
    ptr = _aligned_malloc(size, alignment);
#elif defined(__APPLE__) || defined(__linux__)
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#else
    // Fallback: over-allocate and manually align
    void* raw = std::malloc(size + alignment);
    if (raw) {
        ptr = reinterpret_cast<void*>(
            (reinterpret_cast<uintptr_t>(raw) + alignment) & ~(alignment - 1)
        );
        
        // Store original pointer for deallocation
        void** metadata = reinterpret_cast<void**>(ptr) - 1;
        *metadata = raw;
    }
#endif
    
    // Update stats
    if (ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        stats.currentUsage += size;
        stats.totalAllocated += size;
        stats.allocationCount++;
        stats.peakUsage = std::max(stats.peakUsage, stats.currentUsage);
        
        if (stats.categoryUsage.find(category) == stats.categoryUsage.end()) {
            stats.categoryUsage[category] = size;
        } else {
            stats.categoryUsage[category] += size;
        }
    }
    
    return ptr;
}

void MemoryManager::deallocateAligned(void* ptr, size_t size, const std::string& category) {
    if (!ptr) return;
    
    // Use platform-specific aligned deallocation
#if defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(__APPLE__) || defined(__linux__)
    std::free(ptr);
#else
    // Fallback: retrieve original pointer and free it
    void** metadata = reinterpret_cast<void**>(ptr) - 1;
    void* raw = *metadata;
    std::free(raw);
#endif
    
    // Update stats
    std::lock_guard<std::mutex> lock(mutex);
    stats.currentUsage -= size;
    stats.totalDeallocated += size;
    stats.deallocationCount++;
    
    if (stats.categoryUsage.find(category) != stats.categoryUsage.end()) {
        if (stats.categoryUsage[category] >= size) {
            stats.categoryUsage[category] -= size;
        } else {
            // Shouldn't happen, but protect against underflow
            stats.categoryUsage[category] = 0;
        }
    }
}

const MemoryStats& MemoryManager::getStats() const {
    return stats;
}

void MemoryManager::resetStats() {
    std::lock_guard<std::mutex> lock(mutex);
    stats.reset();
}

void MemoryManager::setMemoryLimit(size_t limit) {
    memoryLimit = limit;
}

size_t MemoryManager::getMemoryLimit() const {
    return memoryLimit;
}

bool MemoryManager::isNearLimit(float thresholdPercent) const {
    if (memoryLimit == 0) {
        return false; // No limit set
    }
    
    float usagePercent = (static_cast<float>(stats.currentUsage) / memoryLimit) * 100.0f;
    return usagePercent >= thresholdPercent;
}

void MemoryManager::registerCleanupCallback(std::function<void()> callback) {
    cleanupCallbacks.push_back(callback);
}

MemoryManager::PoolAllocator* MemoryManager::getPool(size_t size) {
    // Round up to a power of 2
    size_t poolSize = 1;
    while (poolSize < size) {
        poolSize *= 2;
    }
    
    // Get or create pool
    auto it = pools.find(poolSize);
    if (it == pools.end()) {
        // Create a new pool
        pools[poolSize] = std::make_unique<PoolAllocator>(
            poolSize, config.poolSize
        );
        return pools[poolSize].get();
    }
    
    return it->second.get();
}

//------------------------------------------------------------------------------
// MemoryManager::PoolAllocator implementation
//------------------------------------------------------------------------------

MemoryManager::PoolAllocator::PoolAllocator(size_t blockSize, size_t poolSize)
    : blockSize(blockSize), poolSize(poolSize), memory(nullptr), 
      freeList(nullptr), numBlocks(0), numAllocated(0) {
    
    // Calculate number of blocks that fit in the pool
    numBlocks = poolSize / blockSize;
    
    // Allocate memory for the pool
    memory = static_cast<char*>(std::malloc(numBlocks * blockSize));
    
    // Initialize free list
    freeList = memory;
    
    // Link blocks in the free list
    for (size_t i = 0; i < numBlocks - 1; ++i) {
        void** next = reinterpret_cast<void**>(memory + i * blockSize);
        *next = memory + (i + 1) * blockSize;
    }
    
    // Set the last block's next pointer to nullptr
    void** last = reinterpret_cast<void**>(memory + (numBlocks - 1) * blockSize);
    *last = nullptr;
    
    std::cout << "Created memory pool with " << numBlocks 
              << " blocks of size " << blockSize << " bytes" << std::endl;
}

MemoryManager::PoolAllocator::~PoolAllocator() {
    // Free the pool memory
    if (memory) {
        std::free(memory);
        memory = nullptr;
    }
}

void* MemoryManager::PoolAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mutex);
    
    // Check if we have free blocks
    if (!freeList) {
        // No free blocks available
        return nullptr;
    }
    
    // Get the first free block
    void* block = freeList;
    
    // Update free list to point to the next free block
    freeList = *reinterpret_cast<void**>(freeList);
    
    // Update allocation count
    numAllocated++;
    
    return block;
}

void MemoryManager::PoolAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    // Check if the pointer belongs to this pool
    if (ptr < memory || ptr >= memory + numBlocks * blockSize) {
        // Not from this pool, ignore
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex);
    
    // Add the block back to the free list
    *reinterpret_cast<void**>(ptr) = freeList;
    freeList = ptr;
    
    // Update allocation count
    if (numAllocated > 0) {
        numAllocated--;
    }
}

size_t MemoryManager::PoolAllocator::available() const {
    return numBlocks - numAllocated;
}

size_t MemoryManager::PoolAllocator::total() const {
    return numBlocks;
}

//------------------------------------------------------------------------------
// VertexDataManager implementation
//------------------------------------------------------------------------------

VertexDataManager::VertexDataManager(const MemoryConfig& config)
    : config(config), positions(nullptr), normals(nullptr), 
      textureCoords(nullptr), colors(nullptr) {
}

void VertexDataManager::allocate(size_t vertexCount, 
                              bool withNormals,
                              bool withTexCoords,
                              bool withColors) {
    // Deallocate any existing data
    deallocate();
    
    this->vertexCount = vertexCount;
    
    // Allocate vertex positions (3 floats per vertex)
    positionsSize = vertexCount * 3 * sizeof(float);
    positions = static_cast<float*>(MemoryManager::getInstance().allocateAligned(
        positionsSize, config.cacheLineSize, "vertex_positions"
    ));
    
    // Allocate other arrays if needed
    if (withNormals) {
        normalsSize = vertexCount * 3 * sizeof(float);
        normals = static_cast<float*>(MemoryManager::getInstance().allocateAligned(
            normalsSize, config.cacheLineSize, "vertex_normals"
        ));
        hasNormals = true;
    }
    
    if (withTexCoords) {
        textureCoordsSize = vertexCount * 2 * sizeof(float);
        textureCoords = static_cast<float*>(MemoryManager::getInstance().allocateAligned(
            textureCoordsSize, config.cacheLineSize, "vertex_texcoords"
        ));
        hasTextureCoords = true;
    }
    
    if (withColors) {
        colorsSize = vertexCount * 4 * sizeof(float);
        colors = static_cast<float*>(MemoryManager::getInstance().allocateAligned(
            colorsSize, config.cacheLineSize, "vertex_colors"
        ));
        hasColors = true;
    }
}

void VertexDataManager::resize(size_t newVertexCount) {
    // Create a new manager with the new size
    VertexDataManager newManager(config);
    newManager.allocate(newVertexCount, hasNormals, hasTextureCoords, hasColors);
    
    // Copy existing data
    size_t copyVertexCount = std::min(vertexCount, newVertexCount);
    
    // Copy positions
    if (positions && newManager.positions) {
        memcpy(newManager.positions, positions, copyVertexCount * 3 * sizeof(float));
    }
    
    // Copy normals
    if (normals && newManager.normals) {
        memcpy(newManager.normals, normals, copyVertexCount * 3 * sizeof(float));
    }
    
    // Copy texture coordinates
    if (textureCoords && newManager.textureCoords) {
        memcpy(newManager.textureCoords, textureCoords, copyVertexCount * 2 * sizeof(float));
    }
    
    // Copy colors
    if (colors && newManager.colors) {
        memcpy(newManager.colors, colors, copyVertexCount * 4 * sizeof(float));
    }
    
    // Swap the managers
    std::swap(positions, newManager.positions);
    std::swap(normals, newManager.normals);
    std::swap(textureCoords, newManager.textureCoords);
    std::swap(colors, newManager.colors);
    
    std::swap(positionsSize, newManager.positionsSize);
    std::swap(normalsSize, newManager.normalsSize);
    std::swap(textureCoordsSize, newManager.textureCoordsSize);
    std::swap(colorsSize, newManager.colorsSize);
    
    std::swap(vertexCount, newManager.vertexCount);
    
    // newManager will deallocate the old data when it goes out of scope
}

float* VertexDataManager::getPositions() {
    return positions;
}

float* VertexDataManager::getNormals() {
    return normals;
}

float* VertexDataManager::getTextureCoords() {
    return textureCoords;
}

float* VertexDataManager::getColors() {
    return colors;
}

size_t VertexDataManager::getVertexCount() const {
    return vertexCount;
}

size_t VertexDataManager::getMemoryUsage() const {
    return positionsSize + normalsSize + textureCoordsSize + colorsSize;
}

const MemoryConfig& VertexDataManager::getConfig() const {
    return config;
}

void VertexDataManager::setConfig(const MemoryConfig& config) {
    this->config = config;
}

void VertexDataManager::deallocate() {
    // Deallocate arrays
    if (positions) {
        MemoryManager::getInstance().deallocateAligned(
            positions, positionsSize, "vertex_positions"
        );
        positions = nullptr;
        positionsSize = 0;
    }
    
    if (normals) {
        MemoryManager::getInstance().deallocateAligned(
            normals, normalsSize, "vertex_normals"
        );
        normals = nullptr;
        normalsSize = 0;
        hasNormals = false;
    }
    
    if (textureCoords) {
        MemoryManager::getInstance().deallocateAligned(
            textureCoords, textureCoordsSize, "vertex_texcoords"
        );
        textureCoords = nullptr;
        textureCoordsSize = 0;
        hasTextureCoords = false;
    }
    
    if (colors) {
        MemoryManager::getInstance().deallocateAligned(
            colors, colorsSize, "vertex_colors"
        );
        colors = nullptr;
        colorsSize = 0;
        hasColors = false;
    }
    
    vertexCount = 0;
}

} // namespace aeonterracpp