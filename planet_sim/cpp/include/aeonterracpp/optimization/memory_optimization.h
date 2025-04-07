#pragma once

#include <vector>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>

namespace aeonterracpp {

/**
 * @brief Memory allocation strategy
 */
enum class MemoryStrategy {
    Default,            // Standard allocator
    PoolAllocator,      // Pool-based allocation for same-sized objects
    ArenaAllocator,     // Arena allocation for variable-sized objects
    ChunkAllocator,     // Chunk-based allocation for collections
    VectorizedAllocator // SIMD-aligned allocation
};

/**
 * @brief Memory optimization configuration
 */
struct MemoryConfig {
    MemoryStrategy strategy = MemoryStrategy::Default;
    size_t poolSize = 1024 * 1024;     // Default pool size (1MB)
    size_t chunkSize = 4096;           // Default chunk size (4KB)
    size_t cacheLineSize = 64;         // Default cache line size (64B)
    bool enableCompression = false;    // Enable data compression
    bool trackUsage = true;            // Track memory usage
    float growthFactor = 1.5f;         // Growth factor for dynamic containers
    size_t initialReservation = 4096;  // Initial reservation for containers
    bool useMemoryMapping = false;     // Use memory-mapped files for large datasets
};

/**
 * @brief Memory usage statistics
 */
struct MemoryStats {
    size_t currentUsage = 0;            // Current memory usage in bytes
    size_t peakUsage = 0;               // Peak memory usage in bytes
    size_t totalAllocated = 0;          // Total allocated memory
    size_t totalDeallocated = 0;        // Total deallocated memory
    size_t allocationCount = 0;         // Number of allocations
    size_t deallocationCount = 0;       // Number of deallocations
    size_t largestAllocation = 0;       // Largest single allocation
    std::unordered_map<std::string, size_t> categoryUsage; // Usage by category
    
    /**
     * @brief Get human-readable memory size
     * 
     * @param bytes Size in bytes
     * @return std::string Human-readable size (KB, MB, GB)
     */
    static std::string formatSize(size_t bytes);
    
    /**
     * @brief Reset all statistics
     */
    void reset();
};

/**
 * @brief Memory manager for optimized allocations
 * 
 * This class manages memory allocations using different strategies
 * to optimize performance and reduce memory fragmentation.
 */
class MemoryManager {
public:
    /**
     * @brief Get singleton instance
     * 
     * @return MemoryManager& Singleton instance
     */
    static MemoryManager& getInstance();
    
    /**
     * @brief Configure memory manager
     * 
     * @param config Memory configuration
     */
    void configure(const MemoryConfig& config);
    
    /**
     * @brief Get current configuration
     * 
     * @return const MemoryConfig& Current configuration
     */
    const MemoryConfig& getConfig() const;
    
    /**
     * @brief Allocate memory
     * 
     * @param size Size in bytes
     * @param category Optional category for tracking
     * @return void* Allocated memory
     */
    void* allocate(size_t size, const std::string& category = "default");
    
    /**
     * @brief Deallocate memory
     * 
     * @param ptr Pointer to memory
     * @param size Size in bytes
     * @param category Optional category for tracking
     */
    void deallocate(void* ptr, size_t size, const std::string& category = "default");
    
    /**
     * @brief Allocate aligned memory
     * 
     * @param size Size in bytes
     * @param alignment Alignment in bytes
     * @param category Optional category for tracking
     * @return void* Allocated memory
     */
    void* allocateAligned(size_t size, size_t alignment, const std::string& category = "default");
    
    /**
     * @brief Deallocate aligned memory
     * 
     * @param ptr Pointer to memory
     * @param size Size in bytes
     * @param category Optional category for tracking
     */
    void deallocateAligned(void* ptr, size_t size, const std::string& category = "default");
    
    /**
     * @brief Get memory statistics
     * 
     * @return const MemoryStats& Memory statistics
     */
    const MemoryStats& getStats() const;
    
    /**
     * @brief Reset memory statistics
     */
    void resetStats();
    
    /**
     * @brief Set memory limit
     * 
     * @param limit Maximum memory usage in bytes (0 = no limit)
     */
    void setMemoryLimit(size_t limit);
    
    /**
     * @brief Get memory limit
     * 
     * @return size_t Memory limit in bytes
     */
    size_t getMemoryLimit() const;
    
    /**
     * @brief Check if memory usage is near limit
     * 
     * @param thresholdPercent Threshold percentage (0-100)
     * @return true Memory usage is above threshold
     * @return false Memory usage is below threshold
     */
    bool isNearLimit(float thresholdPercent = 90.0f) const;
    
    /**
     * @brief Register cleanup callback
     * 
     * @param callback Function to call when memory limit is reached
     */
    void registerCleanupCallback(std::function<void()> callback);
    
private:
    // Private constructor for singleton
    MemoryManager();
    // Delete copy/move constructors and assignment operators
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;
    
    // Configuration
    MemoryConfig config;
    
    // Statistics
    MemoryStats stats;
    
    // Memory limit
    size_t memoryLimit = 0;
    
    // Cleanup callbacks
    std::vector<std::function<void()>> cleanupCallbacks;
    
    // Mutex for thread safety
    mutable std::mutex mutex;
    
    // Pool allocator implementation
    class PoolAllocator {
    public:
        /**
         * @brief Construct a new Pool Allocator
         * 
         * @param blockSize Size of each block
         * @param poolSize Total pool size in bytes
         */
        PoolAllocator(size_t blockSize, size_t poolSize);
        
        /**
         * @brief Destroy the Pool Allocator
         */
        ~PoolAllocator();
        
        /**
         * @brief Allocate memory
         * 
         * @return void* Allocated memory
         */
        void* allocate();
        
        /**
         * @brief Deallocate memory
         * 
         * @param ptr Pointer to memory
         */
        void deallocate(void* ptr);
        
        /**
         * @brief Get available blocks
         * 
         * @return size_t Number of available blocks
         */
        size_t available() const;
        
        /**
         * @brief Get total blocks
         * 
         * @return size_t Total number of blocks
         */
        size_t total() const;
        
    private:
        size_t blockSize;
        size_t poolSize;
        char* memory;
        void* freeList;
        size_t numBlocks;
        size_t numAllocated;
        std::mutex mutex;
    };
    
    // Pool allocators for common sizes
    std::unordered_map<size_t, std::unique_ptr<PoolAllocator>> pools;
    
    // Get or create pool allocator for size
    PoolAllocator* getPool(size_t size);
};

/**
 * @brief Memory optimized container for vertex data
 * 
 * This container optimizes memory usage for vertex arrays
 * by using Structure of Arrays (SoA) layout instead of
 * Array of Structures (AoS) for better cache performance.
 */
class VertexDataManager {
public:
    /**
     * @brief Construct a new Vertex Data Manager
     * 
     * @param config Memory configuration
     */
    VertexDataManager(const MemoryConfig& config = MemoryConfig());
    
    /**
     * @brief Allocate vertex data
     * 
     * @param vertexCount Number of vertices
     * @param withNormals Include normals
     * @param withTexCoords Include texture coordinates
     * @param withColors Include colors
     */
    void allocate(size_t vertexCount, 
                  bool withNormals = true, 
                  bool withTexCoords = false, 
                  bool withColors = false);
    
    /**
     * @brief Resize vertex data
     * 
     * @param newVertexCount New vertex count
     */
    void resize(size_t newVertexCount);
    
    /**
     * @brief Get positions data
     * 
     * @return float* Positions data (x,y,z interleaved)
     */
    float* getPositions();
    
    /**
     * @brief Get normals data
     * 
     * @return float* Normals data (nx,ny,nz interleaved)
     */
    float* getNormals();
    
    /**
     * @brief Get texture coordinates data
     * 
     * @return float* Texture coordinates data (u,v interleaved)
     */
    float* getTextureCoords();
    
    /**
     * @brief Get colors data
     * 
     * @return float* Colors data (r,g,b,a interleaved)
     */
    float* getColors();
    
    /**
     * @brief Get vertex count
     * 
     * @return size_t Number of vertices
     */
    size_t getVertexCount() const;
    
    /**
     * @brief Get memory usage in bytes
     * 
     * @return size_t Memory usage
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Get memory configuration
     * 
     * @return const MemoryConfig& Memory configuration
     */
    const MemoryConfig& getConfig() const;
    
    /**
     * @brief Set memory configuration
     * 
     * @param config New configuration
     */
    void setConfig(const MemoryConfig& config);
    
private:
    // Memory configuration
    MemoryConfig config;
    
    // Vertex count
    size_t vertexCount = 0;
    
    // Data arrays
    float* positions = nullptr;
    float* normals = nullptr;
    float* textureCoords = nullptr;
    float* colors = nullptr;
    
    // Feature flags
    bool hasNormals = false;
    bool hasTextureCoords = false;
    bool hasColors = false;
    
    // Memory allocation tracking
    size_t positionsSize = 0;
    size_t normalsSize = 0;
    size_t textureCoordsSize = 0;
    size_t colorsSize = 0;
    
    // Deallocate memory
    void deallocate();
};

/**
 * @brief Memory-mapped large array for simulation data
 * 
 * This container uses memory mapping for large arrays
 * to reduce memory pressure for huge datasets.
 * 
 * @tparam T Element type
 */
template<typename T>
class MappedArray {
public:
    /**
     * @brief Construct a new Mapped Array
     * 
     * @param size Initial size
     * @param tempFilePath Path for temp file (empty = auto-generate)
     */
    MappedArray(size_t size = 0, const std::string& tempFilePath = "");
    
    /**
     * @brief Destroy the Mapped Array
     */
    ~MappedArray();
    
    /**
     * @brief Resize array
     * 
     * @param newSize New size
     */
    void resize(size_t newSize);
    
    /**
     * @brief Access element
     * 
     * @param index Element index
     * @return T& Reference to element
     */
    T& operator[](size_t index);
    
    /**
     * @brief Access element (const)
     * 
     * @param index Element index
     * @return const T& Reference to element
     */
    const T& operator[](size_t index) const;
    
    /**
     * @brief Get data pointer
     * 
     * @return T* Pointer to data
     */
    T* data();
    
    /**
     * @brief Get data pointer (const)
     * 
     * @return const T* Pointer to data
     */
    const T* data() const;
    
    /**
     * @brief Get size
     * 
     * @return size_t Number of elements
     */
    size_t size() const;
    
    /**
     * @brief Get memory usage in bytes
     * 
     * @return size_t Memory usage
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Flush changes to disk
     */
    void flush();
    
private:
    // Size
    size_t size_ = 0;
    
    // File path
    std::string filePath;
    
    // File descriptor
    int fileDescriptor = -1;
    
    // Mapped memory
    T* mappedData = nullptr;
    
    // Map size
    size_t mappedSize = 0;
    
    // Close and reopen with new size
    void remapFile(size_t newSize);
};

/**
 * @brief Compressed storage for sparse data
 * 
 * Optimizes memory usage for sparse arrays by
 * storing only non-zero values with indices.
 * 
 * @tparam T Element type
 */
template<typename T>
class SparseArray {
public:
    /**
     * @brief Construct a new Sparse Array
     * 
     * @param size Logical size of the array
     * @param defaultValue Default value for unset elements
     */
    SparseArray(size_t size = 0, const T& defaultValue = T());
    
    /**
     * @brief Set element value
     * 
     * @param index Element index
     * @param value Element value
     */
    void set(size_t index, const T& value);
    
    /**
     * @brief Get element value
     * 
     * @param index Element index
     * @return const T& Element value
     */
    const T& get(size_t index) const;
    
    /**
     * @brief Access element (may create if not exists)
     * 
     * @param index Element index
     * @return T& Reference to element
     */
    T& operator[](size_t index);
    
    /**
     * @brief Get logical size
     * 
     * @return size_t Logical size
     */
    size_t size() const;
    
    /**
     * @brief Get number of stored elements
     * 
     * @return size_t Number of stored elements
     */
    size_t storedElements() const;
    
    /**
     * @brief Resize array
     * 
     * @param newSize New logical size
     */
    void resize(size_t newSize);
    
    /**
     * @brief Clear all elements
     */
    void clear();
    
    /**
     * @brief Get memory usage in bytes
     * 
     * @return size_t Memory usage
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Get compression ratio
     * 
     * @return float Compression ratio (stored/total)
     */
    float getCompressionRatio() const;
    
    /**
     * @brief Get default value
     * 
     * @return const T& Default value
     */
    const T& getDefaultValue() const;
    
    /**
     * @brief Set default value
     * 
     * @param value New default value
     */
    void setDefaultValue(const T& value);
    
private:
    // Logical size
    size_t size_ = 0;
    
    // Default value
    T defaultValue;
    
    // Stored values
    std::unordered_map<size_t, T> values;
};

} // namespace aeonterracpp