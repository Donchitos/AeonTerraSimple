#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <functional>
#include <atomic>
#include <memory>

namespace aeonterracpp {

/**
 * @brief Parallel processing configuration
 */
struct ParallelConfig {
    int numThreads = 0;                // Number of threads (0 = use hardware concurrency)
    bool enableDynamicLoad = true;     // Enable dynamic load balancing
    size_t chunkSize = 256;            // Chunk size for work division
    bool enableTaskSteal = true;       // Enable work stealing between threads
    bool enableVectorization = true;   // Enable compiler vectorization hints
    bool trackPerformance = true;      // Track performance metrics
};

/**
 * @brief Performance metrics for parallel processing
 */
struct ParallelMetrics {
    double totalExecutionTime = 0.0;   // Total execution time in milliseconds
    double setupTime = 0.0;            // Time spent in setup phase
    double processingTime = 0.0;       // Time spent in processing phase
    double syncTime = 0.0;             // Time spent synchronizing threads
    size_t numChunks = 0;              // Number of chunks processed
    size_t numSteals = 0;              // Number of work stealing events
    size_t maxQueueSize = 0;           // Maximum task queue size
    std::vector<double> threadTimes;   // Per-thread execution times
    
    /**
     * @brief Reset all metrics to zero
     */
    void reset() {
        totalExecutionTime = 0.0;
        setupTime = 0.0;
        processingTime = 0.0;
        syncTime = 0.0;
        numChunks = 0;
        numSteals = 0;
        maxQueueSize = 0;
        threadTimes.clear();
    }
    
    /**
     * @brief Merge metrics from another instance
     * 
     * @param other Metrics to merge
     */
    void merge(const ParallelMetrics& other) {
        totalExecutionTime = std::max(totalExecutionTime, other.totalExecutionTime);
        setupTime += other.setupTime;
        processingTime += other.processingTime;
        syncTime += other.syncTime;
        numChunks += other.numChunks;
        numSteals += other.numSteals;
        maxQueueSize = std::max(maxQueueSize, other.maxQueueSize);
        
        // Merge thread times
        threadTimes.insert(threadTimes.end(), 
                           other.threadTimes.begin(), 
                           other.threadTimes.end());
    }
};

/**
 * @brief Thread pool for parallel processing
 * 
 * A high-performance thread pool for parallelizing simulation tasks.
 * Supports dynamic load balancing, work stealing, and performance tracking.
 */
class ThreadPool {
public:
    /**
     * @brief Construct a new Thread Pool
     * 
     * @param numThreads Number of threads (0 = use hardware concurrency)
     */
    ThreadPool(int numThreads = 0);
    
    /**
     * @brief Destroy the Thread Pool
     */
    ~ThreadPool();
    
    /**
     * @brief Get the number of threads
     * 
     * @return int Number of threads
     */
    int getNumThreads() const;
    
    /**
     * @brief Submit a task to the pool
     * 
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments for the function
     * @return std::future<typename std::result_of<F(Args...)>::type> Future for the result
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    /**
     * @brief Wait for all tasks to complete
     */
    void waitForAll();
    
    /**
     * @brief Pause all threads
     */
    void pause();
    
    /**
     * @brief Resume all threads
     */
    void resume();
    
    /**
     * @brief Check if pool is paused
     * 
     * @return true Pool is paused
     * @return false Pool is running
     */
    bool isPaused() const;
    
    /**
     * @brief Get performance metrics
     * 
     * @return const ParallelMetrics& Performance metrics
     */
    const ParallelMetrics& getMetrics() const;
    
    /**
     * @brief Reset performance metrics
     */
    void resetMetrics();
    
private:
    // Implementation details
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<bool> paused;
    
    // Performance tracking
    ParallelMetrics metrics;
    std::mutex metricsMutex;
    
    // Worker thread function
    void workerFunction(int threadId);
};

/**
 * @brief Parallel for loop
 * 
 * High-performance parallel for loop with load balancing.
 */
class ParallelFor {
public:
    /**
     * @brief Construct a new Parallel For
     * 
     * @param config Parallel configuration
     */
    ParallelFor(const ParallelConfig& config = ParallelConfig());
    
    /**
     * @brief Execute a function for each element in the range
     * 
     * @tparam F Function type
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @param f Function to execute for each index
     */
    template<typename F>
    void execute(size_t start, size_t end, F&& f);
    
    /**
     * @brief Execute a function for each element in the range with index
     * 
     * @tparam F Function type
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @param f Function to execute for each index
     */
    template<typename F>
    void executeWithIndex(size_t start, size_t end, F&& f);
    
    /**
     * @brief Execute a function for each element in a container
     * 
     * @tparam Container Container type
     * @tparam F Function type
     * @param container Container to iterate over
     * @param f Function to execute for each element
     */
    template<typename Container, typename F>
    void executeForEach(Container& container, F&& f);
    
    /**
     * @brief Get performance metrics
     * 
     * @return const ParallelMetrics& Performance metrics
     */
    const ParallelMetrics& getMetrics() const;
    
    /**
     * @brief Reset performance metrics
     */
    void resetMetrics();
    
    /**
     * @brief Set parallel configuration
     * 
     * @param config New configuration
     */
    void setConfig(const ParallelConfig& config);
    
    /**
     * @brief Get parallel configuration
     * 
     * @return const ParallelConfig& Current configuration
     */
    const ParallelConfig& getConfig() const;
    
private:
    // Thread pool
    std::shared_ptr<ThreadPool> threadPool;
    
    // Configuration
    ParallelConfig config;
    
    // Performance tracking
    ParallelMetrics metrics;
    std::mutex metricsMutex;
    
    // Internal methods for different parallel strategies
    template<typename F>
    void staticParallelFor(size_t start, size_t end, F&& f);
    
    template<typename F>
    void dynamicParallelFor(size_t start, size_t end, F&& f);
    
    // Determine optimal chunk size based on range and thread count
    size_t determineChunkSize(size_t rangeSize) const;
};

/**
 * @brief Data parallel array
 * 
 * Container that supports efficient parallel operations.
 * 
 * @tparam T Element type
 */
template<typename T>
class ParallelArray {
public:
    /**
     * @brief Construct a new Parallel Array
     * 
     * @param size Initial size
     * @param value Initial value
     */
    ParallelArray(size_t size = 0, const T& value = T());
    
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
     * @brief Get size
     * 
     * @return size_t Container size
     */
    size_t size() const;
    
    /**
     * @brief Resize container
     * 
     * @param size New size
     * @param value Value for new elements
     */
    void resize(size_t size, const T& value = T());
    
    /**
     * @brief Apply function to all elements in parallel
     * 
     * @tparam F Function type
     * @param f Function to apply
     */
    template<typename F>
    void parallelApply(F&& f);
    
    /**
     * @brief Apply function to all elements with indices in parallel
     * 
     * @tparam F Function type
     * @param f Function to apply
     */
    template<typename F>
    void parallelApplyWithIndex(F&& f);
    
    /**
     * @brief Transform to another array in parallel
     * 
     * @tparam U Output element type
     * @tparam F Function type
     * @param f Transformation function
     * @return ParallelArray<U> Transformed array
     */
    template<typename U, typename F>
    ParallelArray<U> parallelTransform(F&& f) const;
    
    /**
     * @brief Reduce array to single value
     * 
     * @tparam U Result type
     * @tparam F Function type
     * @param initialValue Initial value
     * @param f Reduction function
     * @return U Reduced value
     */
    template<typename U, typename F>
    U parallelReduce(const U& initialValue, F&& f) const;
    
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
     * @brief Set parallel configuration
     * 
     * @param config New configuration
     */
    void setParallelConfig(const ParallelConfig& config);
    
private:
    std::vector<T> data_;
    ParallelFor parallelFor;
};

/**
 * @brief SIMD-optimized vector operations
 * 
 * Provides vectorized operations for common vector types.
 */
class VectorizedOps {
public:
    /**
     * @brief Add vectors (a + b)
     * 
     * @param a First vector
     * @param b Second vector
     * @param result Output vector
     * @param size Number of elements
     */
    static void add(const float* a, const float* b, float* result, size_t size);
    
    /**
     * @brief Subtract vectors (a - b)
     * 
     * @param a First vector
     * @param b Second vector
     * @param result Output vector
     * @param size Number of elements
     */
    static void subtract(const float* a, const float* b, float* result, size_t size);
    
    /**
     * @brief Multiply vectors element-wise (a * b)
     * 
     * @param a First vector
     * @param b Second vector
     * @param result Output vector
     * @param size Number of elements
     */
    static void multiply(const float* a, const float* b, float* result, size_t size);
    
    /**
     * @brief Divide vectors element-wise (a / b)
     * 
     * @param a First vector
     * @param b Second vector
     * @param result Output vector
     * @param size Number of elements
     */
    static void divide(const float* a, const float* b, float* result, size_t size);
    
    /**
     * @brief Scale vector (a * scalar)
     * 
     * @param a Input vector
     * @param scalar Scalar value
     * @param result Output vector
     * @param size Number of elements
     */
    static void scale(const float* a, float scalar, float* result, size_t size);
    
    /**
     * @brief Calculate dot product (a·b)
     * 
     * @param a First vector
     * @param b Second vector
     * @param size Number of elements
     * @return float Dot product
     */
    static float dot(const float* a, const float* b, size_t size);
    
    /**
     * @brief Calculate vector norm (|a|)
     * 
     * @param a Input vector
     * @param size Number of elements
     * @return float Vector norm
     */
    static float norm(const float* a, size_t size);
    
    /**
     * @brief Normalize vector (a/|a|)
     * 
     * @param a Input vector
     * @param result Output vector
     * @param size Number of elements
     */
    static void normalize(const float* a, float* result, size_t size);
    
    /**
     * @brief Interpolate between vectors (a*(1-t) + b*t)
     * 
     * @param a First vector
     * @param b Second vector
     * @param t Interpolation factor
     * @param result Output vector
     * @param size Number of elements
     */
    static void lerp(const float* a, const float* b, float t, float* result, size_t size);
    
    /**
     * @brief Check if SIMD operations are available
     * 
     * @return true SIMD operations available
     * @return false SIMD operations not available
     */
    static bool isAvailable();
    
    /**
     * @brief Get SIMD instruction set being used
     * 
     * @return const char* Instruction set name
     */
    static const char* getInstructionSet();
};

} // namespace aeonterracpp