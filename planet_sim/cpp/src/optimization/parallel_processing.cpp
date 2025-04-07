#include "aeonterracpp/optimization/parallel_processing.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>

namespace aeonterracpp {

//------------------------------------------------------------------------------
// ThreadPool Implementation
//------------------------------------------------------------------------------

ThreadPool::ThreadPool(int numThreads) 
    : stop(false), paused(false) {
    
    // Determine number of threads
    int threadCount = numThreads > 0 ? 
                      numThreads : 
                      std::thread::hardware_concurrency();
    
    // Use at least one thread
    if (threadCount < 1) threadCount = 1;
    
    std::cout << "Creating thread pool with " << threadCount << " threads" << std::endl;
    
    // Create worker threads
    for (int i = 0; i < threadCount; ++i) {
        workers.emplace_back(&ThreadPool::workerFunction, this, i);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    
    // Wake up all threads to ensure they see the stop flag
    condition.notify_all();
    
    // Wait for all threads to finish
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    std::cout << "Thread pool destroyed" << std::endl;
}

int ThreadPool::getNumThreads() const {
    return static_cast<int>(workers.size());
}

void ThreadPool::waitForAll() {
    std::unique_lock<std::mutex> lock(queueMutex);
    
    // Wait until the task queue is empty
    condition.wait(lock, [this]() {
        return tasks.empty();
    });
}

void ThreadPool::pause() {
    std::unique_lock<std::mutex> lock(queueMutex);
    paused = true;
}

void ThreadPool::resume() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        paused = false;
    }
    
    // Wake up all threads
    condition.notify_all();
}

bool ThreadPool::isPaused() const {
    return paused;
}

const ParallelMetrics& ThreadPool::getMetrics() const {
    return metrics;
}

void ThreadPool::resetMetrics() {
    std::unique_lock<std::mutex> lock(metricsMutex);
    metrics.reset();
}

void ThreadPool::workerFunction(int threadId) {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            
            // Wait for a task or stop/pause signal
            condition.wait(lock, [this]() {
                return stop || (!tasks.empty() && !paused);
            });
            
            // If pool is being stopped and there are no more tasks, exit
            if (stop && tasks.empty()) {
                return;
            }
            
            // If paused, continue waiting
            if (paused) {
                continue;
            }
            
            // Get the next task
            if (!tasks.empty()) {
                task = std::move(tasks.front());
                tasks.pop();
            }
        }
        
        // Execute the task
        if (task) {
            auto start = std::chrono::high_resolution_clock::now();
            
            task();
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            
            // Update metrics
            std::unique_lock<std::mutex> lock(metricsMutex);
            metrics.processingTime += duration.count();
            
            if (metrics.threadTimes.size() <= static_cast<size_t>(threadId)) {
                metrics.threadTimes.resize(threadId + 1, 0.0);
            }
            
            metrics.threadTimes[threadId] += duration.count();
        }
    }
}

//------------------------------------------------------------------------------
// ParallelFor Implementation
//------------------------------------------------------------------------------

ParallelFor::ParallelFor(const ParallelConfig& config)
    : config(config) {
    
    // Create thread pool
    threadPool = std::make_shared<ThreadPool>(config.numThreads);
}

void ParallelFor::setConfig(const ParallelConfig& config) {
    this->config = config;
    
    // Re-create thread pool if number of threads changed
    if (threadPool->getNumThreads() != config.numThreads) {
        threadPool = std::make_shared<ThreadPool>(config.numThreads);
    }
}

const ParallelConfig& ParallelFor::getConfig() const {
    return config;
}

const ParallelMetrics& ParallelFor::getMetrics() const {
    return metrics;
}

void ParallelFor::resetMetrics() {
    std::unique_lock<std::mutex> lock(metricsMutex);
    metrics.reset();
    threadPool->resetMetrics();
}

size_t ParallelFor::determineChunkSize(size_t rangeSize) const {
    if (config.chunkSize > 0) {
        return config.chunkSize;
    }
    
    // Calculate a reasonable chunk size based on range size and thread count
    int numThreads = threadPool->getNumThreads();
    
    // Aim for at least 4 chunks per thread for good load balancing
    size_t targetChunks = numThreads * 4;
    
    // But don't make chunks too small
    size_t chunkSize = std::max(rangeSize / targetChunks, size_t(16));
    
    // But don't make chunks too large either
    chunkSize = std::min(chunkSize, size_t(1024));
    
    return chunkSize;
}

// Explicit template instantiations for types used in the project
// This is just a placeholder for actual implementation
template void ParallelFor::execute<std::function<void()>>(size_t, size_t, std::function<void()>&&);
template void ParallelFor::executeWithIndex<std::function<void(size_t)>>(size_t, size_t, std::function<void(size_t)>&&);
template void ParallelFor::executeForEach<std::vector<float>, std::function<void(float&)>>(std::vector<float>&, std::function<void(float&)>&&);

//------------------------------------------------------------------------------
// VectorizedOps Implementation
//------------------------------------------------------------------------------

void VectorizedOps::add(const float* a, const float* b, float* result, size_t size) {
    // Simple scalar implementation for now
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorizedOps::subtract(const float* a, const float* b, float* result, size_t size) {
    // Simple scalar implementation for now
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorizedOps::multiply(const float* a, const float* b, float* result, size_t size) {
    // Simple scalar implementation for now
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorizedOps::divide(const float* a, const float* b, float* result, size_t size) {
    // Simple scalar implementation for now
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] / b[i];
    }
}

void VectorizedOps::scale(const float* a, float scalar, float* result, size_t size) {
    // Simple scalar implementation for now
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

float VectorizedOps::dot(const float* a, const float* b, size_t size) {
    // Simple scalar implementation for now
    float result = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float VectorizedOps::norm(const float* a, size_t size) {
    // Simple scalar implementation for now
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

void VectorizedOps::normalize(const float* a, float* result, size_t size) {
    // Simple scalar implementation for now
    float norm = VectorizedOps::norm(a, size);
    if (norm > 0.0f) {
        float invNorm = 1.0f / norm;
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * invNorm;
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            result[i] = 0.0f;
        }
    }
}

void VectorizedOps::lerp(const float* a, const float* b, float t, float* result, size_t size) {
    // Simple scalar implementation for now
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * (1.0f - t) + b[i] * t;
    }
}

bool VectorizedOps::isAvailable() {
    // Always return true for the scalar implementation
    return true;
}

const char* VectorizedOps::getInstructionSet() {
    // Return the name of the instruction set being used
    return "Scalar";
}

} // namespace aeonterracpp