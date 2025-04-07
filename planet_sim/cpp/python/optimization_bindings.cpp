#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "aeonterracpp/optimization/spatial_acceleration.h"
#include "aeonterracpp/optimization/parallel_processing.h"
#include "aeonterracpp/optimization/memory_optimization.h"
#include "aeonterracpp/optimization/configuration.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_optimization_bindings(py::module& m) {
    // Create optimization submodule
    py::module optimization = m.def_submodule("optimization", "Performance optimization components");
    
    //--------------------------------------------------------------------------
    // Spatial Acceleration bindings
    //--------------------------------------------------------------------------
    py::enum_<SpatialAccelerationType>(optimization, "SpatialAccelerationType")
        .value("None", SpatialAccelerationType::None)
        .value("Grid", SpatialAccelerationType::Grid)
        .value("Octree", SpatialAccelerationType::Octree)
        .value("KdTree", SpatialAccelerationType::KdTree)
        .value("BVH", SpatialAccelerationType::BVH)
        .export_values();
    
    py::class_<SpatialAccelerationParams>(optimization, "SpatialAccelerationParams")
        .def(py::init<>())
        .def_readwrite("type", &SpatialAccelerationParams::type)
        .def_readwrite("max_depth", &SpatialAccelerationParams::maxDepth)
        .def_readwrite("max_elements_per_cell", &SpatialAccelerationParams::maxElementsPerCell)
        .def_readwrite("grid_resolution", &SpatialAccelerationParams::gridResolution)
        .def_readwrite("parallel_construction", &SpatialAccelerationParams::parallelConstruction)
        .def_readwrite("enable_caching", &SpatialAccelerationParams::enableCaching)
        .def_readwrite("num_threads", &SpatialAccelerationParams::numThreads);
    
    py::class_<SpatialAcceleration, std::shared_ptr<SpatialAcceleration>>(optimization, "SpatialAcceleration")
        .def("build", &SpatialAcceleration::build)
        .def("update", &SpatialAcceleration::update)
        .def("find_nearest_vertex", &SpatialAcceleration::findNearestVertex)
        .def("find_vertices_in_radius", &SpatialAcceleration::findVerticesInRadius)
        .def("find_k_nearest_vertices", &SpatialAcceleration::findKNearestVertices)
        .def("get_type", &SpatialAcceleration::getType)
        .def("get_params", &SpatialAcceleration::getParams)
        .def("get_memory_usage", &SpatialAcceleration::getMemoryUsage)
        .def("get_average_query_time", &SpatialAcceleration::getAverageQueryTime)
        .def("clear_query_time_stats", &SpatialAcceleration::clearQueryTimeStats)
        .def_static("create", &SpatialAcceleration::create,
             py::arg("type"), py::arg("planet"), py::arg("params") = SpatialAccelerationParams());
    
    //--------------------------------------------------------------------------
    // Parallel Processing bindings
    //--------------------------------------------------------------------------
    py::class_<ParallelConfig>(optimization, "ParallelConfig")
        .def(py::init<>())
        .def_readwrite("num_threads", &ParallelConfig::numThreads)
        .def_readwrite("enable_dynamic_load", &ParallelConfig::enableDynamicLoad)
        .def_readwrite("chunk_size", &ParallelConfig::chunkSize)
        .def_readwrite("enable_task_steal", &ParallelConfig::enableTaskSteal)
        .def_readwrite("enable_vectorization", &ParallelConfig::enableVectorization)
        .def_readwrite("track_performance", &ParallelConfig::trackPerformance);
    
    py::class_<ParallelMetrics>(optimization, "ParallelMetrics")
        .def(py::init<>())
        .def_readwrite("total_execution_time", &ParallelMetrics::totalExecutionTime)
        .def_readwrite("setup_time", &ParallelMetrics::setupTime)
        .def_readwrite("processing_time", &ParallelMetrics::processingTime)
        .def_readwrite("sync_time", &ParallelMetrics::syncTime)
        .def_readwrite("num_chunks", &ParallelMetrics::numChunks)
        .def_readwrite("num_steals", &ParallelMetrics::numSteals)
        .def_readwrite("max_queue_size", &ParallelMetrics::maxQueueSize)
        .def_readwrite("thread_times", &ParallelMetrics::threadTimes)
        .def("reset", &ParallelMetrics::reset)
        .def("merge", &ParallelMetrics::merge);
    
    py::class_<ThreadPool, std::shared_ptr<ThreadPool>>(optimization, "ThreadPool")
        .def(py::init<int>(), py::arg("num_threads") = 0)
        .def("get_num_threads", &ThreadPool::getNumThreads)
        .def("wait_for_all", &ThreadPool::waitForAll)
        .def("pause", &ThreadPool::pause)
        .def("resume", &ThreadPool::resume)
        .def("is_paused", &ThreadPool::isPaused)
        .def("get_metrics", &ThreadPool::getMetrics)
        .def("reset_metrics", &ThreadPool::resetMetrics);
    
    py::class_<ParallelFor>(optimization, "ParallelFor")
        .def(py::init<const ParallelConfig&>(), py::arg("config") = ParallelConfig())
        .def("set_config", &ParallelFor::setConfig)
        .def("get_config", &ParallelFor::getConfig)
        .def("get_metrics", &ParallelFor::getMetrics)
        .def("reset_metrics", &ParallelFor::resetMetrics);
    
    py::class_<VectorizedOps>(optimization, "VectorizedOps")
        .def_static("is_available", &VectorizedOps::isAvailable)
        .def_static("get_instruction_set", &VectorizedOps::getInstructionSet);
    
    //--------------------------------------------------------------------------
    // Memory Optimization bindings
    //--------------------------------------------------------------------------
    py::enum_<MemoryStrategy>(optimization, "MemoryStrategy")
        .value("Default", MemoryStrategy::Default)
        .value("PoolAllocator", MemoryStrategy::PoolAllocator)
        .value("ArenaAllocator", MemoryStrategy::ArenaAllocator)
        .value("ChunkAllocator", MemoryStrategy::ChunkAllocator)
        .value("VectorizedAllocator", MemoryStrategy::VectorizedAllocator)
        .export_values();
    
    py::class_<MemoryConfig>(optimization, "MemoryConfig")
        .def(py::init<>())
        .def_readwrite("strategy", &MemoryConfig::strategy)
        .def_readwrite("pool_size", &MemoryConfig::poolSize)
        .def_readwrite("chunk_size", &MemoryConfig::chunkSize)
        .def_readwrite("cache_line_size", &MemoryConfig::cacheLineSize)
        .def_readwrite("enable_compression", &MemoryConfig::enableCompression)
        .def_readwrite("track_usage", &MemoryConfig::trackUsage)
        .def_readwrite("growth_factor", &MemoryConfig::growthFactor)
        .def_readwrite("initial_reservation", &MemoryConfig::initialReservation)
        .def_readwrite("use_memory_mapping", &MemoryConfig::useMemoryMapping);
    
    py::class_<MemoryStats>(optimization, "MemoryStats")
        .def(py::init<>())
        .def_readwrite("current_usage", &MemoryStats::currentUsage)
        .def_readwrite("peak_usage", &MemoryStats::peakUsage)
        .def_readwrite("total_allocated", &MemoryStats::totalAllocated)
        .def_readwrite("total_deallocated", &MemoryStats::totalDeallocated)
        .def_readwrite("allocation_count", &MemoryStats::allocationCount)
        .def_readwrite("deallocation_count", &MemoryStats::deallocationCount)
        .def_readwrite("largest_allocation", &MemoryStats::largestAllocation)
        .def_readwrite("category_usage", &MemoryStats::categoryUsage)
        .def_static("format_size", &MemoryStats::formatSize)
        .def("reset", &MemoryStats::reset);
    
    py::class_<MemoryManager>(optimization, "MemoryManager", py::dynamic_attr())
        .def_static("get_instance", &MemoryManager::getInstance, py::return_value_policy::reference)
        .def("configure", &MemoryManager::configure)
        .def("get_config", &MemoryManager::getConfig)
        .def("get_stats", &MemoryManager::getStats)
        .def("reset_stats", &MemoryManager::resetStats)
        .def("set_memory_limit", &MemoryManager::setMemoryLimit)
        .def("get_memory_limit", &MemoryManager::getMemoryLimit)
        .def("is_near_limit", &MemoryManager::isNearLimit, py::arg("threshold_percent") = 90.0f)
        .def("register_cleanup_callback", &MemoryManager::registerCleanupCallback);
    
    py::class_<VertexDataManager>(optimization, "VertexDataManager")
        .def(py::init<const MemoryConfig&>(), py::arg("config") = MemoryConfig())
        .def("allocate", &VertexDataManager::allocate,
             py::arg("vertex_count"), py::arg("with_normals") = true,
             py::arg("with_tex_coords") = false, py::arg("with_colors") = false)
        .def("resize", &VertexDataManager::resize)
        .def("get_vertex_count", &VertexDataManager::getVertexCount)
        .def("get_memory_usage", &VertexDataManager::getMemoryUsage)
        .def("get_config", &VertexDataManager::getConfig)
        .def("set_config", &VertexDataManager::setConfig);
    
    //--------------------------------------------------------------------------
    // Configuration bindings
    //--------------------------------------------------------------------------
    py::enum_<ParamType>(optimization, "ParamType")
        .value("Int", ParamType::Int)
        .value("Float", ParamType::Float)
        .value("Double", ParamType::Double)
        .value("Bool", ParamType::Bool)
        .value("String", ParamType::String)
        .value("Enum", ParamType::Enum)
        .value("Path", ParamType::Path)
        .value("Color", ParamType::Color)
        .value("Vector2", ParamType::Vector2)
        .value("Vector3", ParamType::Vector3)
        .value("Range", ParamType::Range)
        .value("Array", ParamType::Array)
        .value("Object", ParamType::Object)
        .export_values();
    
    py::class_<ParamMetadata>(optimization, "ParamMetadata")
        .def(py::init<>())
        .def_readwrite("name", &ParamMetadata::name)
        .def_readwrite("description", &ParamMetadata::description)
        .def_readwrite("type", &ParamMetadata::type)
        .def_readwrite("required", &ParamMetadata::required)
        .def_readwrite("advanced", &ParamMetadata::advanced)
        .def_readwrite("category", &ParamMetadata::category)
        .def_readwrite("units", &ParamMetadata::units)
        .def_readwrite("min_value", &ParamMetadata::minValue)
        .def_readwrite("max_value", &ParamMetadata::maxValue)
        .def_readwrite("step_size", &ParamMetadata::stepSize)
        .def_readwrite("default_value", &ParamMetadata::defaultValue)
        .def_readwrite("enum_options", &ParamMetadata::enumOptions)
        .def_readwrite("file_extension", &ParamMetadata::fileExtension)
        .def_readwrite("regex_pattern", &ParamMetadata::regexPattern)
        .def_readwrite("hidden", &ParamMetadata::hidden)
        .def_readwrite("ui_widget", &ParamMetadata::uiWidget)
        .def_readwrite("ui_order", &ParamMetadata::uiOrder)
        .def_readwrite("ui_hint", &ParamMetadata::uiHint);
    
    py::class_<ParamValue>(optimization, "ParamValue")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init<float>())
        .def(py::init<double>())
        .def(py::init<bool>())
        .def(py::init<const std::string&>())
        .def("get_type", &ParamValue::getType)
        .def("to_string", &ParamValue::toString)
        .def("from_string", &ParamValue::fromString);
    
    py::class_<ConfigSection>(optimization, "ConfigSection")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("name"), py::arg("description") = "")
        .def("get_name", &ConfigSection::getName)
        .def("get_description", &ConfigSection::getDescription)
        .def("add_param", &ConfigSection::addParam,
             py::arg("key"), py::arg("value"), py::arg("metadata") = ParamMetadata())
        .def("has_param", &ConfigSection::hasParam)
        .def("get_param", static_cast<const ParamValue& (ConfigSection::*)(const std::string&) const>(&ConfigSection::getParam))
        .def("get_metadata", &ConfigSection::getMetadata)
        .def("set_param", &ConfigSection::setParam)
        .def("get_param_keys", &ConfigSection::getParamKeys)
        .def("remove_param", &ConfigSection::removeParam);
    
    py::class_<ConfigManager>(optimization, "ConfigManager", py::dynamic_attr())
        .def_static("get_instance", &ConfigManager::getInstance, py::return_value_policy::reference)
        .def("create_section", &ConfigManager::createSection,
             py::arg("name"), py::arg("description") = "")
        .def("get_section", static_cast<ConfigSection& (ConfigManager::*)(const std::string&)>(&ConfigManager::getSection))
        .def("has_section", &ConfigManager::hasSection)
        .def("get_section_names", &ConfigManager::getSectionNames)
        .def("remove_section", &ConfigManager::removeSection)
        .def("load_from_file", &ConfigManager::loadFromFile)
        .def("save_to_file", &ConfigManager::saveToFile)
        .def("validate", &ConfigManager::validate)
        .def("clear", &ConfigManager::clear)
        .def("register_processor", &ConfigManager::registerProcessor)
        .def("initialize_processors", &ConfigManager::initializeProcessors);
}

} // namespace aeonterracpp