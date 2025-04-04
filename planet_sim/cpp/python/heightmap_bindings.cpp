#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/heightmap/heightmap_generator.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_heightmap_bindings(py::module& m) {
    // Bind HeightmapData
    py::class_<HeightmapData>(m, "HeightmapData")
        .def(py::init<int, int, const BoundingBox&>())
        .def_readonly("width", &HeightmapData::width)
        .def_readonly("height", &HeightmapData::height)
        .def_readonly("min_value", &HeightmapData::minValue)
        .def_readonly("max_value", &HeightmapData::maxValue)
        .def_readonly("region", &HeightmapData::region)
        .def("get", &HeightmapData::get)
        .def("set", &HeightmapData::set)
        .def("normalize", &HeightmapData::normalize)
        .def("blend", &HeightmapData::blend)
        .def_property_readonly("data", [](const HeightmapData& hm) {
            // Convert heightmap data to numpy array
            auto data = py::array_t<float>({hm.height, hm.width});  // Note: height first for numpy
            auto buffer = data.request();
            float* ptr = static_cast<float*>(buffer.ptr);
            
            for (int y = 0; y < hm.height; y++) {
                for (int x = 0; x < hm.width; x++) {
                    ptr[y * hm.width + x] = hm.get(x, y);
                }
            }
            
            return data;
        });
    
    // Bind HeightmapConfig
    py::class_<HeightmapConfig>(m, "HeightmapConfig")
        .def(py::init<>())
        .def_readwrite("region", &HeightmapConfig::region)
        .def_readwrite("width", &HeightmapConfig::width)
        .def_readwrite("height", &HeightmapConfig::height)
        .def_readwrite("enhance_detail", &HeightmapConfig::enhanceDetail)
        .def_readwrite("roughness", &HeightmapConfig::roughness)
        .def_readwrite("octaves", &HeightmapConfig::octaves)
        .def_readwrite("detail_scale", &HeightmapConfig::detailScale)
        .def_readwrite("normalize_values", &HeightmapConfig::normalizeValues)
        .def_readwrite("bit_depth", &HeightmapConfig::bitDepth)
        .def_readwrite("use_tiling", &HeightmapConfig::useTiling)
        .def_readwrite("tile_size", &HeightmapConfig::tileSize)
        .def_readwrite("tile_overlap", &HeightmapConfig::tileOverlap);
    
    // Bind HeightmapGenerator
    py::class_<HeightmapGenerator>(m, "HeightmapGenerator")
        .def(py::init<int>(), py::arg("max_resolution") = 16384)
        .def("generate_heightmap", &HeightmapGenerator::generateHeightmap)
        .def("enhance_detail", &HeightmapGenerator::enhanceDetail)
        .def("export_png", &HeightmapGenerator::exportPNG,
             py::arg("heightmap"), py::arg("filename"), py::arg("normalize_values") = true)
        .def("export_tiff", &HeightmapGenerator::exportTIFF,
             py::arg("heightmap"), py::arg("filename"), py::arg("bit_depth") = 16)
        .def("export_raw", &HeightmapGenerator::exportRAW,
             py::arg("heightmap"), py::arg("filename"), py::arg("normalize_values") = true);
    
    // Bind HeightmapConfigFactory
    py::class_<HeightmapConfigFactory>(m, "HeightmapConfigFactory")
        .def_static("create_global_config", &HeightmapConfigFactory::createGlobalConfig,
                   py::arg("width") = 4096, py::arg("height") = 2048)
        .def_static("create_regional_config", &HeightmapConfigFactory::createRegionalConfig,
                   py::arg("center_lat"), py::arg("center_lon"),
                   py::arg("width"), py::arg("height"),
                   py::arg("resolution") = 16)
        .def_static("create_detailed_terrain_config", &HeightmapConfigFactory::createDetailedTerrainConfig,
                   py::arg("region"), py::arg("resolution") = 2048);
}

} // namespace aeonterracpp