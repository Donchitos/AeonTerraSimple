#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "aeonterracpp/core/heightmap_generator.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_heightmap_bindings(py::module& m) {
    // Bind HeightmapParams struct
    py::class_<HeightmapParams>(m, "HeightmapParams")
        .def(py::init<>())
        // Resolution parameters
        .def_readwrite("width", &HeightmapParams::width)
        .def_readwrite("height", &HeightmapParams::height)
        // Basic parameters
        .def_readwrite("ocean_level", &HeightmapParams::oceanLevel)
        .def_readwrite("max_elevation", &HeightmapParams::maxElevation)
        .def_readwrite("min_elevation", &HeightmapParams::minElevation)
        .def_readwrite("continental_shelf_depth", &HeightmapParams::continentalShelfDepth)
        // Noise parameters
        .def_readwrite("noise_seed", &HeightmapParams::noiseSeed)
        .def_readwrite("noise_scale", &HeightmapParams::noiseScale)
        .def_readwrite("noise_strength", &HeightmapParams::noiseStrength)
        .def_readwrite("noise_octaves", &HeightmapParams::noiseOctaves)
        .def_readwrite("noise_persistence", &HeightmapParams::noisePersistence)
        .def_readwrite("noise_lacunarity", &HeightmapParams::noiseLacunarity)
        // Mountain parameters
        .def_readwrite("mountain_roughness", &HeightmapParams::mountainRoughness)
        .def_readwrite("plateau_flatness", &HeightmapParams::plateauFlatness)
        // Erosion parameters
        .def_readwrite("erosion_iterations", &HeightmapParams::erosionIterations)
        .def_readwrite("erosion_strength", &HeightmapParams::erosionStrength)
        .def_readwrite("river_erosion_factor", &HeightmapParams::riverErosionFactor)
        // Tectonic influence
        .def_readwrite("tectonic_influence", &HeightmapParams::tectonicInfluence)
        .def_readwrite("fault_displacement", &HeightmapParams::faultDisplacement)
        // Climate influence
        .def_readwrite("climate_influence", &HeightmapParams::climateInfluence)
        .def_readwrite("precipitation_erosion", &HeightmapParams::precipitationErosion)
        .def("__repr__", [](const HeightmapParams& params) {
            return "HeightmapParams(width=" + std::to_string(params.width) + 
                   ", height=" + std::to_string(params.height) +
                   ", max_elevation=" + std::to_string(params.maxElevation) +
                   ", min_elevation=" + std::to_string(params.minElevation) + ")";
        });

    // Bind HeightmapGenerator class
    py::class_<HeightmapGenerator>(m, "HeightmapGenerator")
        .def(py::init<const Planet&, const TectonicSimulation*>(),
             py::arg("planet"),
             py::arg("tectonics") = nullptr)
        .def("set_parameters", &HeightmapGenerator::setParameters)
        .def("get_parameters", &HeightmapGenerator::getParameters)
        .def("generate", &HeightmapGenerator::generate,
             py::arg("equirectangular") = true,
             "Generate a heightmap from the planet data")
        .def("generate_normal_map", &HeightmapGenerator::generateNormalMap,
             "Generate a normal map from the current heightmap data")
        .def("generate_slope_map", &HeightmapGenerator::generateSlopeMap,
             "Generate a slope map from the current heightmap data")
        .def("save_to_file", &HeightmapGenerator::saveToFile,
             py::arg("filename"),
             py::arg("format") = "png",
             "Save the heightmap to a file")
        .def("get_width", &HeightmapGenerator::getWidth)
        .def("get_height", &HeightmapGenerator::getHeight)
        .def("set_terrain_modifier", &HeightmapGenerator::setTerrainModifier,
             "Set a custom terrain modifier function")
        .def_property_readonly("data", [](const HeightmapGenerator& generator) {
            const auto& data = generator.getData();
            int width = generator.getWidth();
            int height = generator.getHeight();
            
            return py::array_t<float>(
                {height, width},                         // Shape (2D array, rows first)
                {width * sizeof(float), sizeof(float)},  // Strides for row-major layout
                data.data(),                             // Data pointer
                py::cast(generator)                      // Keep alive reference
            );
        }, "Get the raw heightmap data as a numpy array");
}

} // namespace aeonterracpp