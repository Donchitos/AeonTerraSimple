#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/core/biome_system.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_biome_bindings(py::module& m) {
    // Bind BiomeType enum
    py::enum_<BiomeType>(m, "BiomeType")
        .value("TropicalRainforest", BiomeType::TropicalRainforest)
        .value("TemperateRainforest", BiomeType::TemperateRainforest)
        .value("TropicalSeasonalForest", BiomeType::TropicalSeasonalForest)
        .value("TemperateDeciduousForest", BiomeType::TemperateDeciduousForest)
        .value("ConiferousForest", BiomeType::ConiferousForest)
        .value("Savanna", BiomeType::Savanna)
        .value("Grassland", BiomeType::Grassland)
        .value("Shrubland", BiomeType::Shrubland)
        .value("HotDesert", BiomeType::HotDesert)
        .value("ColdDesert", BiomeType::ColdDesert)
        .value("Tundra", BiomeType::Tundra)
        .value("IceSheet", BiomeType::IceSheet)
        .value("Swamp", BiomeType::Swamp)
        .value("Marsh", BiomeType::Marsh)
        .value("Mountain", BiomeType::Mountain)
        .value("HighMountain", BiomeType::HighMountain)
        .value("VolcanicRegion", BiomeType::VolcanicRegion)
        .value("OceanicIsland", BiomeType::OceanicIsland)
        .value("Ocean", BiomeType::Ocean)
        .value("CoastalWaters", BiomeType::CoastalWaters)
        .value("DeepOcean", BiomeType::DeepOcean)
        .value("OceanTrench", BiomeType::OceanTrench)
        .export_values();
    
    // Bind BiomeProperties
    py::class_<BiomeProperties>(m, "BiomeProperties")
        .def(py::init<const std::string&, float, float, float, float, const std::string&>(),
             py::arg("name") = "Unknown",
             py::arg("erosion_rate") = 0.5f,
             py::arg("veg_density") = 0.5f,
             py::arg("soil_depth") = 1.0f,
             py::arg("albedo") = 0.3f,
             py::arg("description") = "")
        .def_readwrite("name", &BiomeProperties::name)
        .def_readwrite("base_erosion_rate", &BiomeProperties::baseErosionRate)
        .def_readwrite("vegetation_density", &BiomeProperties::vegetationDensity)
        .def_readwrite("soil_depth", &BiomeProperties::soilDepth)
        .def_readwrite("albedo", &BiomeProperties::albedo)
        .def_readwrite("description", &BiomeProperties::description);
    
    // Bind BiomeSystem
    py::class_<BiomeSystem>(m, "BiomeSystem")
        .def(py::init<Planet&, const ClimateSystem&>())
        .def("initialize", &BiomeSystem::initialize)
        .def("update", &BiomeSystem::update, py::arg("time_step"))
        .def("get_biome_type", &BiomeSystem::getBiomeType, py::arg("position"))
        .def("get_biome_properties", &BiomeSystem::getBiomeProperties, py::arg("position"))
        .def("get_vegetation_density", &BiomeSystem::getVegetationDensity, py::arg("position"))
        .def("classify_biomes", &BiomeSystem::classifyBiomes)
        .def("calculate_soil_properties", &BiomeSystem::calculateSoilProperties)
        .def("calculate_vegetation_cover", &BiomeSystem::calculateVegetationCover)
        .def("get_biome_data", &BiomeSystem::getBiomeData, py::return_value_policy::reference_internal)
        .def("get_vegetation_data", &BiomeSystem::getVegetationData, py::return_value_policy::reference_internal)
        .def("get_soil_depth_data", &BiomeSystem::getSoilDepthData, py::return_value_policy::reference_internal)
        .def("get_biome_name", &BiomeSystem::getBiomeName, py::arg("type"))
        .def("get_biome_color", &BiomeSystem::getBiomeColor, py::arg("type"));
}

} // namespace aeonterracpp