#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/core/enhanced_erosion.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_erosion_bindings(py::module& m) {
    // Bind ErosionProcessType enum
    py::enum_<ErosionProcessType>(m, "ErosionProcessType")
        .value("Fluvial", ErosionProcessType::Fluvial)
        .value("Coastal", ErosionProcessType::Coastal)
        .value("Glacial", ErosionProcessType::Glacial)
        .value("Aeolian", ErosionProcessType::Aeolian)
        .value("Chemical", ErosionProcessType::Chemical)
        .value("Thermal", ErosionProcessType::Thermal)
        .value("Gravity", ErosionProcessType::Gravity)
        .value("Biotic", ErosionProcessType::Biotic)
        .value("Tectonic", ErosionProcessType::Tectonic)
        .export_values();
    
    // Bind ErosionParams struct
    py::class_<ErosionParams>(m, "ErosionParams")
        .def(py::init<>())
        .def_readwrite("global_erosion_rate", &ErosionParams::globalErosionRate)
        .def_readwrite("fluvial_erosion_strength", &ErosionParams::fluvialErosionStrength)
        .def_readwrite("coastal_erosion_strength", &ErosionParams::coastalErosionStrength)
        .def_readwrite("glacial_erosion_strength", &ErosionParams::glacialErosionStrength)
        .def_readwrite("aeolian_erosion_strength", &ErosionParams::aeolianErosionStrength)
        .def_readwrite("chemical_weathering_strength", &ErosionParams::chemicalWeatheringStrength)
        .def_readwrite("thermal_weathering_strength", &ErosionParams::thermalWeatheringStrength)
        .def_readwrite("mass_movement_strength", &ErosionParams::massMovementStrength)
        .def_readwrite("biotic_erosion_strength", &ErosionParams::bioticErosionStrength)
        .def_readwrite("river_flow_threshold", &ErosionParams::riverFlowThreshold)
        .def_readwrite("coastal_wave_energy", &ErosionParams::coastalWaveEnergy)
        .def_readwrite("glacial_temperature_threshold", &ErosionParams::glacialTemperatureThreshold)
        .def_readwrite("wind_erosion_threshold", &ErosionParams::windErosionThreshold)
        .def_readwrite("critical_slope", &ErosionParams::criticalSlope)
        .def_readwrite("sediment_capacity_factor", &ErosionParams::sedimentCapacityFactor)
        .def_readwrite("time_scale_factor", &ErosionParams::timeScaleFactor);
    
    // Bind EnhancedErosion class
    py::class_<EnhancedErosion>(m, "EnhancedErosion")
        .def(py::init<Planet&, const ClimateSystem&, const BiomeSystem&>(),
             py::arg("planet"), py::arg("climate"), py::arg("biomes"))
        .def("initialize", &EnhancedErosion::initialize,
             py::arg("params") = ErosionParams())
        .def("set_parameters", &EnhancedErosion::setParameters)
        .def("get_parameters", &EnhancedErosion::getParameters)
        .def("simulate_step", &EnhancedErosion::simulateStep,
             py::arg("time_step"))
        .def("get_erosion_rate", &EnhancedErosion::getErosionRate,
             py::arg("position"))
        .def("get_process_erosion_rate", &EnhancedErosion::getProcessErosionRate,
             py::arg("position"), py::arg("process_type"))
        .def("get_accumulated_erosion", &EnhancedErosion::getAccumulatedErosion,
             py::arg("position"))
        .def("get_accumulated_deposition", &EnhancedErosion::getAccumulatedDeposition,
             py::arg("position"))
        .def("simulate_fluvial_erosion", &EnhancedErosion::simulateFluvialErosion,
             py::arg("time_step"))
        .def("simulate_coastal_erosion", &EnhancedErosion::simulateCoastalErosion,
             py::arg("time_step"))
        .def("simulate_glacial_erosion", &EnhancedErosion::simulateGlacialErosion,
             py::arg("time_step"))
        .def("simulate_aeolian_erosion", &EnhancedErosion::simulateAeolianErosion,
             py::arg("time_step"))
        .def("simulate_chemical_weathering", &EnhancedErosion::simulateChemicalWeathering,
             py::arg("time_step"))
        .def("simulate_thermal_weathering", &EnhancedErosion::simulateThermalWeathering,
             py::arg("time_step"))
        .def("simulate_mass_movements", &EnhancedErosion::simulateMassMovements,
             py::arg("time_step"))
        .def("simulate_biotic_erosion", &EnhancedErosion::simulateBioticErosion,
             py::arg("time_step"))
        .def("apply_erosion_to_surface", &EnhancedErosion::applyErosionToSurface)
        .def("update_biome_effects", &EnhancedErosion::updateBiomeEffects,
             py::arg("biomes"))
        .def("update_hydrology_effects", &EnhancedErosion::updateHydrologyEffects,
             py::arg("climate"))
        .def("get_erosion_data", &EnhancedErosion::getErosionData,
             py::return_value_policy::reference_internal)
        .def("get_process_erosion_data", &EnhancedErosion::getProcessErosionData,
             py::arg("process_type"), py::return_value_policy::reference_internal)
        .def("get_sediment_data", &EnhancedErosion::getSedimentData,
             py::return_value_policy::reference_internal);
}

} // namespace aeonterracpp