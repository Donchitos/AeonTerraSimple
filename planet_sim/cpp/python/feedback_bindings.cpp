#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/core/biome_climate_feedback.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_feedback_bindings(py::module& m) {
    // Bind BiomeClimateFeedbackParams struct
    py::class_<BiomeClimateFeedbackParams>(m, "BiomeClimateFeedbackParams")
        .def(py::init<>())
        .def_readwrite("albedo_feedback_strength", &BiomeClimateFeedbackParams::albedoFeedbackStrength)
        .def_readwrite("evapotranspiration_strength", &BiomeClimateFeedbackParams::evapotranspirationStrength)
        .def_readwrite("roughness_length_effect", &BiomeClimateFeedbackParams::roughnessLengthEffect)
        .def_readwrite("vegetation_growth_rate", &BiomeClimateFeedbackParams::vegetationGrowthRate)
        .def_readwrite("vegetation_dieback_rate", &BiomeClimateFeedbackParams::vegetationDiebackRate)
        .def_readwrite("carbon_sequestration_rate", &BiomeClimateFeedbackParams::carbonSequestrationRate)
        .def_readwrite("local_warming_effect", &BiomeClimateFeedbackParams::localWarmingEffect);
    
    // Bind BiomeClimateFeedback class
    py::class_<BiomeClimateFeedback>(m, "BiomeClimateFeedback")
        .def(py::init<Planet&, ClimateSystem&, BiomeSystem&>(),
             py::arg("planet"), py::arg("climate"), py::arg("biomes"))
        .def("initialize", &BiomeClimateFeedback::initialize)
        .def("set_parameters", &BiomeClimateFeedback::setParameters)
        .def("get_parameters", &BiomeClimateFeedback::getParameters)
        .def("update", &BiomeClimateFeedback::update,
             py::arg("time_step"), py::arg("day_of_year") = 182.0f)
        .def("calculate_albedo_effects", &BiomeClimateFeedback::calculateAlbedoEffects)
        .def("calculate_evapotranspiration_effects", &BiomeClimateFeedback::calculateEvapotranspirationEffects)
        .def("calculate_roughness_effects", &BiomeClimateFeedback::calculateRoughnessEffects)
        .def("calculate_carbon_sequestration", &BiomeClimateFeedback::calculateCarbonSequestration,
             py::arg("time_step"))
        .def("apply_feedback_to_climate", &BiomeClimateFeedback::applyFeedbackToClimate)
        .def("track_biome_changes", &BiomeClimateFeedback::trackBiomeChanges,
             py::arg("time_step"))
        .def("get_albedo_modifier", &BiomeClimateFeedback::getAlbedoModifier,
             py::arg("position"))
        .def("get_evapotranspiration_rate", &BiomeClimateFeedback::getEvapotranspirationRate,
             py::arg("position"))
        .def("get_roughness_length", &BiomeClimateFeedback::getRoughnessLength,
             py::arg("position"))
        .def("get_sequestered_carbon", &BiomeClimateFeedback::getSequesteredCarbon,
             py::arg("position"))
        .def("get_albedo_history", &BiomeClimateFeedback::getAlbedoHistory,
             py::return_value_policy::reference_internal)
        .def("get_temperature_history", &BiomeClimateFeedback::getTemperatureHistory,
             py::return_value_policy::reference_internal)
        .def("get_vegetation_history", &BiomeClimateFeedback::getVegetationHistory,
             py::return_value_policy::reference_internal);
}

} // namespace aeonterracpp