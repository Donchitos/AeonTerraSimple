#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/core/river_ocean_chemistry.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_chemistry_bindings(py::module& m) {
    // Bind ChemicalElement enum
    py::enum_<ChemicalElement>(m, "ChemicalElement")
        .value("Sediment", ChemicalElement::Sediment)
        .value("Nitrogen", ChemicalElement::Nitrogen)
        .value("Phosphorus", ChemicalElement::Phosphorus)
        .value("Carbon", ChemicalElement::Carbon)
        .value("Oxygen", ChemicalElement::Oxygen)
        .value("Calcium", ChemicalElement::Calcium)
        .value("Silica", ChemicalElement::Silica)
        .value("Iron", ChemicalElement::Iron)
        .export_values();
    
    // Bind RiverOceanChemistryParams struct
    py::class_<RiverOceanChemistryParams>(m, "RiverOceanChemistryParams")
        .def(py::init<>())
        .def_readwrite("river_erosion_sediment_factor", &RiverOceanChemistryParams::riverErosionSedimentFactor)
        .def_readwrite("river_nutrient_leaching_rate", &RiverOceanChemistryParams::riverNutrientLeachingRate)
        .def_readwrite("river_carbon_export_rate", &RiverOceanChemistryParams::riverCarbonExportRate)
        .def_readwrite("ocean_circulation_mixing_rate", &RiverOceanChemistryParams::oceanCirculationMixingRate)
        .def_readwrite("ocean_sedimentation_rate", &RiverOceanChemistryParams::oceanSedimentationRate)
        .def_readwrite("ocean_primary_production_rate", &RiverOceanChemistryParams::oceanPrimaryProductionRate)
        .def_readwrite("ocean_oxygenation_rate", &RiverOceanChemistryParams::oceanOxygenationRate)
        .def_readwrite("carbonate_precipitation_rate", &RiverOceanChemistryParams::carbonatePrecipitationRate)
        .def_readwrite("silica_precipitation_rate", &RiverOceanChemistryParams::silicaPrecipitationRate)
        .def_readwrite("iron_oxidation_rate", &RiverOceanChemistryParams::ironOxidationRate)
        .def_readwrite("time_scale_factor", &RiverOceanChemistryParams::timeScaleFactor);
    
    // Bind RiverOceanChemistry class
    py::class_<RiverOceanChemistry>(m, "RiverOceanChemistry")
        .def(py::init<Planet&, ClimateSystem&, const EnhancedErosion&>(),
             py::arg("planet"), py::arg("climate"), py::arg("erosion"))
        .def("initialize", &RiverOceanChemistry::initialize,
             py::arg("params") = RiverOceanChemistryParams())
        .def("set_parameters", &RiverOceanChemistry::setParameters)
        .def("get_parameters", &RiverOceanChemistry::getParameters)
        .def("update", &RiverOceanChemistry::update,
             py::arg("time_step"))
        .def("calculate_river_input", &RiverOceanChemistry::calculateRiverInput)
        .def("calculate_ocean_mixing", &RiverOceanChemistry::calculateOceanMixing,
             py::arg("time_step"))
        .def("calculate_precipitation_and_sedimentation", &RiverOceanChemistry::calculatePrecipitationAndSedimentation,
             py::arg("time_step"))
        .def("calculate_biological_processes", &RiverOceanChemistry::calculateBiologicalProcesses,
             py::arg("time_step"))
        .def("apply_chemical_effects_to_climate", &RiverOceanChemistry::applyChemicalEffectsToClimate)
        .def("get_chemical_concentration", &RiverOceanChemistry::getChemicalConcentration,
             py::arg("position"), py::arg("element"))
        .def("get_total_river_input", &RiverOceanChemistry::getTotalRiverInput,
             py::arg("element"))
        .def("get_average_ocean_concentration", &RiverOceanChemistry::getAverageOceanConcentration,
             py::arg("element"))
        .def("get_river_input_history", &RiverOceanChemistry::getRiverInputHistory,
             py::arg("element"), py::return_value_policy::reference_internal)
        .def("get_ocean_concentration_history", &RiverOceanChemistry::getOceanConcentrationHistory,
             py::arg("element"), py::return_value_policy::reference_internal)
        .def("get_element_name", &RiverOceanChemistry::getElementName,
             py::arg("element"))
        .def("get_element_unit", &RiverOceanChemistry::getElementUnit,
             py::arg("element"));
}

} // namespace aeonterracpp