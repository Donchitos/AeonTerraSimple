#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "aeonterracpp/core/climate_system.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_climate_bindings(py::module& m) {
    // Bind ClimateParams struct
    py::class_<ClimateParams>(m, "ClimateParams")
        .def(py::init<>())
        // General parameters
        .def_readwrite("solar_constant", &ClimateParams::solarConstant)
        .def_readwrite("planetary_albedo", &ClimateParams::planetaryAlbedo)
        .def_readwrite("atmospheric_composition", &ClimateParams::atmosphericComposition)
        .def_readwrite("axial_tilt", &ClimateParams::axialTilt)
        .def_readwrite("day_length", &ClimateParams::dayLength)
        .def_readwrite("year_length", &ClimateParams::yearLength)
        .def_readwrite("eccentricity", &ClimateParams::eccentricity)
        // Atmospheric parameters
        .def_readwrite("hadley_cell_width", &ClimateParams::hadleyCellWidth)
        .def_readwrite("ferrel_cell_width", &ClimateParams::ferrelCellWidth)
        .def_readwrite("polar_cell_width", &ClimateParams::polarCellWidth)
        .def_readwrite("jet_stream_latitude", &ClimateParams::jetStreamLatitude)
        .def_readwrite("trade_wind_speed", &ClimateParams::tradeWindSpeed)
        .def_readwrite("westerlies_speed", &ClimateParams::westerliesSpeed)
        .def_readwrite("polar_easterlies_speed", &ClimateParams::polarEasterliesSpeed)
        // Oceanic parameters
        .def_readwrite("ocean_heat_capacity", &ClimateParams::oceanHeatCapacity)
        .def_readwrite("gyre_strength", &ClimateParams::gyreStrength)
        .def_readwrite("deep_ocean_circulation", &ClimateParams::deepOceanCirculation)
        .def_readwrite("upwelling_strength", &ClimateParams::upwellingStrength)
        // Precipitation parameters
        .def_readwrite("global_precipitation", &ClimateParams::globalPrecipitation)
        .def_readwrite("orographic_effect", &ClimateParams::orographicEffect)
        .def_readwrite("convectional_effect", &ClimateParams::convectionalEffect)
        .def_readwrite("frontal_effect", &ClimateParams::frontalEffect)
        // Seasonal parameters
        .def_readwrite("seasonal_variation", &ClimateParams::seasonalVariation)
        .def_readwrite("monsoon_strength", &ClimateParams::monsoonStrength);
    
    // Bind AtmosphericCirculation class
    py::class_<AtmosphericCirculation>(m, "AtmosphericCirculation")
        .def("get_wind_vector", &AtmosphericCirculation::getWindVector)
        .def("get_pressure", &AtmosphericCirculation::getPressure)
        .def("get_temperature", &AtmosphericCirculation::getTemperature)
        .def("get_wind_data", &AtmosphericCirculation::getWindData, py::return_value_policy::reference_internal)
        .def("get_pressure_data", &AtmosphericCirculation::getPressureData, py::return_value_policy::reference_internal)
        .def("get_temperature_data", &AtmosphericCirculation::getTemperatureData, py::return_value_policy::reference_internal);
    
    // Bind OceanicCirculation class
    py::class_<OceanicCirculation>(m, "OceanicCirculation")
        .def("get_current_vector", &OceanicCirculation::getCurrentVector)
        .def("get_sea_surface_temperature", &OceanicCirculation::getSeaSurfaceTemperature)
        .def("get_current_data", &OceanicCirculation::getCurrentData, py::return_value_policy::reference_internal)
        .def("get_sea_surface_temperature_data", &OceanicCirculation::getSeaSurfaceTemperatureData, py::return_value_policy::reference_internal);
    
    // Bind HydrologicalSystem class
    py::class_<HydrologicalSystem>(m, "HydrologicalSystem")
        .def("get_precipitation", &HydrologicalSystem::getPrecipitation)
        .def("get_evaporation", &HydrologicalSystem::getEvaporation)
        .def("get_river_flow", &HydrologicalSystem::getRiverFlow)
        .def("get_precipitation_data", &HydrologicalSystem::getPrecipitationData, py::return_value_policy::reference_internal)
        .def("get_evaporation_data", &HydrologicalSystem::getEvaporationData, py::return_value_policy::reference_internal)
        .def("get_river_flow_data", &HydrologicalSystem::getRiverFlowData, py::return_value_policy::reference_internal)
        .def("get_river_data", &HydrologicalSystem::getRiverData, py::return_value_policy::reference_internal)
        .def("get_lake_data", &HydrologicalSystem::getLakeData, py::return_value_policy::reference_internal);
    
    // Bind ClimateSystem class
    py::class_<ClimateSystem>(m, "ClimateSystem")
        .def(py::init<Planet&>())
        .def("set_parameters", &ClimateSystem::setParameters)
        .def("get_parameters", &ClimateSystem::getParameters)
        .def("initialize", &ClimateSystem::initialize)
        .def("simulate", &ClimateSystem::simulate,
             py::arg("time_step"),
             py::arg("day_of_year") = 182.0f)
        .def("get_atmosphere", &ClimateSystem::getAtmosphere, py::return_value_policy::reference_internal)
        .def("get_ocean", &ClimateSystem::getOcean, py::return_value_policy::reference_internal)
        .def("get_hydrology", &ClimateSystem::getHydrology, py::return_value_policy::reference_internal)
        .def("calculate_climate_zones", &ClimateSystem::calculateClimateZones)
        .def("simulate_weathering", &ClimateSystem::simulateWeathering)
        .def("save_to_file", &ClimateSystem::saveToFile)
        .def("load_from_file", &ClimateSystem::loadFromFile);
}

} // namespace aeonterracpp