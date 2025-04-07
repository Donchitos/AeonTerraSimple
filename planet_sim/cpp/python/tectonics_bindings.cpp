#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/core/tectonics.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_tectonics_bindings(py::module& m) {
    // Enum for plate interaction types
    py::enum_<PlateInteractionType>(m, "PlateInteractionType")
        .value("None", PlateInteractionType::None)
        .value("Divergent", PlateInteractionType::Divergent)
        .value("Convergent", PlateInteractionType::Convergent)
        .value("Transform", PlateInteractionType::Transform)
        .value("Subduction", PlateInteractionType::Subduction)
        .value("Collision", PlateInteractionType::Collision)
        .export_values();

    // Bind TectonicPlate class
    py::class_<TectonicPlate>(m, "TectonicPlate")
        .def(py::init<int, const Vec3f&>())
        .def("get_id", &TectonicPlate::getId)
        .def("get_center", &TectonicPlate::getCenter)
        .def("get_movement", &TectonicPlate::getMovement)
        .def("set_movement", &TectonicPlate::setMovement)
        .def("get_vertices", &TectonicPlate::getVertices)
        .def("add_vertex", &TectonicPlate::addVertex)
        .def("calculate_area", &TectonicPlate::calculateArea)
        .def("is_oceanic", &TectonicPlate::isOceanic)
        .def("set_oceanic", &TectonicPlate::setOceanic)
        .def("calculate_bounding_box", &TectonicPlate::calculateBoundingBox);

    // Bind ConvectionCell class
    py::class_<ConvectionCell>(m, "ConvectionCell")
        .def(py::init<const Vec3f&, float>())
        .def("get_center", &ConvectionCell::getCenter)
        .def("get_strength", &ConvectionCell::getStrength)
        .def("calculate_force_at", &ConvectionCell::calculateForceAt);

    // Bind TectonicSimulation class
    py::class_<TectonicSimulation>(m, "TectonicSimulation")
        .def(py::init<Planet&, int, unsigned int>(), 
             py::arg("planet"), 
             py::arg("num_plates") = 7, 
             py::arg("seed") = 0)
        .def("initialize_plates", &TectonicSimulation::initializePlates)
        .def("initialize_mantle", &TectonicSimulation::initializeMantle)
        .def("simulate_step", &TectonicSimulation::simulateStep)
        .def("get_boundary_vertices", &TectonicSimulation::getBoundaryVertices)
        .def("get_interaction_type", &TectonicSimulation::getInteractionType)
        .def("get_plates", &TectonicSimulation::getPlates, py::return_value_policy::reference_internal)
        .def("set_tectonic_activity", &TectonicSimulation::setTectonicActivity)
        .def("get_tectonic_activity", &TectonicSimulation::getTectonicActivity)
        .def("calculate_isostatic_adjustment", &TectonicSimulation::calculateIsostaticAdjustment)
        .def("get_super_continent_cycle_phase", &TectonicSimulation::getSuperContinentCyclePhase)
        
        // Expose ClimateSystem's data
        .def_property_readonly("temperature", [](const TectonicSimulation& sim) {
            const auto& climate = sim.getClimateSystem();
            return py::array_t<float>(
                {climate.temperature.size()},
                {sizeof(float)},
                climate.temperature.data(),
                py::cast(sim)
            );
        })
        .def_property_readonly("precipitation", [](const TectonicSimulation& sim) {
            const auto& climate = sim.getClimateSystem();
            return py::array_t<float>(
                {climate.precipitation.size()},
                {sizeof(float)},
                climate.precipitation.data(),
                py::cast(sim)
            );
        })
        .def_property_readonly("weathering_rate", [](const TectonicSimulation& sim) {
            const auto& climate = sim.getClimateSystem();
            return py::array_t<float>(
                {climate.weatheringRate.size()},
                {sizeof(float)},
                climate.weatheringRate.data(),
                py::cast(sim)
            );
        })
        
        // Expose HydrologicalSystem's data
        .def_property_readonly("drainage", [](const TectonicSimulation& sim) {
            const auto& hydro = sim.getHydrologicalSystem();
            return py::array_t<float>(
                {hydro.drainage.size()},
                {sizeof(float)},
                hydro.drainage.data(),
                py::cast(sim)
            );
        })
        .def_property_readonly("river_flow", [](const TectonicSimulation& sim) {
            const auto& hydro = sim.getHydrologicalSystem();
            return py::array_t<float>(
                {hydro.riverFlow.size()},
                {sizeof(float)},
                hydro.riverFlow.data(),
                py::cast(sim)
            );
        })
        .def_property_readonly("is_river", [](const TectonicSimulation& sim) {
            const auto& hydro = sim.getHydrologicalSystem();
            return py::array_t<bool>(
                {hydro.isRiver.size()},
                {sizeof(bool)},
                hydro.isRiver.data(),
                py::cast(sim)
            );
        })
        .def_property_readonly("is_lake", [](const TectonicSimulation& sim) {
            const auto& hydro = sim.getHydrologicalSystem();
            return py::array_t<bool>(
                {hydro.isLake.size()},
                {sizeof(bool)},
                hydro.isLake.data(),
                py::cast(sim)
            );
        });
}

} // namespace aeonterracpp