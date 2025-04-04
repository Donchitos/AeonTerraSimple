#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/core/tectonics.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_tectonics_bindings(py::module& m) {
    // Bind PlateInteractionType enum
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
        .def("calculate_isostatic_adjustment", &TectonicSimulation::calculateIsostaticAdjustment);
}

} // namespace aeonterracpp