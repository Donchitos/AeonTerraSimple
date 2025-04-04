#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "aeonterracpp/core/planet.h"

namespace py = pybind11;

namespace aeonterracpp {

void init_planet_bindings(py::module& m) {
    // Bind HistoricalSnapshot struct
    py::class_<HistoricalSnapshot>(m, "HistoricalSnapshot")
        .def(py::init<double, size_t>())
        .def_readonly("age", &HistoricalSnapshot::age)
        .def_readonly("max_elevation", &HistoricalSnapshot::maxElevation)
        .def_readonly("min_elevation", &HistoricalSnapshot::minElevation)
        .def_readonly("ocean_percentage", &HistoricalSnapshot::oceanPercentage)
        .def_readonly("plate_count", &HistoricalSnapshot::plateCount)
        .def_property_readonly("elevation", [](const HistoricalSnapshot& snapshot) {
            return py::array_t<float>(
                {snapshot.elevation.size()},          // Shape
                {sizeof(float)},                      // Strides
                snapshot.elevation.data(),            // Data pointer
                py::cast(snapshot)                    // Parent object (keeps data alive)
            );
        })
        .def_property_readonly("plate_id", [](const HistoricalSnapshot& snapshot) {
            return py::array_t<int>(
                {snapshot.plateId.size()},           // Shape
                {sizeof(int)},                       // Strides
                snapshot.plateId.data(),             // Data pointer
                py::cast(snapshot)                   // Parent object (keeps data alive)
            );
        });

    // Bind Planet class
    py::class_<Planet>(m, "Planet")
        .def(py::init<float, int, const std::string&>(),
             py::arg("radius") = 6371.0f,
             py::arg("resolution") = 5,
             py::arg("name") = "AeonTerra")
        
        // Property access
        .def("get_elevation", &Planet::getElevation)
        .def("set_elevation", &Planet::setElevation)
        .def("get_crust_thickness", &Planet::getCrustThickness)
        .def("set_crust_thickness", &Planet::setCrustThickness)
        .def("get_crust_density", &Planet::getCrustDensity)
        .def("set_crust_density", &Planet::setCrustDensity)
        .def("get_crust_type", &Planet::getCrustType)
        .def("set_crust_type", &Planet::setCrustType)
        .def("get_plate_id", &Planet::getPlateId)
        .def("set_plate_id", &Planet::setPlateId)
        
        // Bulk property access as numpy arrays
        .def_property_readonly("elevation", [](const Planet& planet) {
            size_t vertexCount = planet.getVertexCount();
            std::vector<float> temp(vertexCount);
            for (size_t i = 0; i < vertexCount; ++i) {
                temp[i] = planet.getElevation(i);
            }
            return py::array_t<float>(
                {vertexCount},                       // Shape
                {sizeof(float)},                     // Strides
                temp.data()                          // Data pointer
            );
        })
        .def_property_readonly("crust_thickness", [](const Planet& planet) {
            size_t vertexCount = planet.getVertexCount();
            std::vector<float> temp(vertexCount);
            for (size_t i = 0; i < vertexCount; ++i) {
                temp[i] = planet.getCrustThickness(i);
            }
            return py::array_t<float>(
                {vertexCount},                       // Shape
                {sizeof(float)},                     // Strides
                temp.data()                          // Data pointer
            );
        })
        .def_property_readonly("plate_id", [](const Planet& planet) {
            size_t vertexCount = planet.getVertexCount();
            std::vector<int> temp(vertexCount);
            for (size_t i = 0; i < vertexCount; ++i) {
                temp[i] = planet.getPlateId(i);
            }
            return py::array_t<int>(
                {vertexCount},                       // Shape
                {sizeof(int)},                       // Strides
                temp.data()                          // Data pointer
            );
        })
        
        // Grid access
        .def("get_grid", &Planet::getGrid, py::return_value_policy::reference_internal)
        .def("get_vertex_count", &Planet::getVertexCount)
        .def("get_radius", &Planet::getRadius)
        
        // Simulation control
        .def("advance_age", &Planet::advanceAge)
        .def("get_age", &Planet::getAge)
        .def("record_snapshot", &Planet::recordSnapshot)
        .def("get_name", &Planet::getName)
        .def("set_name", &Planet::setName)
        
        // History access
        .def("get_history", &Planet::getHistory, py::return_value_policy::reference_internal)
        .def("clear_history", &Planet::clearHistory)
        
        // Serialization
        .def("save_to_file", &Planet::saveToFile)
        .def_static("load_from_file", &Planet::loadFromFile);
}

} // namespace aeonterracpp