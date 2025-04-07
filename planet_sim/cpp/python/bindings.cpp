#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations of binding functions
void init_vector_types(py::module_& m);
void init_spherical_grid(py::module_& m);
void init_planet_bindings(py::module_& m);
void init_tectonics_bindings(py::module_& m);
void init_heightmap_bindings(py::module_& m);
void init_climate_bindings(py::module_& m);
void init_biome_bindings(py::module_& m);
void init_erosion_bindings(py::module_& m);
void init_feedback_bindings(py::module_& m);
void init_chemistry_bindings(py::module_& m);

PYBIND11_MODULE(_aeonterracpp, m) {
    m.doc() = "Python bindings for AeonTerra C++ implementation";
    
    // Vector types and utilities
    init_vector_types(m);
    
    // SphericalGrid
    init_spherical_grid(m);
    
    // Planet
    init_planet_bindings(m);
    
    // Tectonics
    init_tectonics_bindings(m);
    
    // Heightmap generation
    init_heightmap_bindings(m);
    
    // Climate system
    init_climate_bindings(m);
    
    // Biome system
    init_biome_bindings(m);
    
    // Erosion system
    init_erosion_bindings(m);
    
    // Biome-Climate feedback system
    init_feedback_bindings(m);
    
    // River-Ocean chemistry system
    init_chemistry_bindings(m);
}

// Define vector types bindings
void init_vector_types(py::module_& m) {
    // Import vector types
    using namespace aeonterracpp;
    
    // Vec3f binding
    py::class_<Vec3f>(m, "Vec3f", "3D vector with float precision")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def("__repr__", [](const Vec3f& v) {
            return "Vec3f(" + std::to_string(v.x()) + ", " + 
                            std::to_string(v.y()) + ", " + 
                            std::to_string(v.z()) + ")";
        })
        .def_property_readonly("x", &Vec3f::x)
        .def_property_readonly("y", &Vec3f::y)
        .def_property_readonly("z", &Vec3f::z)
        .def("norm", &Vec3f::norm, "Return the Euclidean norm")
        .def("normalized", &Vec3f::normalized, "Return a normalized copy of the vector")
        .def("dot", [](const Vec3f& self, const Vec3f& other) {
            return self.dot(other);
        }, "Compute dot product")
        .def("cross", [](const Vec3f& self, const Vec3f& other) {
            return self.cross(other);
        }, "Compute cross product")
        // Operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * float())
        .def(float() * py::self)
        .def(py::self / float());
    
    // Vec3d binding (similar to Vec3f)
    py::class_<Vec3d>(m, "Vec3d", "3D vector with double precision")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def("__repr__", [](const Vec3d& v) {
            return "Vec3d(" + std::to_string(v.x()) + ", " + 
                            std::to_string(v.y()) + ", " + 
                            std::to_string(v.z()) + ")";
        })
        .def_property_readonly("x", &Vec3d::x)
        .def_property_readonly("y", &Vec3d::y)
        .def_property_readonly("z", &Vec3d::z)
        .def("norm", &Vec3d::norm, "Return the Euclidean norm")
        .def("normalized", &Vec3d::normalized, "Return a normalized copy of the vector")
        .def("dot", [](const Vec3d& self, const Vec3d& other) {
            return self.dot(other);
        }, "Compute dot product")
        .def("cross", [](const Vec3d& self, const Vec3d& other) {
            return self.cross(other);
        }, "Compute cross product")
        // Operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self / double());
    
    // Vec3i binding
    py::class_<Vec3i>(m, "Vec3i", "3D vector with integer components")
        .def(py::init<>())
        .def(py::init<int, int, int>())
        .def("__repr__", [](const Vec3i& v) {
            return "Vec3i(" + std::to_string(v.x()) + ", " + 
                            std::to_string(v.y()) + ", " + 
                            std::to_string(v.z()) + ")";
        })
        .def_property_readonly("x", &Vec3i::x)
        .def_property_readonly("y", &Vec3i::y)
        .def_property_readonly("z", &Vec3i::z)
        // Indexing like a tuple/list
        .def("__getitem__", [](const Vec3i& v, size_t i) {
            if (i >= 3) throw py::index_error();
            return v[i];
        })
        .def("__len__", [](const Vec3i&) { return 3; })
        // Operators
        .def(py::self + py::self)
        .def(py::self - py::self);
    
    // BoundingBox binding
    py::class_<BoundingBox>(m, "BoundingBox", "Bounding box in lat/lon coordinates")
        .def(py::init<>())
        .def(py::init<float, float, float, float>(),
             py::arg("lat_min"), py::arg("lon_min"), py::arg("lat_max"), py::arg("lon_max"))
        .def_readwrite("lat_min", &BoundingBox::latMin)
        .def_readwrite("lon_min", &BoundingBox::lonMin)
        .def_readwrite("lat_max", &BoundingBox::latMax)
        .def_readwrite("lon_max", &BoundingBox::lonMax)
        .def("contains", &BoundingBox::contains, 
             "Check if a lat/lon point is within the bounding box")
        .def("width", &BoundingBox::width, "Get the width in longitude degrees")
        .def("height", &BoundingBox::height, "Get the height in latitude degrees")
        .def("__repr__", [](const BoundingBox& bb) {
            return "BoundingBox(lat_min=" + std::to_string(bb.latMin) + 
                   ", lon_min=" + std::to_string(bb.lonMin) + 
                   ", lat_max=" + std::to_string(bb.latMax) + 
                   ", lon_max=" + std::to_string(bb.lonMax) + ")";
        });
    
    // Utility functions
    m.def("to_radians", &toRadians, "Convert degrees to radians");
    m.def("to_degrees", &toDegrees, "Convert radians to degrees");
}