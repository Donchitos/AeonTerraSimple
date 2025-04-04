#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "aeonterracpp/core/spherical_grid.h"

namespace py = pybind11;

void init_spherical_grid(py::module_& m) {
    using namespace aeonterracpp;
    
    py::class_<SphericalGrid>(m, "SphericalGrid", "Represents a planet's surface as a spherical mesh")
        // Constructors
        .def(py::init<float, int>(),
             py::arg("radius") = 6371.0f,
             py::arg("resolution") = 5,
             "Create a new SphericalGrid with specified radius and resolution")
        
        // Basic properties
        .def_property_readonly("vertex_count", &SphericalGrid::getVertexCount,
                              "Get the number of vertices in the grid")
        .def_property_readonly("face_count", &SphericalGrid::getFaceCount,
                              "Get the number of faces in the grid")
        .def_property_readonly("resolution", &SphericalGrid::getResolution,
                              "Get the resolution level of the grid")
        .def_property_readonly("radius", &SphericalGrid::getRadius,
                              "Get the radius of the planet in kilometers")
        
        // Vertex and face access
        .def("get_vertex", &SphericalGrid::getVertex, py::arg("index"),
             "Get the 3D position of a vertex by index")
        .def("get_face", &SphericalGrid::getFace, py::arg("index"),
             "Get the indices of vertices forming a triangular face")
        .def("get_neighbors", &SphericalGrid::getNeighbors, py::arg("vertex_idx"),
             "Get the indices of vertices adjacent to the given vertex")
        
        // Manipulation
        .def("subdivide", &SphericalGrid::subdivide, py::arg("levels"),
             "Increase the resolution of the entire grid by subdividing each face")
        .def("subdivide_region", &SphericalGrid::subdivideRegion, 
             py::arg("region"), py::arg("additional_levels"),
             "Increase the resolution of a specific region of the grid")
        
        // Coordinate conversion
        .def("lat_lon_to_cartesian", &SphericalGrid::latLonToCartesian,
             py::arg("lat"), py::arg("lon"),
             "Convert latitude and longitude (in degrees) to 3D Cartesian coordinates")
        .def("cartesian_to_lat_lon", &SphericalGrid::cartesianToLatLon,
             py::arg("position"),
             "Convert 3D Cartesian coordinates to latitude and longitude (in degrees)")
        
        // Queries
        .def("find_nearest_vertex", &SphericalGrid::findNearestVertex,
             py::arg("lat"), py::arg("lon"),
             "Find the index of the vertex nearest to the given lat/lon position")
        .def("get_vertices_in_region", &SphericalGrid::getVerticesInRegion,
             py::arg("region"),
             "Get all vertex indices within the specified lat/lon region")
        .def("calculate_distance", &SphericalGrid::calculateDistance,
             py::arg("vertex_idx1"), py::arg("vertex_idx2"),
             "Calculate the great-circle distance between two vertices in kilometers")
        
        // Python-specific extensions
        // Get vertices as numpy array
        .def("get_vertices_array", [](const SphericalGrid& self) {
            size_t count = self.getVertexCount();
            py::array_t<float> result({count, 3});
            py::buffer_info buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);
            
            for (size_t i = 0; i < count; i++) {
                const Vec3f& v = self.getVertex(i);
                ptr[i*3 + 0] = v.x();
                ptr[i*3 + 1] = v.y();
                ptr[i*3 + 2] = v.z();
            }
            
            return result;
        }, "Get all vertices as a numpy array of shape (n, 3)")
        
        // Get faces as numpy array
        .def("get_faces_array", [](const SphericalGrid& self) {
            size_t count = self.getFaceCount();
            py::array_t<int> result({count, 3});
            py::buffer_info buf = result.request();
            int* ptr = static_cast<int*>(buf.ptr);
            
            for (size_t i = 0; i < count; i++) {
                const Vec3i& f = self.getFace(i);
                ptr[i*3 + 0] = f.x();
                ptr[i*3 + 1] = f.y();
                ptr[i*3 + 2] = f.z();
            }
            
            return result;
        }, "Get all faces as a numpy array of shape (n, 3)")
        
        // Iterator for vertices
        .def("__iter__", [](const SphericalGrid& self) {
            return py::make_iterator(self.getVertex(0), self.getVertex(self.getVertexCount()));
        }, py::keep_alive<0, 1>())
        
        // Length operator (returns vertex count)
        .def("__len__", &SphericalGrid::getVertexCount)
        
        // Representation
        .def("__repr__", [](const SphericalGrid& self) {
            return "SphericalGrid(radius=" + std::to_string(self.getRadius()) + 
                   ", resolution=" + std::to_string(self.getResolution()) + 
                   ", vertices=" + std::to_string(self.getVertexCount()) + 
                   ", faces=" + std::to_string(self.getFaceCount()) + ")";
        });
}