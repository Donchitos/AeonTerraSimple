import numpy as np

class SphericalGrid:
    """
    Represents a planet's surface as a spherical grid.
    """
    
    def __init__(self, radius=6371.0, resolution=5):
        """
        Initialize a spherical grid.
        
        Parameters:
        - radius: Planet radius in kilometers
        - resolution: Detail level of the icosphere (higher = more detailed)
        """
        self.radius = radius
        self.resolution = resolution
        
        # Initialize the base icosahedron
        self.vertices, self.faces = self._create_icosahedron()
        
        # Subdivide to desired resolution
        for _ in range(resolution):
            self.vertices, self.faces = self._subdivide()
    
    def _create_icosahedron(self):
        """Creates a base icosahedron"""
        # Golden ratio
        t = (1.0 + np.sqrt(5.0)) / 2.0
        
        # Vertices of a icosahedron
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        
        # Normalize to unit sphere
        for i in range(len(vertices)):
            norm = np.linalg.norm(vertices[i])
            vertices[i] = vertices[i] / norm * self.radius
        
        # Faces of an icosahedron (20 triangular faces)
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        return vertices, faces
    
    def _subdivide(self):
        """Subdivide each face into 4 faces"""
        new_vertices = list(self.vertices)
        new_faces = []
        
        # Cache for midpoints
        midpoint_cache = {}
        
        def get_midpoint(a, b):
            """Get or compute the midpoint between two vertices"""
            # Create a unique key for this edge
            edge_key = tuple(sorted([a, b]))
            
            # Return cached midpoint if exists
            if edge_key in midpoint_cache:
                return midpoint_cache[edge_key]
            
            # Otherwise compute it
            pa = self.vertices[a]
            pb = self.vertices[b]
            midpoint = (pa + pb) / 2.0
            
            # Normalize to sphere
            midpoint = midpoint / np.linalg.norm(midpoint) * self.radius
            
            # Add to vertices and cache
            idx = len(new_vertices)
            new_vertices.append(midpoint)
            midpoint_cache[edge_key] = idx
            
            return idx
        
        # Subdivide each face into 4 new faces
        for face in self.faces:
            v0, v1, v2 = face
            
            # Get midpoints of each edge
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            
            # Create 4 new faces
            new_faces.append([v0, a, c])
            new_faces.append([v1, b, a])
            new_faces.append([v2, c, b])
            new_faces.append([a, b, c])
        
        return np.array(new_vertices), new_faces
    
    def get_vertex_neighbors(self):
        """Build adjacency list for each vertex"""
        neighbors = [[] for _ in range(len(self.vertices))]
        
        for face in self.faces:
            a, b, c = face
            neighbors[a].extend([b, c])
            neighbors[b].extend([a, c])
            neighbors[c].extend([a, b])
        
        # Remove duplicates
        for i in range(len(neighbors)):
            neighbors[i] = list(set(neighbors[i]))
        
        return neighbors
    
    def get_lat_lon(self, vertex_idx):
        """Convert vertex coordinates to latitude/longitude (in degrees)"""
        x, y, z = self.vertices[vertex_idx]
        
        # Convert to latitude and longitude
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arcsin(z / self.radius))
        
        return lat, lon