import numpy as np

class ErosionSimulation:
    """
    Simulates terrain erosion processes including hydraulic erosion,
    thermal erosion, and sedimentation.
    """
    
    def __init__(self, planet):
        """
        Initialize the erosion simulation.
        
        Parameters:
        - planet: A Planet object
        """
        self.planet = planet
        self.neighbors = self.planet.grid.get_vertex_neighbors()
    
    def simulate(self, iterations=5, rainfall=1.0, erosion_rate=0.05):
        """
        Run erosion simulation.
        
        Parameters:
        - iterations: Number of erosion cycles
        - rainfall: Global rainfall factor (0-2.0)
        - erosion_rate: Rate of erosion (0-1.0)
        """
        print(f"Running erosion simulation for {iterations} iterations...")
        
        for i in range(iterations):
            print(f"  Erosion cycle {i+1}/{iterations}")
            
            # Run each erosion type
            self._hydraulic_erosion(rainfall, erosion_rate)
            self._thermal_erosion(erosion_rate * 0.5)
            self._sedimentation()
        
        return self
    
    def _hydraulic_erosion(self, rainfall, erosion_rate):
        """Simulate erosion caused by water flow"""
        # Temporary arrays for water and sediment
        water = np.zeros_like(self.planet.elevation)
        sediment = np.zeros_like(self.planet.elevation)
        
        # Add rainfall (more on higher elevations to simulate orographic effect)
        elevation_normalized = np.clip((self.planet.elevation - np.min(self.planet.elevation)) / 
                                      (np.max(self.planet.elevation) - np.min(self.planet.elevation)), 
                                      0, 1)
        water = rainfall * (0.5 + 0.5 * elevation_normalized)
        
        # Water shouldn't accumulate in oceans
        water[self.planet.elevation < 0] = 0
        
        # Track vertices by elevation (for processing from highest to lowest)
        indices = np.argsort(self.planet.elevation)[::-1]
        
        # Process water flow from highest to lowest points
        for idx in indices:
            if water[idx] <= 0:
                continue
                
            # Check neighbors
            neighbors = self.neighbors[idx]
            lower_neighbors = [n for n in neighbors if self.planet.elevation[n] < self.planet.elevation[idx]]
            
            if not lower_neighbors:
                continue  # No lower neighbors, water stays here
                
            # Calculate flow to lower neighbors
            total_elevation_diff = sum(self.planet.elevation[idx] - self.planet.elevation[n] for n in lower_neighbors)
            
            if total_elevation_diff > 0:
                # Erode current vertex based on water amount and slope
                erosion_amount = min(water[idx] * erosion_rate, self.planet.elevation[idx] * 0.01)
                self.planet.elevation[idx] -= erosion_amount
                sediment[idx] += erosion_amount * 0.3  # Some sediment stays
                
                # Distribute remaining water and sediment to lower neighbors
                for n in lower_neighbors:
                    # Proportion of flow to this neighbor
                    flow_ratio = (self.planet.elevation[idx] - self.planet.elevation[n]) / total_elevation_diff
                    
                    # Move water
                    water_transfer = water[idx] * flow_ratio * 0.9  # Some water is lost
                    water[idx] -= water_transfer
                    water[n] += water_transfer
                    
                    # Move sediment
                    sediment_transfer = erosion_amount * 0.7 * flow_ratio  # Rest of the sediment moves
                    sediment[n] += sediment_transfer
        
        # Deposit sediment where water slows down
        for idx in indices[::-1]:  # Process from lowest to highest
            if sediment[idx] > 0:
                # Deposit a portion of the sediment
                deposit_amount = sediment[idx] * 0.8
                self.planet.elevation[idx] += deposit_amount
                sediment[idx] -= deposit_amount
    
    def _thermal_erosion(self, erosion_rate):
        """Simulate erosion caused by weathering and gravity"""
        # Temporary array for elevation changes
        elevation_delta = np.zeros_like(self.planet.elevation)
        
        # For each vertex, check its neighbors
        for idx in range(len(self.planet.elevation)):
            if self.planet.elevation[idx] < 0:
                continue  # Skip underwater points
                
            neighbors = self.neighbors[idx]
            
            # Calculate average elevation of neighbors
            avg_neighbor_elevation = sum(self.planet.elevation[n] for n in neighbors) / len(neighbors)
            
            # If this point is higher than average, erode it
            if self.planet.elevation[idx] > avg_neighbor_elevation:
                # Erosion rate is proportional to the difference
                diff = self.planet.elevation[idx] - avg_neighbor_elevation
                # Steeper slopes erode faster
                erosion = diff * erosion_rate
                
                # Erode this point
                elevation_delta[idx] -= erosion
                
                # Distribute the eroded material to lower neighbors
                lower_neighbors = [n for n in neighbors if self.planet.elevation[n] < self.planet.elevation[idx]]
                
                if lower_neighbors:
                    # Add the eroded material to lower neighbors
                    deposit_per_neighbor = erosion / len(lower_neighbors)
                    for n in lower_neighbors:
                        elevation_delta[n] += deposit_per_neighbor
        
        # Apply all changes
        self.planet.elevation += elevation_delta
    
    def _sedimentation(self):
        """Fill in low areas with sediment"""
        # Find depressions (points lower than all neighbors)
        for idx in range(len(self.planet.elevation)):
            if self.planet.elevation[idx] < 0:
                continue  # Skip underwater points
                
            neighbors = self.neighbors[idx]
            avg_neighbor_elevation = sum(self.planet.elevation[n] for n in neighbors) / len(neighbors)
            
            # If this is a depression, fill it slightly
            if all(self.planet.elevation[idx] < self.planet.elevation[n] for n in neighbors):
                # Fill depression partially
                self.planet.elevation[idx] += (avg_neighbor_elevation - self.planet.elevation[idx]) * 0.2
    
    def generate_rivers(self, min_flow=10):
        """
        Generate rivers based on water flow.
        
        Parameters:
        - min_flow: Minimum water flow to create a river
        
        Returns a list of river paths (each path is a list of vertex indices)
        """
        # Simulate water flow to find river paths
        water_flow = np.zeros(len(self.planet.elevation))
        
        # Add rainfall to all land areas
        for idx in range(len(self.planet.elevation)):
            if self.planet.elevation[idx] > 0:  # Land only
                water_flow[idx] = 1.0
        
        # Flow direction for each vertex (where water goes from here)
        flow_dir = [-1] * len(self.planet.elevation)
        
        # Compute flow directions - water flows to the lowest neighbor
        for idx in range(len(self.planet.elevation)):
            if self.planet.elevation[idx] < 0:
                continue  # Skip ocean
            
            neighbors = self.neighbors[idx]
            if not neighbors:
                continue
                
            # Find lowest neighbor
            lowest = min(neighbors, key=lambda n: self.planet.elevation[n])
            
            # If lowest neighbor is lower than current, water flows there
            if self.planet.elevation[lowest] < self.planet.elevation[idx]:
                flow_dir[idx] = lowest
        
        # Accumulate water flow
        for idx in range(len(self.planet.elevation)):
            if flow_dir[idx] != -1:
                next_idx = flow_dir[idx]
                water_flow[next_idx] += water_flow[idx]
        
        # Identify river paths
        rivers = []
        for idx in range(len(self.planet.elevation)):
            # If this point has significant water flow and isn't already part of a river
            if water_flow[idx] > min_flow and self.planet.elevation[idx] > 0:
                # Trace the river downstream
                river = [idx]
                current = idx
                
                while True:
                    next_idx = flow_dir[current]
                    if next_idx == -1 or self.planet.elevation[next_idx] < 0:
                        # Reached ocean or flow endpoint
                        break
                        
                    river.append(next_idx)
                    current = next_idx
                    
                    # Safety to prevent infinite loops
                    if len(river) > 1000:
                        break
                
                # Only keep rivers of reasonable length
                if len(river) > 3:
                    rivers.append(river)
        
        return rivers