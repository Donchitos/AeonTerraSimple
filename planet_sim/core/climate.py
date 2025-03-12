import numpy as np

class ClimateSimulation:
    """
    Simulates climate patterns including temperature and precipitation.
    """
    
    def __init__(self, planet):
        """
        Initialize the climate simulation.
        
        Parameters:
        - planet: A Planet object
        """
        self.planet = planet
        
        # Initialize climate data arrays if not already present
        num_vertices = len(self.planet.grid.vertices)
        if self.planet.temperature is None:
            self.planet.temperature = np.zeros(num_vertices)
        if self.planet.precipitation is None:
            self.planet.precipitation = np.zeros(num_vertices)
            
        # Climate parameters
        self.solar_constant = 1366.0  # W/m², Earth's solar constant
        self.greenhouse_factor = 1.0  # Earth-like greenhouse effect
        self.albedo = 0.3  # Earth's average albedo
        
    def simulate(self, seasons=1):
        """
        Run climate simulation.
        
        Parameters:
        - seasons: Number of seasons to simulate (averaging the results)
        """
        print(f"Simulating climate with {seasons} seasons...")
        
        # Reset climate data
        self.planet.temperature.fill(0)
        self.planet.precipitation.fill(0)
        
        # Run simulation for each season
        for season in range(seasons):
            season_angle = (season / seasons) * 2 * np.pi
            self._calculate_temperature(season_angle)
            self._calculate_precipitation()
            
        # Average results across seasons
        if seasons > 1:
            self.planet.temperature /= seasons
            self.planet.precipitation /= seasons
            
        # Adjust temperature and precipitation based on terrain features
        self._apply_terrain_effects()
        
        print("Climate simulation complete.")
        return self
    
    def _calculate_temperature(self, season_angle):
        """
        Calculate temperature based on latitude, elevation, and season.
        
        Parameters:
        - season_angle: Angle in radians representing the season (0 = northern hemisphere summer)
        """
        # Get coordinates of all vertices
        vertices = self.planet.grid.vertices
        
        for i, vertex in enumerate(vertices):
            # Calculate latitude and longitude
            x, y, z = vertex
            latitude = np.arcsin(z / np.linalg.norm(vertex)) * 180 / np.pi
            
            # Adjust latitude for axial tilt and season
            effective_latitude = latitude - self.planet.axial_tilt * np.cos(season_angle)
            
            # Base temperature depends on latitude (solar input)
            # Uses cosine function to model solar energy distribution
            base_temp = self.solar_constant * (1 - self.albedo) * np.cos(np.radians(effective_latitude)) / 4
            
            # Convert to Celsius and adjust
            base_temp = base_temp / 15 - 273.15  # Simplified conversion to Celsius
            
            # Apply greenhouse effect
            base_temp += 33 * self.greenhouse_factor  # Earth's greenhouse effect adds about 33°C
            
            # Elevation effect (temperature decreases with height)
            # Lapse rate on Earth is about 6.5°C/km
            elevation = self.planet.elevation[i]
            if elevation > 0:  # Only apply to land
                base_temp -= elevation * 6.5
                
            # Add seasonal variation based on hemisphere
            seasonal_variation = 15 * np.sin(season_angle) * (np.sin(np.radians(latitude)))
            base_temp += seasonal_variation
            
            # Add to temperature data
            self.planet.temperature[i] += base_temp
    
    def _calculate_precipitation(self):
        """Calculate precipitation patterns based on temperature and planet features"""
        # Simple precipitation model based on temperature and elevation
        for i in range(len(self.planet.temperature)):
            temp = self.planet.temperature[i]
            elev = self.planet.elevation[i]
            
            # Base precipitation increases with temperature (more evaporation)
            # but drops off at very high temperatures (desert effect)
            if temp < -20:
                precip = 100  # Very cold = little precipitation
            elif temp < 0:
                precip = 300 + (temp + 20) * 10  # Cold to moderate
            elif temp < 20:
                precip = 500 + temp * 25  # Moderate to warm = more precipitation
            else:
                precip = 1000 - (temp - 20) * 30  # Very hot = arid
                precip = max(50, precip)  # Minimum precipitation
            
            # Adjust for elevation - rain shadow effect
            if elev > 0:
                # Higher elevation gets less precipitation after a point
                if elev > 1000:
                    precip *= 0.7  # Rain shadow
                
            # Add to precipitation data
            self.planet.precipitation[i] += precip
    
    def _apply_terrain_effects(self):
        """Apply local terrain effects to climate data"""
        neighbors = self.planet.grid.get_vertex_neighbors()
        
        # Get coordinates for all vertices
        coords = self.planet.grid.vertices
        
        # Create temporary arrays for adjusted values
        temp_adjusted = np.copy(self.planet.temperature)
        precip_adjusted = np.copy(self.planet.precipitation)
        
        # Apply terrain effects for each vertex
        for i in range(len(self.planet.temperature)):
            # Skip ocean points for some effects
            if self.planet.elevation[i] < 0:
                continue
                
            # Get neighboring vertices
            neighbor_indices = neighbors[i]
            
            # Check for mountain ranges causing rain shadows
            # Simple detection: if this point is high and has higher neighbors
            if self.planet.elevation[i] > 1000:
                higher_neighbors = [n for n in neighbor_indices if self.planet.elevation[n] > self.planet.elevation[i]]
                if higher_neighbors:
                    # This is in a mountain range - check if it's on the leeward side
                    # Simplification: just reduce precipitation
                    precip_adjusted[i] *= 0.6
            
            # River valleys are wetter and moderate temperature
            for river in self.planet.rivers:
                if i in river:
                    # Increase precipitation near rivers
                    precip_adjusted[i] *= 1.2
                    
                    # Moderate temperature (cooler in summer, warmer in winter)
                    if self.planet.temperature[i] > 20:
                        temp_adjusted[i] -= 2  # Cooling effect in hot areas
                    elif self.planet.temperature[i] < 0:
                        temp_adjusted[i] += 2  # Warming effect in cold areas
        
        # Apply adjusted values
        self.planet.temperature = temp_adjusted
        self.planet.precipitation = precip_adjusted