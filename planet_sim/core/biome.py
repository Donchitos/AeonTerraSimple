import numpy as np

class BiomeClassifier:
    """
    Classifies regions of the planet into biomes based on climate data.
    """
    
    def __init__(self, planet):
        """
        Initialize the biome classifier.
        
        Parameters:
        - planet: A Planet object with temperature and precipitation data
        """
        self.planet = planet
        
        # Initialize biome data if not already present
        if self.planet.biome_ids is None:
            self.planet.biome_ids = np.zeros(len(self.planet.grid.vertices), dtype=int)
        
        # Define biome types (Whittaker classification)
        self.biome_types = [
            "Ocean",             # 0
            "Ice Sheet",         # 1
            "Tundra",            # 2
            "Taiga/Boreal",      # 3
            "Temperate Forest",  # 4
            "Temperate Rainforest", # 5
            "Temperate Grassland", # 6
            "Desert",            # 7
            "Savanna",           # 8
            "Tropical Seasonal Forest", # 9
            "Tropical Rainforest", # 10
            "Alpine/Mountain"     # 11
        ]
    
    def classify(self):
        """
        Classify each vertex into a biome type based on climate and elevation.
        """
        print("Classifying biomes...")
        
        # Ensure climate data exists
        if self.planet.temperature is None or self.planet.precipitation is None:
            raise ValueError("Climate data (temperature and precipitation) must be generated before classifying biomes")
        
        # Classify each vertex
        for i in range(len(self.planet.biome_ids)):
            elevation = self.planet.elevation[i]
            temperature = self.planet.temperature[i]
            precipitation = self.planet.precipitation[i]
            
            # Determine biome based on climate and elevation
            if elevation < 0:
                # Ocean
                self.planet.biome_ids[i] = 0
            elif elevation > 3000:
                # High mountains
                self.planet.biome_ids[i] = 11
            elif temperature < -10:
                if precipitation < 250:
                    # Ice sheet
                    self.planet.biome_ids[i] = 1
                else:
                    # Tundra
                    self.planet.biome_ids[i] = 2
            elif temperature < 5:
                # Taiga/Boreal forest
                self.planet.biome_ids[i] = 3
            elif temperature < 15:
                if precipitation < 500:
                    # Temperate grassland
                    self.planet.biome_ids[i] = 6
                elif precipitation < 2000:
                    # Temperate forest
                    self.planet.biome_ids[i] = 4
                else:
                    # Temperate rainforest
                    self.planet.biome_ids[i] = 5
            elif temperature < 20:
                if precipitation < 250:
                    # Desert
                    self.planet.biome_ids[i] = 7
                elif precipitation < 1000:
                    # Savanna
                    self.planet.biome_ids[i] = 8
                else:
                    # Tropical seasonal forest
                    self.planet.biome_ids[i] = 9
            else:
                if precipitation < 500:
                    # Desert
                    self.planet.biome_ids[i] = 7
                elif precipitation < 1500:
                    # Savanna
                    self.planet.biome_ids[i] = 8
                else:
                    # Tropical rainforest
                    self.planet.biome_ids[i] = 10
        
        print(f"Classified into {len(np.unique(self.planet.biome_ids))} biome types")
        return self
    
    def get_biome_name(self, biome_id):
        """Get the name of a biome by ID"""
        if 0 <= biome_id < len(self.biome_types):
            return self.biome_types[biome_id]
        return "Unknown"
    
    def get_biome_color(self, biome_id):
        """Get a color (RGB tuple) representing a biome"""
        # Define colors for each biome type
        colors = [
            (0.0, 0.2, 0.8),    # Ocean - blue
            (1.0, 1.0, 1.0),    # Ice Sheet - white
            (0.8, 0.8, 0.7),    # Tundra - light gray/tan
            (0.2, 0.4, 0.4),    # Taiga/Boreal - dark green/blue
            (0.1, 0.6, 0.3),    # Temperate Forest - green
            (0.0, 0.5, 0.3),    # Temperate Rainforest - deep green
            (0.8, 0.8, 0.4),    # Temperate Grassland - tan
            (0.9, 0.8, 0.2),    # Desert - yellow/tan
            (0.8, 0.7, 0.2),    # Savanna - light brown
            (0.2, 0.7, 0.2),    # Tropical Seasonal Forest - bright green
            (0.0, 0.4, 0.0),    # Tropical Rainforest - dark green
            (0.6, 0.6, 0.6)     # Alpine/Mountain - gray
        ]
        
        if 0 <= biome_id < len(colors):
            return colors[biome_id]
        return (0.5, 0.5, 0.5)  # Default gray