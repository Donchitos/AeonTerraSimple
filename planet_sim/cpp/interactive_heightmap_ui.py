#!/usr/bin/env python3
"""
Interactive Heightmap Generator UI

This application provides a graphical user interface for the AeonTerra heightmap
generator, allowing users to interactively adjust parameters and see the results
in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import sys
import os
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AeonTerra C++ bindings
try:
    from planet_sim.cpp.python import _aeonterracpp
except ImportError:
    print("Failed to import AeonTerra C++ bindings. Make sure they're built properly.")
    print("Try running: cd planet_sim/cpp && ./optimized_config_build.sh")
    sys.exit(1)

class HeightmapGeneratorApp:
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("AeonTerra Heightmap Generator")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set up the config manager
        self.config = _aeonterracpp.optimization.ConfigManager.get_instance()
        self.setup_configuration()
        
        # Variables for terrain generation
        self.planet = None
        self.heightmap_gen = None
        self.erosion = None
        self.generate_thread = None
        self.is_generating = False
        self.heightmap_data = None
        
        # UI variables
        self.parameter_widgets = {}
        
        # Create the UI
        self.create_ui()
        
        # Initialize planet
        self.initialize_planet()
    
    def setup_configuration(self):
        """Set up the initial configuration"""
        # Create heightmap section
        if not self.config.has_section("heightmap"):
            heightmap_section = self.config.create_section("heightmap", "Heightmap Generation Parameters")
            
            # Add parameters
            heightmap_section.add_param("resolution", _aeonterracpp.optimization.ParamValue(1024), 
                                       self.create_metadata("resolution", "Resolution of the heightmap", _aeonterracpp.optimization.ParamType.Int))
            
            heightmap_section.add_param("seed", _aeonterracpp.optimization.ParamValue(42),
                                       self.create_metadata("seed", "Random seed", _aeonterracpp.optimization.ParamType.Int))
            
            heightmap_section.add_param("resolution_level", _aeonterracpp.optimization.ParamValue(6),
                                       self.create_metadata("resolution_level", "Planet resolution level (4-8)", _aeonterracpp.optimization.ParamType.Int))
            
            heightmap_section.add_param("noise_octaves", _aeonterracpp.optimization.ParamValue(6),
                                       self.create_metadata("noise_octaves", "Number of noise octaves", _aeonterracpp.optimization.ParamType.Int))
            
            heightmap_section.add_param("noise_persistence", _aeonterracpp.optimization.ParamValue(0.5),
                                       self.create_metadata("noise_persistence", "Noise persistence", _aeonterracpp.optimization.ParamType.Float))
            
            heightmap_section.add_param("noise_frequency", _aeonterracpp.optimization.ParamValue(1.0),
                                       self.create_metadata("noise_frequency", "Noise frequency", _aeonterracpp.optimization.ParamType.Float))
            
            heightmap_section.add_param("mountain_height", _aeonterracpp.optimization.ParamValue(0.8),
                                       self.create_metadata("mountain_height", "Mountain height factor", _aeonterracpp.optimization.ParamType.Float))
            
            heightmap_section.add_param("ocean_level", _aeonterracpp.optimization.ParamValue(0.0),
                                       self.create_metadata("ocean_level", "Ocean level", _aeonterracpp.optimization.ParamType.Float))
        
        # Create erosion section
        if not self.config.has_section("erosion"):
            erosion_section = self.config.create_section("erosion", "Erosion Parameters")
            
            # Add erosion parameters
            erosion_section.add_param("enabled", _aeonterracpp.optimization.ParamValue(True),
                                     self.create_metadata("enabled", "Enable erosion", _aeonterracpp.optimization.ParamType.Bool))
            
            erosion_section.add_param("erosion_cycles", _aeonterracpp.optimization.ParamValue(100),
                                     self.create_metadata("erosion_cycles", "Number of erosion cycles", _aeonterracpp.optimization.ParamType.Int))
            
            erosion_section.add_param("erosion_strength", _aeonterracpp.optimization.ParamValue(0.15),
                                     self.create_metadata("erosion_strength", "Erosion strength factor", _aeonterracpp.optimization.ParamType.Float))
            
            erosion_section.add_param("deposition_rate", _aeonterracpp.optimization.ParamValue(0.3),
                                     self.create_metadata("deposition_rate", "Deposition rate", _aeonterracpp.optimization.ParamType.Float))
            
            erosion_section.add_param("river_erosion_weight", _aeonterracpp.optimization.ParamValue(1.0),
                                     self.create_metadata("river_erosion_weight", "River erosion weight", _aeonterracpp.optimization.ParamType.Float))
        
        # Create tectonic section
        if not self.config.has_section("tectonics"):
            tectonic_section = self.config.create_section("tectonics", "Tectonic Parameters")
            
            # Add tectonic parameters
            tectonic_section.add_param("enabled", _aeonterracpp.optimization.ParamValue(False),
                                      self.create_metadata("enabled", "Enable tectonic simulation", _aeonterracpp.optimization.ParamType.Bool))
            
            tectonic_section.add_param("num_plates", _aeonterracpp.optimization.ParamValue(8),
                                      self.create_metadata("num_plates", "Number of tectonic plates", _aeonterracpp.optimization.ParamType.Int))
            
            tectonic_section.add_param("simulation_time", _aeonterracpp.optimization.ParamValue(100),
                                      self.create_metadata("simulation_time", "Simulation time (Myr)", _aeonterracpp.optimization.ParamType.Float))
            
            tectonic_section.add_param("plate_speed", _aeonterracpp.optimization.ParamValue(1.0),
                                      self.create_metadata("plate_speed", "Plate movement speed", _aeonterracpp.optimization.ParamType.Float))
        
        # Create performance section
        if not self.config.has_section("performance"):
            perf_section = self.config.create_section("performance", "Performance Parameters")
            
            # Add performance parameters
            perf_section.add_param("use_spatial_acceleration", _aeonterracpp.optimization.ParamValue(True),
                                  self.create_metadata("use_spatial_acceleration", "Use spatial acceleration", _aeonterracpp.optimization.ParamType.Bool))
            
            perf_section.add_param("multithreading", _aeonterracpp.optimization.ParamValue(True),
                                  self.create_metadata("multithreading", "Enable multithreading", _aeonterracpp.optimization.ParamType.Bool))
            
            perf_section.add_param("thread_count", _aeonterracpp.optimization.ParamValue(0),
                                  self.create_metadata("thread_count", "Thread count (0=auto)", _aeonterracpp.optimization.ParamType.Int))
    
    def create_metadata(self, name, description, param_type):
        """Create parameter metadata for the configuration system"""
        metadata = _aeonterracpp.optimization.ParamMetadata()
        metadata.name = name
        metadata.description = description
        metadata.type = param_type
        return metadata
    
    def create_ui(self):
        """Create the user interface"""
        # Create main frame with two panels
        self.main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for parameters
        self.param_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.param_frame, weight=30)
        
        # Right panel for visualization
        self.visual_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.visual_frame, weight=70)
        
        # Create a notebook for parameter categories
        self.param_notebook = ttk.Notebook(self.param_frame)
        self.param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for each parameter section
        self.heightmap_tab = ttk.Frame(self.param_notebook)
        self.erosion_tab = ttk.Frame(self.param_notebook)
        self.tectonic_tab = ttk.Frame(self.param_notebook)
        self.performance_tab = ttk.Frame(self.param_notebook)
        
        self.param_notebook.add(self.heightmap_tab, text="Heightmap")
        self.param_notebook.add(self.erosion_tab, text="Erosion")
        self.param_notebook.add(self.tectonic_tab, text="Tectonics")
        self.param_notebook.add(self.performance_tab, text="Performance")
        
        # Create widgets for each parameter
        self.create_parameter_widgets()
        
        # Create buttons frame
        self.button_frame = ttk.Frame(self.param_frame)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create action buttons
        self.generate_button = ttk.Button(self.button_frame, text="Generate Heightmap", 
                                          command=self.generate_heightmap)
        self.generate_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_button = ttk.Button(self.button_frame, text="Save Heightmap", 
                                      command=self.save_heightmap)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_config_button = ttk.Button(self.button_frame, text="Save Config", 
                                            command=self.save_config)
        self.save_config_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.load_config_button = ttk.Button(self.button_frame, text="Load Config", 
                                            command=self.load_config)
        self.load_config_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create figure for visualization
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.plot = self.figure.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.visual_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_parameter_widgets(self):
        """Create widgets for all parameters in the configuration"""
        # Create heightmap parameter widgets
        self.create_section_widgets(self.heightmap_tab, "heightmap")
        
        # Create erosion parameter widgets
        self.create_section_widgets(self.erosion_tab, "erosion")
        
        # Create tectonic parameter widgets
        self.create_section_widgets(self.tectonic_tab, "tectonics")
        
        # Create performance parameter widgets
        self.create_section_widgets(self.performance_tab, "performance")
    
    def create_section_widgets(self, parent, section_name):
        """Create widgets for a configuration section"""
        if not self.config.has_section(section_name):
            return
        
        section = self.config.get_section(section_name)
        param_keys = section.get_param_keys()
        
        # Create a frame with scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create widgets for each parameter
        for key in param_keys:
            param = section.get_param(key)
            param_type = param.getType()
            
            # Create frame for this parameter
            param_frame = ttk.Frame(scrollable_frame)
            param_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Label with parameter name and tooltip
            label = ttk.Label(param_frame, text=key, width=20)
            label.pack(side=tk.LEFT, padx=5, pady=2)
            
            # Create different widgets based on parameter type
            if param_type == _aeonterracpp.optimization.ParamType.Int:
                value = param.getValue(0)
                var = tk.IntVar(value=value)
                widget = ttk.Spinbox(param_frame, from_=0, to=10000, textvariable=var, width=10)
                widget.pack(side=tk.LEFT, padx=5, pady=2)
                
                # Add slider for common parameters
                if key in ["noise_octaves", "erosion_cycles", "num_plates"]:
                    max_val = 20 if key == "noise_octaves" else 500 if key == "erosion_cycles" else 20
                    slider = ttk.Scale(param_frame, from_=0, to=max_val, orient=tk.HORIZONTAL,
                                     command=lambda val, v=var: v.set(int(float(val))))
                    slider.set(value)
                    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
                    var.trace_add("write", lambda *args, s=slider, v=var: s.set(v.get()))
                
                # Add callback to update configuration when value changes
                var.trace_add("write", lambda *args, s=section_name, k=key, v=var:
                             self.update_config(s, k, _aeonterracpp.optimization.ParamValue(v.get())))
                
                self.parameter_widgets[(section_name, key)] = (var, widget)
                
            elif param_type == _aeonterracpp.optimization.ParamType.Float:
                value = param.getValue(0.0)
                var = tk.DoubleVar(value=value)
                widget = ttk.Spinbox(param_frame, from_=0.0, to=100.0, increment=0.05, 
                                   textvariable=var, width=10)
                widget.pack(side=tk.LEFT, padx=5, pady=2)
                
                # Add slider for visual parameters
                if key in ["noise_persistence", "noise_frequency", "mountain_height", 
                          "erosion_strength", "deposition_rate", "ocean_level"]:
                    max_val = 1.0 if key in ["noise_persistence", "erosion_strength", "deposition_rate"] else 10.0
                    slider = ttk.Scale(param_frame, from_=0.0, to=max_val, orient=tk.HORIZONTAL,
                                     command=lambda val, v=var: v.set(float(val)))
                    slider.set(value)
                    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
                    var.trace_add("write", lambda *args, s=slider, v=var: s.set(v.get()))
                
                # Add callback to update configuration when value changes
                var.trace_add("write", lambda *args, s=section_name, k=key, v=var:
                             self.update_config(s, k, _aeonterracpp.optimization.ParamValue(v.get())))
                
                self.parameter_widgets[(section_name, key)] = (var, widget)
                
            elif param_type == _aeonterracpp.optimization.ParamType.Bool:
                value = param.getValue(False)
                var = tk.BooleanVar(value=value)
                widget = ttk.Checkbutton(param_frame, variable=var)
                widget.pack(side=tk.LEFT, padx=5, pady=2)
                
                # Add callback to update configuration when value changes
                var.trace_add("write", lambda *args, s=section_name, k=key, v=var:
                             self.update_config(s, k, _aeonterracpp.optimization.ParamValue(v.get())))
                
                self.parameter_widgets[(section_name, key)] = (var, widget)
                
            elif param_type == _aeonterracpp.optimization.ParamType.String:
                value = param.getValue("")
                var = tk.StringVar(value=value)
                widget = ttk.Entry(param_frame, textvariable=var, width=20)
                widget.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)
                
                # Add callback to update configuration when value changes
                var.trace_add("write", lambda *args, s=section_name, k=key, v=var:
                             self.update_config(s, k, _aeonterracpp.optimization.ParamValue(v.get())))
                
                self.parameter_widgets[(section_name, key)] = (var, widget)
    
    def update_config(self, section_name, key, value):
        """Update the configuration with a new parameter value"""
        if not self.config.has_section(section_name):
            return
        
        section = self.config.get_section(section_name)
        if section.has_param(key):
            section.set_param(key, value)
    
    def initialize_planet(self):
        """Initialize the planet object with current parameters"""
        try:
            # Get parameter values
            heightmap_section = self.config.get_section("heightmap")
            resolution_level = heightmap_section.get_param("resolution_level").getValue(6)
            
            # Create planet
            self.planet = _aeonterracpp.Planet(6371.0, resolution_level, "InteractivePlanet")
            
            # Set status
            self.status_var.set(f"Planet initialized with resolution level {resolution_level} " +
                               f"({self.planet.get_vertex_count()} vertices)")
            
            # Create initial heightmap generator
            self.heightmap_gen = _aeonterracpp.HeightmapGenerator(self.planet)
            
            # Generate initial heightmap
            self.generate_heightmap()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize planet: {str(e)}")
            print(f"Error: {str(e)}")
    
    def generate_heightmap(self):
        """Generate heightmap with current parameters"""
        if self.is_generating:
            return
        
        # Disable generate button during generation
        self.generate_button.configure(state="disabled")
        self.status_var.set("Generating heightmap...")
        self.is_generating = True
        
        # Start generation in a separate thread
        self.generate_thread = threading.Thread(target=self._generate_heightmap_thread)
        self.generate_thread.daemon = True
        self.generate_thread.start()
    
    def _generate_heightmap_thread(self):
        """Thread function for heightmap generation"""
        try:
            # Get parameter values
            heightmap_section = self.config.get_section("heightmap")
            erosion_section = self.config.get_section("erosion")
            tectonic_section = self.config.get_section("tectonics")
            perf_section = self.config.get_section("performance")
            
            # Get heightmap parameters
            seed = heightmap_section.get_param("seed").getValue(42)
            noise_octaves = heightmap_section.get_param("noise_octaves").getValue(6)
            noise_persistence = heightmap_section.get_param("noise_persistence").getValue(0.5)
            noise_frequency = heightmap_section.get_param("noise_frequency").getValue(1.0)
            mountain_height = heightmap_section.get_param("mountain_height").getValue(0.8)
            ocean_level = heightmap_section.get_param("ocean_level").getValue(0.0)
            
            # Get performance parameters
            use_accel = perf_section.get_param("use_spatial_acceleration").getValue(True)
            multithreading = perf_section.get_param("multithreading").getValue(True)
            thread_count = perf_section.get_param("thread_count").getValue(0)
            
            # Configure heightmap generator
            self.heightmap_gen.set_random_seed(seed)
            self.heightmap_gen.set_noise_octaves(noise_octaves)
            self.heightmap_gen.set_noise_persistence(noise_persistence)
            self.heightmap_gen.set_noise_frequency(noise_frequency)
            self.heightmap_gen.set_mountain_height(mountain_height)
            
            # Use spatial acceleration if enabled
            if use_accel:
                accel_params = _aeonterracpp.optimization.SpatialAccelerationParams()
                accel_params.type = _aeonterracpp.optimization.SpatialAccelerationType.Octree
                accel_params.max_depth = 6
                accel_params.grid_resolution = 1.0
                accel_params.parallel_construction = multithreading
                
                # Create spatial acceleration
                acceleration = _aeonterracpp.optimization.SpatialAcceleration.create(
                    _aeonterracpp.optimization.SpatialAccelerationType.Octree,
                    self.planet, accel_params)
                
                # Set it in the heightmap generator
                self.heightmap_gen.use_spatial_acceleration(acceleration)
            
            # Set up parallel processing
            if multithreading:
                parallel_config = _aeonterracpp.optimization.ParallelConfig()
                parallel_config.num_threads = thread_count
                parallel_config.enable_dynamic_load = True
                
                # Enable parallel processing in the heightmap generator
                self.heightmap_gen.enable_parallel_processing(parallel_config)
            
            # Generate basic heightmap
            self.status_var.set("Generating base heightmap...")
            self.heightmap_gen.generate_heightmap()
            
            # Apply tectonics if enabled
            if tectonic_section.get_param("enabled").getValue(False):
                self.status_var.set("Applying tectonic simulation...")
                num_plates = tectonic_section.get_param("num_plates").getValue(8)
                sim_time = tectonic_section.get_param("simulation_time").getValue(100.0)
                plate_speed = tectonic_section.get_param("plate_speed").getValue(1.0)
                
                # Create tectonic simulation
                tectonics = _aeonterracpp.EnhancedTectonicSimulation(self.planet, num_plates, seed)
                tectonics.set_plate_movement_speed(plate_speed)
                
                # Simulate tectonics
                step_size = 20.0
                steps = int(sim_time / step_size)
                for i in range(steps):
                    self.status_var.set(f"Tectonic simulation: {i*step_size}-{(i+1)*step_size} Myr...")
                    tectonics.simulate_step(step_size)
            
            # Apply erosion if enabled
            if erosion_section.get_param("enabled").getValue(True):
                self.status_var.set("Applying erosion...")
                erosion_cycles = erosion_section.get_param("erosion_cycles").getValue(100)
                erosion_strength = erosion_section.get_param("erosion_strength").getValue(0.15)
                deposition_rate = erosion_section.get_param("deposition_rate").getValue(0.3)
                river_weight = erosion_section.get_param("river_erosion_weight").getValue(1.0)
                
                # Setup erosion
                self.erosion = _aeonterracpp.EnhancedErosion(self.planet)
                self.erosion.set_erosion_strength(erosion_strength)
                self.erosion.set_deposition_rate(deposition_rate)
                self.erosion.set_river_erosion_weight(river_weight)
                
                # Use parallel processing if enabled
                if multithreading:
                    parallel_config = _aeonterracpp.optimization.ParallelConfig()
                    parallel_config.num_threads = thread_count
                    parallel_config.enable_dynamic_load = True
                    self.erosion.enable_parallel_processing(parallel_config)
                
                # Apply erosion
                self.erosion.apply_erosion(erosion_cycles)
            
            # Set ocean level
            if ocean_level != 0.0:
                # Adjust elevations to set ocean level
                for i in range(self.planet.get_vertex_count()):
                    elev = self.planet.get_elevation(i)
                    if elev < ocean_level:
                        self.planet.set_elevation(i, ocean_level)
            
            # Update the visualization
            self.update_visualization()
            
            # Set status to complete
            self.status_var.set("Heightmap generation complete.")
            
        except Exception as e:
            messagebox.showerror("Generation Error", f"Failed to generate heightmap: {str(e)}")
            print(f"Error: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
        
        finally:
            # Re-enable generate button
            self.root.after(0, lambda: self.generate_button.configure(state="normal"))
            self.is_generating = False
    
    def update_visualization(self):
        """Update the visualization with the current heightmap"""
        try:
            # Clear the plot
            self.plot.clear()
            
            # Create a grid for sampling
            resolution = 500
            lat = np.linspace(-90, 90, resolution)
            lon = np.linspace(-180, 180, resolution)
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            
            # Sample heightmap
            heightmap = np.zeros((resolution, resolution))
            
            for i in range(resolution):
                for j in range(resolution):
                    # Convert lat/lon to 3D position
                    lat_rad = np.radians(lat_grid[i, j])
                    lon_rad = np.radians(lon_grid[i, j])
                    
                    # Create position vector
                    x = np.cos(lat_rad) * np.cos(lon_rad)
                    y = np.cos(lat_rad) * np.sin(lon_rad)
                    z = np.sin(lat_rad)
                    
                    # Create normalized position vector
                    pos = _aeonterracpp.Vec3f(x, y, z)
                    
                    # Scale to planet radius
                    pos = pos * self.planet.get_radius()
                    
                    # Get elevation at position
                    elevation = self.planet.get_elevation_at_position(pos)
                    heightmap[i, j] = elevation
            
            # Store heightmap data
            self.heightmap_data = heightmap
            
            # Create custom colormap that transitions from blue to green to brown to white
            cmap = plt.cm.terrain
            
            # Plot heightmap
            im = self.plot.imshow(heightmap, cmap=cmap, extent=[-180, 180, -90, 90])
            self.plot.set_title("Planet Heightmap")
            self.plot.set_xlabel("Longitude")
            self.plot.set_ylabel("Latitude")
            
            # Add colorbar
            cbar = self.figure.colorbar(im, ax=self.plot)
            cbar.set_label("Elevation (km)")
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to update visualization: {str(e)}")
            print(f"Error: {str(e)}")
    
    def save_heightmap(self):
        """Save heightmap to a file"""
        if self.heightmap_data is None:
            messagebox.showwarning("Warning", "No heightmap to save.")
            return
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Save as image
            plt.figure(figsize=(12, 8))
            plt.imshow(self.heightmap_data, cmap=plt.cm.terrain, extent=[-180, 180, -90, 90])
            plt.colorbar(label="Elevation (km)")
            plt.title("Planet Heightmap")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(file_path, dpi=300)
            plt.close()
            
            # Also save as numpy array for later use
            np_path = file_path.replace(".png", ".npy")
            np.save(np_path, self.heightmap_data)
            
            messagebox.showinfo("Success", f"Heightmap saved to {file_path} and {np_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save heightmap: {str(e)}")
            print(f"Error: {str(e)}")
    
    def save_config(self):
        """Save current configuration to a file"""
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Save configuration
            self.config.save_to_file(file_path)
            messagebox.showinfo("Success", f"Configuration saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration: {str(e)}")
            print(f"Error: {str(e)}")
    
    def load_config(self):
        """Load configuration from a file"""
        # Ask for file location
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load configuration
            if self.config.load_from_file(file_path):
                messagebox.showinfo("Success", f"Configuration loaded from {file_path}")
                
                # Update UI widgets with new values
                self.update_widgets_from_config()
                
                # Reinitialize planet if resolution level changed
                heightmap_section = self.config.get_section("heightmap")
                resolution_level = heightmap_section.get_param("resolution_level").getValue(6)
                
                if self.planet and self.planet.get_resolution_level() != resolution_level:
                    self.initialize_planet()
                else:
                    # Generate new heightmap with loaded config
                    self.generate_heightmap()
            else:
                messagebox.showwarning("Warning", "Failed to load configuration.")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration: {str(e)}")
            print(f"Error: {str(e)}")
    
    def update_widgets_from_config(self):
        """Update UI widgets with values from the configuration"""
        for section_name in self.config.get_section_names():
            section = self.config.get_section(section_name)
            param_keys = section.get_param_keys()
            
            for key in param_keys:
                if (section_name, key) in self.parameter_widgets:
                    param = section.get_param(key)
                    var, widget = self.parameter_widgets[(section_name, key)]
                    
                    # Update widget value based on parameter type
                    param_type = param.getType()
                    
                    if param_type == _aeonterracpp.optimization.ParamType.Int:
                        var.set(param.getValue(0))
                    elif param_type == _aeonterracpp.optimization.ParamType.Float:
                        var.set(param.getValue(0.0))
                    elif param_type == _aeonterracpp.optimization.ParamType.Bool:
                        var.set(param.getValue(False))
                    elif param_type == _aeonterracpp.optimization.ParamType.String:
                        var.set(param.getValue(""))

def main():
    """Main function"""
    # Create the root window
    root = tk.Tk()
    
    # Create application
    app = HeightmapGeneratorApp(root)
    
    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()