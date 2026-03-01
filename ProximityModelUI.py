"""
Proximity Model Configuration UI

A graphical interface for configuring proximity model analysis parameters.
This UI provides a layer on top of the text-based configuration in ProximityModel.py,
allowing users to configure analysis settings through a visual interface.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
import json
import subprocess
import sys
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd

# Set appearance and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# Column mapping definitions for each data type
# Only data attributes are mapped - geometry is auto-detected
COLUMN_MAPPINGS = {
    'street_centerlines': {
        'public_data_id': 'Unique ID for street segment',
        'name': 'Street name',
        'highway': 'Highway type (e.g., residential, primary)',
        'maxspeed': 'Maximum speed limit',
        'oneway': 'One-way indicator (yes/no)',
        'lanes': 'Number of lanes',
        'lane_width': 'Width of each lane',
        'surface': 'Surface type (e.g., asphalt, concrete)',
    },
    'sidewalks': {
        'public_data_id': 'Unique ID for sidewalk segment',
        'surface': 'Surface type',
        'width': 'Sidewalk width',
        'incline': 'Incline/slope',
    },
    'bikelanes': {
        'public_data_id': 'Unique ID for bikeway segment',
        'type': 'Bikeway type (lane/track/shared_lane/etc.)',
        'surface': 'Surface type',
        'permitted': 'Bicycle permission (yes/no/designated/etc.)',
        'width': 'Bikeway width',
        'incline': 'Incline/slope',
    },
    'curb_ramps': {
        'public_data_id': 'Unique ID for curb ramp (e.g., LocID)',
        'returnloc': 'Direction of curb return (NW, N, NE, E, SE, S, SW, W)',
        'returnposition': 'Position on return (Left, Center, Right)',
        'condition_score': 'Condition rating/score',
    },
    'crosswalks': {
        'public_data_id': 'Unique ID for crosswalk',
        'type': 'Crosswalk type (marked/unmarked)',
        'controlled': 'Traffic control (signals/uncontrolled)',
        'marked': 'Marked or unmarked (yes/no)',
        'markings': 'Marking type/pattern',
        'signals': 'Traffic signals - comma-separated list: signal(yes/no), button(yes/no), sound(yes/no), vibration(yes/no), flashing_lights(yes/button/sensor)',
        'island': 'Crossing island present (yes/no)',
        'kerb': 'Kerb type',
        'tactile_paving': 'Tactile paving (yes/no)',
        'traffic_calming': 'Traffic calming (table/etc.)',
        'continuous': 'Continuous crossing (yes/no)',
        'condition': 'Condition rating',
    },
    'street_features': {
        'public_data_id': 'Unique ID for feature',
        'feature_type': 'Type of feature (hydrant, light, bench, etc.)',
    },
    'sidewalk_features': {
        'public_data_id': 'Unique ID for feature',
        'feature_type': 'Type of feature (bench, tree, trash can, etc.)',
    },
    'bikeway_features': {
        'public_data_id': 'Unique ID for feature',
        'feature_type': 'Type of feature (bike rack, repair station, etc.)',
    },
}


# Parameter help text dictionary
HELP_TEXT = {
    # City-specific parameters
    'name': (
        "City Name\n\n"
        "Display name for the city being analyzed.\n\n"
        "Examples:\n"
        "• San Francisco County, CA\n"
        "• Boston, MA\n"
        "• New York City, NY\n\n"
        "This name will be used in output file names and visualizations."
    ),
    'parcels': (
        "Parcels File\n\n"
        "Property parcel geometries for the city.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: Polygon or Point geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "The parcels represent individual properties or land units for which "
        "accessibility will be calculated."
    ),
    'street_centerlines': (
        "Street Centerlines File\n\n"
        "Street network centerlines from government data.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: LineString geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "Optional: If provided, will be merged with OSM street data."
    ),
    'intersection_nodes': (
        "Intersection Nodes File (Optional)\n\n"
        "Government-provided intersection node points.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: Point geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "If provided, these nodes will be used to set the grid for block detection."
    ),
    'sidewalks': (
        "Sidewalks File (Optional)\n\n"
        "Government sidewalk data.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: LineString geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "If provided, will be merged with OSM sidewalk data."
    ),
    'bikelanes': (
        "Bikelanes File (Optional)\n\n"
        "Government bikelane data.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: LineString geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "If provided, will be merged with OSM bikelane data."
    ),
    'curb_ramps': (
        "Curb Ramps File (Optional)\n\n"
        "Government curb ramp point data.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: Point geometries\n"
        "• CSV: With Latitude/Longitude columns (auto-detected)\n"
        "• CSV: With WKT/WKB geometry column\n\n"
        "Column names like 'Latitude', 'Longitude', 'lat', 'lon', 'x', 'y', \n"
        "'xLoc', 'yLoc' are automatically detected.\n\n"
        "If provided, will be used to populate curb ramp locations."
    ),
    'crosswalks': (
        "Crosswalks File (Optional)\n\n"
        "Government crosswalk data.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: Point or LineString geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "If provided, will take precedence over OSM crosswalk data."
    ),
    'street_features': (
        "Street Features File (Optional)\n\n"
        "Street furniture and features (fire hydrants, street lights, benches, etc.).\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: Point geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "Features will be spatially joined to street segments."
    ),
    'sidewalk_features': (
        "Sidewalk Features File (Optional)\n\n"
        "Sidewalk furniture and features (benches, trash cans, trees, etc.).\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: Point geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "Features will be spatially joined to sidewalk segments."
    ),
    'bikeway_features': (
        "Bikeway Features File (Optional)\n\n"
        "Bikeway furniture and features (bike parking, repair stations, etc.).\n\n"
        "Accepted formats:\n"
        "• GeoJSON/Shapefile: Point geometries\n"
        "• CSV: With lat/lon columns or WKT/WKB geometry\n\n"
        "Features will be spatially joined to bikeway segments."
    ),
    'curbramp_trustworthy': (
        "Curb Ramp Trustworthy\n\n"
        "Whether to trust government curb ramp data.\n\n"
        "When enabled:\n"
        "• Government curb ramps will be used to replace default locations\n"
        "• Curb ramp geometry will be adjusted based on government data\n\n"
        "When disabled:\n"
        "• Government curb ramps will be recorded but not used for geometry"
    ),
    'output_dir': (
        "Output Directory\n\n"
        "Directory where all analysis outputs will be saved.\n\n"
        "Structure created automatically:\n"
        "  Output/\n"
        "  ├── {CityName}_network.parquet\n"
        "  └── {CityName}_sanity.parquet\n\n"
        "Example: Output\n"
        "  Creates: Output/San_Francisco_County,_CA_network.parquet"
    ),
}


class ToolTip:
    """Simple, reliable tooltip using tkinter Toplevel."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.hide_job = None
        
        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)
    
    def on_enter(self, event):
        if self.hide_job:
            self.widget.after_cancel(self.hide_job)
            self.hide_job = None
        
        if self.tooltip:
            return
            
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 30
        
        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        self.tooltip.attributes('-topmost', True)
        
        label = tk.Label(
            self.tooltip,
            text=self.text,
            justify="left",
            background="#1a1a1a",
            foreground="#ffffff",
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Inter", 9),
            wraplength=400
        )
        label.pack()
    
    def on_leave(self, event):
        if self.hide_job:
            self.widget.after_cancel(self.hide_job)
        self.hide_job = self.widget.after(200, self.destroy_tooltip)
    
    def destroy_tooltip(self):
        if self.tooltip:
            try:
                self.tooltip.destroy()
            except:
                pass
            self.tooltip = None
        self.hide_job = None


def create_info_button(parent, help_key):
    """Create an info button with tooltip for a parameter."""
    info_btn = ctk.CTkLabel(
        parent,
        text="ⓘ",
        font=("Inter", 14, "bold"),
        text_color=("#666666", "#999999"),
        cursor="hand2",
        width=20
    )
    
    if help_key in HELP_TEXT:
        ToolTip(info_btn, HELP_TEXT[help_key])
    
    return info_btn


class ProximityModelConfigUI:
    """Main configuration UI for proximity model analysis."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Proximity Model Configuration")
        
        self.root.update_idletasks()
        
        try:
            self.root.state('zoomed')
            self.root.update()
        except:
            try:
                self.root.attributes('-zoomed', True)
                self.root.update()
            except:
                self.root.geometry("1600x900")
        
        # Configuration data
        self.cities = []
        self.current_city = None
        self.city_configs = {}
        self.global_settings = {}
        self.column_config_buttons = {}  # Track column config buttons for show/hide
        
        # Default global settings
        self.default_global_settings = {
            "output_dir": "Notebooks/Karna/Proximity Model/Output",
            "default_max_speed": 25,
            "curb_ramp_trustworthiness_outer_buffer": 15.0,
            "curb_ramp_trustworthiness_inner_buffer": 5.0,
            "use_gpu": True,
            "python_env": sys.executable,  # Default to the Python running the UI
        }
        
        # Default city config template
        self.default_city_config = {
            'name': '',
            'government_data_paths': {
                'parcels': '',
                'street_centerlines': '',
                'intersection_nodes': '',
                'sidewalks': '',
                'bikelanes': '',
                'curb_ramps': '',
                'crosswalks': '',
                'street_features': '',
                'sidewalk_features': '',
                'bikeway_features': '',
            },
            'column_mappings': {},
            'curbramp_trustworthy': False,
        }
        
        self.show_city_selection()
    
    def show_city_selection(self):
        """Show initial dialog to select cities for analysis."""
        for widget in self.root.winfo_children():
            widget.destroy()
        
        try:
            self.root.state('zoomed')
        except:
            pass
        
        frame = ctk.CTkFrame(self.root, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        title_label = ctk.CTkLabel(
            frame, 
            text="PROXIMITY MODEL CONFIGURATION", 
            font=("Inter", 24, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ctk.CTkLabel(
            frame,
            text="Select Cities for Analysis",
            font=("Inter", 16)
        )
        subtitle_label.pack(pady=(0, 20))
        
        instruction_label = ctk.CTkLabel(
            frame,
            text="Enter city names (one per line):",
            font=("Inter", 12, "bold")
        )
        instruction_label.pack(pady=5)
        
        self.city_text = ctk.CTkTextbox(
            frame, 
            height=300, 
            width=400,
            font=("Inter", 12),
            corner_radius=10
        )
        self.city_text.pack(pady=10)
        self.city_text.insert("1.0", "San Francisco County, CA\nAlameda County, CA")
        
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=20)
        
        continue_btn = ctk.CTkButton(
            button_frame,
            text="CONTINUE",
            command=self.process_cities,
            font=("Inter", 13, "bold"),
            height=45,
            width=150,
            corner_radius=10
        )
        continue_btn.pack(side="left", padx=5)
        
        load_btn = ctk.CTkButton(
            button_frame,
            text="LOAD EXISTING CONFIG",
            command=self.load_existing_config,
            font=("Inter", 13, "bold"),
            height=45,
            width=200,
            corner_radius=10,
            fg_color="transparent",
            border_width=2
        )
        load_btn.pack(side="left", padx=5)
    
    def process_cities(self):
        """Process the entered city names and show main config UI."""
        city_text = self.city_text.get("1.0", "end").strip()
        self.cities = [c.strip() for c in city_text.split("\n") if c.strip()]
        
        if not self.cities:
            messagebox.showerror("Error", "Please enter at least one city name.")
            return
        
        for city in self.cities:
            if city not in self.city_configs:
                config = self.default_city_config.copy()
                config['name'] = city
                config['government_data_paths'] = self.default_city_config['government_data_paths'].copy()
                self.city_configs[city] = config
        
        self.global_settings = self.default_global_settings.copy()
        
        self.show_main_config()
    
    def load_existing_config(self):
        """Load configuration from ProximityModel.py file."""
        try:
            # Get the ProximityModel.py file path
            script_dir = Path(__file__).parent
            proximity_model_file = script_dir / "ProximityModel.py"
            
            if not proximity_model_file.exists():
                messagebox.showerror("Error", f"ProximityModel.py not found at {proximity_model_file}")
                return
            
            # Read the file and extract config
            with open(proximity_model_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract GLOBAL_CONFIG definition using regex
            import re
            
            # Extract global settings
            output_dir_match = re.search(r'output_dir\s*=\s*["\']([^"\']+)["\']', content)
            max_speed_match = re.search(r'default_max_speed\s*=\s*(\d+)', content)
            outer_buffer_match = re.search(r'curb_ramp_trustworthiness_outer_buffer\s*=\s*([\d.]+)', content)
            inner_buffer_match = re.search(r'curb_ramp_trustworthiness_inner_buffer\s*=\s*([\d.]+)', content)
            use_gpu_match = re.search(r'use_gpu\s*=\s*(True|False)', content)
            python_env_match = re.search(r'python_env\s*=\s*["\']([^"\']+)["\']', content)
            
            self.global_settings = {
                "output_dir": output_dir_match.group(1) if output_dir_match else "Output",
                "default_max_speed": int(max_speed_match.group(1)) if max_speed_match else 25,
                "curb_ramp_trustworthiness_outer_buffer": float(outer_buffer_match.group(1)) if outer_buffer_match else 15.0,
                "curb_ramp_trustworthiness_inner_buffer": float(inner_buffer_match.group(1)) if inner_buffer_match else 5.0,
                "use_gpu": use_gpu_match.group(1) == 'True' if use_gpu_match else True,
                "python_env": python_env_match.group(1) if python_env_match else sys.executable,
            }
            
            # Extract city configs
            # Find all CityConfig instances by manually tracking parentheses
            city_matches = []
            start_pattern = r'CityConfig\s*\('
            
            for match in re.finditer(start_pattern, content):
                start_pos = match.end()
                paren_count = 1
                pos = start_pos
                
                # Find the matching closing parenthesis
                while pos < len(content) and paren_count > 0:
                    if content[pos] == '(':
                        paren_count += 1
                    elif content[pos] == ')':
                        paren_count -= 1
                    pos += 1
                
                if paren_count == 0:
                    # Extract the content between the parentheses
                    city_content = content[start_pos:pos-1]
                    city_matches.append(city_content)
            
            self.cities = []
            self.city_configs = {}
            
            for city_match in city_matches:
                # Extract city name
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', city_match)
                if not name_match:
                    continue
                
                city_name = name_match.group(1)
                self.cities.append(city_name)
                
                # Extract government data paths
                gov_paths = {}
                for field in ['parcels', 'street_centerlines', 'intersection_nodes', 'sidewalks', 
                             'bikelanes', 'curb_ramps', 'crosswalks', 'street_features', 
                             'sidewalk_features', 'bikeway_features']:
                    path_match = re.search(rf'{field}\s*=\s*["\']([^"\']+)["\']', city_match)
                    gov_paths[field] = path_match.group(1) if path_match else ''
                
                # Extract column mappings
                col_mappings = {}
                
                # Extract curb_ramps mappings
                curbramp_mappings = {}
                if re.search(r'curbramp_id\s*=\s*["\']([^"\']+)["\']', city_match):
                    curbramp_mappings['public_data_id'] = re.search(r'curbramp_id\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'curbramp_return_loc\s*=\s*["\']([^"\']+)["\']', city_match):
                    curbramp_mappings['returnloc'] = re.search(r'curbramp_return_loc\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'curbramp_position\s*=\s*["\']([^"\']+)["\']', city_match):
                    curbramp_mappings['returnposition'] = re.search(r'curbramp_position\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'curbramp_condition\s*=\s*["\']([^"\']+)["\']', city_match):
                    curbramp_mappings['condition_score'] = re.search(r'curbramp_condition\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if curbramp_mappings:
                    col_mappings['curb_ramps'] = curbramp_mappings
                
                # Extract street_centerlines mappings
                street_mappings = {}
                if re.search(r'street_id\s*=\s*["\']([^"\']+)["\']', city_match):
                    street_mappings['public_data_id'] = re.search(r'street_id\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'street_name\s*=\s*["\']([^"\']+)["\']', city_match):
                    street_mappings['name'] = re.search(r'street_name\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'street_highway\s*=\s*["\']([^"\']+)["\']', city_match):
                    street_mappings['highway'] = re.search(r'street_highway\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'street_maxspeed\s*=\s*["\']([^"\']+)["\']', city_match):
                    street_mappings['maxspeed'] = re.search(r'street_maxspeed\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'street_lanes\s*=\s*["\']([^"\']+)["\']', city_match):
                    street_mappings['lanes'] = re.search(r'street_lanes\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'street_surface\s*=\s*["\']([^"\']+)["\']', city_match):
                    street_mappings['surface'] = re.search(r'street_surface\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if street_mappings:
                    col_mappings['street_centerlines'] = street_mappings
                
                # Extract sidewalks mappings
                sidewalk_mappings = {}
                if re.search(r'sidewalk_id\s*=\s*["\']([^"\']+)["\']', city_match):
                    sidewalk_mappings['public_data_id'] = re.search(r'sidewalk_id\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'sidewalk_surface\s*=\s*["\']([^"\']+)["\']', city_match):
                    sidewalk_mappings['surface'] = re.search(r'sidewalk_surface\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'sidewalk_width\s*=\s*["\']([^"\']+)["\']', city_match):
                    sidewalk_mappings['width'] = re.search(r'sidewalk_width\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'sidewalk_incline\s*=\s*["\']([^"\']+)["\']', city_match):
                    sidewalk_mappings['incline'] = re.search(r'sidewalk_incline\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if sidewalk_mappings:
                    col_mappings['sidewalks'] = sidewalk_mappings
                
                # Extract bikelanes mappings
                bikelane_mappings = {}
                if re.search(r'bikelane_id\s*=\s*["\']([^"\']+)["\']', city_match):
                    bikelane_mappings['public_data_id'] = re.search(r'bikelane_id\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'bikelane_type\s*=\s*["\']([^"\']+)["\']', city_match):
                    bikelane_mappings['type'] = re.search(r'bikelane_type\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'bikelane_surface\s*=\s*["\']([^"\']+)["\']', city_match):
                    bikelane_mappings['surface'] = re.search(r'bikelane_surface\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'bikelane_width\s*=\s*["\']([^"\']+)["\']', city_match):
                    bikelane_mappings['width'] = re.search(r'bikelane_width\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if bikelane_mappings:
                    col_mappings['bikelanes'] = bikelane_mappings
                
                # Extract feature mappings (shared across street/sidewalk/bikeway features)
                feature_mappings = {}
                if re.search(r'feature_id\s*=\s*["\']([^"\']+)["\']', city_match):
                    feature_mappings['public_data_id'] = re.search(r'feature_id\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if re.search(r'feature_type\s*=\s*["\']([^"\']+)["\']', city_match):
                    feature_mappings['feature_type'] = re.search(r'feature_type\s*=\s*["\']([^"\']+)["\']', city_match).group(1)
                if feature_mappings:
                    # Apply to all feature types that have files specified
                    if gov_paths.get('street_features'):
                        col_mappings['street_features'] = feature_mappings.copy()
                    if gov_paths.get('sidewalk_features'):
                        col_mappings['sidewalk_features'] = feature_mappings.copy()
                    if gov_paths.get('bikeway_features'):
                        col_mappings['bikeway_features'] = feature_mappings.copy()
                
                # Extract curbramp_trustworthy
                trust_match = re.search(r'curbramp_trustworthy\s*=\s*(True|False)', city_match)
                trustworthy = trust_match.group(1) == 'True' if trust_match else False
                
                self.city_configs[city_name] = {
                    'name': city_name,
                    'government_data_paths': gov_paths,
                    'column_mappings': col_mappings,
                    'curbramp_trustworthy': trustworthy,
                }
            
            if not self.cities:
                messagebox.showwarning("Warning", "No cities found in ProximityModel.py. Starting with defaults.")
                self.process_cities()
                return
            
            # Successfully loaded
            self.show_main_config()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config from ProximityModel.py: {e}")
            import traceback
            traceback.print_exc()
    
    def show_main_config(self):
        """Show main configuration interface with sidebar and config panels."""
        for widget in self.root.winfo_children():
            widget.destroy()
        
        try:
            self.root.state('zoomed')
        except:
            pass
        
        # Sidebar for city list
        sidebar = ctk.CTkFrame(self.root, width=200)
        sidebar.pack(side="left", fill="y", padx=5, pady=5)
        sidebar.pack_propagate(False)
        
        ctk.CTkLabel(sidebar, text="Cities", font=("Inter", 16, "bold")).pack(pady=10)
        
        city_buttons_frame = ctk.CTkScrollableFrame(sidebar)
        city_buttons_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.city_buttons = {}
        for city in self.cities:
            # Create a frame for each city button + remove button
            city_frame = ctk.CTkFrame(city_buttons_frame, fg_color="transparent", height=35)
            city_frame.pack(fill="x", pady=2)
            city_frame.pack_propagate(True)
            
            # City selection button
            btn = ctk.CTkButton(
                city_frame,
                text=city,
                command=lambda c=city: self.select_city(c),
                height=35,
                corner_radius=8
            )
            btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
            
            # Remove button (×)
            remove_btn = ctk.CTkButton(
                city_frame,
                text="×",
                command=lambda c=city: self.remove_city(c),
                width=35,
                height=35,
                corner_radius=8,
                fg_color=("gray70", "gray30"),
                hover_color=("red", "darkred"),
                font=("Inter", 18, "bold")
            )
            remove_btn.pack(side="right", padx=0)
            
            self.city_buttons[city] = btn
        
        add_city_btn = ctk.CTkButton(
            city_buttons_frame,
            text="+ Add City",
            command=self.add_new_city,
            height=35,
            corner_radius=8,
            fg_color="transparent",
            border_width=2,
            hover_color=("#3a7ebf", "#1f538d")
        )
        add_city_btn.pack(fill="x", pady=10)
        
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        
        ctk.CTkButton(btn_frame, text="Global Settings", 
                  command=self.show_global_settings, height=32).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="Save Config", 
                  command=self.save_config, height=32).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="Run Analysis", 
                  command=self.run_analysis, height=32).pack(fill="x", pady=2)
        
        self.config_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.config_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        if self.cities:
            self.select_city(self.cities[0])
    
    def select_city(self, city_name):
        """Handle city selection."""
        self.current_city = city_name
        
        for city, btn in self.city_buttons.items():
            if city == city_name:
                btn.configure(fg_color=("#1f538d", "#1f538d"))
            else:
                btn.configure(fg_color=("#3b8ed0", "#1f6aa5"))
        
        self.show_city_config(city_name)
    
    def add_new_city(self):
        """Add a new city to the configuration."""
        dialog = ctk.CTkInputDialog(
            text="Enter city name:",
            title="Add New City"
        )
        city_name = dialog.get_input()
        
        if city_name and city_name.strip():
            city_name = city_name.strip()
            
            if city_name in self.cities:
                messagebox.showwarning("Duplicate City", f"{city_name} already exists in the list.")
                return
            
            self.cities.append(city_name)
            
            # Use deep copy to avoid sharing nested dictionaries between cities
            import copy
            config = copy.deepcopy(self.default_city_config)
            config['name'] = city_name
            self.city_configs[city_name] = config
            
            self.show_main_config()
            self.select_city(city_name)
    
    def remove_city(self, city_name):
        """Remove a city from the configuration."""
        if len(self.cities) <= 1:
            messagebox.showwarning("Cannot Remove", "You must have at least one city in the configuration.")
            return
        
        result = messagebox.askyesno(
            "Remove City",
            f"Are you sure you want to remove '{city_name}' from the configuration?\n\n"
            "This will delete all settings for this city."
        )
        
        if result:
            # Remove from lists
            self.cities.remove(city_name)
            if city_name in self.city_configs:
                del self.city_configs[city_name]
            
            # If this was the current city, select another one
            if self.current_city == city_name:
                self.current_city = None
            
            # Refresh the UI
            self.show_main_config()
            
            # Select the first city if available
            if self.cities:
                self.select_city(self.cities[0])
    
    def show_city_config(self, city_name):
        """Show configuration panel for selected city."""
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        
        scrollable_frame = ctk.CTkScrollableFrame(self.config_frame, fg_color="transparent")
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            scrollable_frame, 
            text=f"Configuration for {city_name}", 
            font=("Inter", 18, "bold")
        ).grid(row=0, column=0, columnspan=3, pady=10)
        
        if city_name not in self.city_configs:
            config = self.default_city_config.copy()
            config['name'] = city_name
            config['government_data_paths'] = self.default_city_config['government_data_paths'].copy()
            self.city_configs[city_name] = config
        
        config = self.city_configs[city_name]
        self.city_widgets = {}
        
        row = 1
        
        # Curb ramp trustworthy checkbox
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Curb Ramp Trustworthy:").pack(side="left")
        info_btn = create_info_button(label_frame, 'curbramp_trustworthy')
        info_btn.pack(side="left", padx=3)
        
        trustworthy_var = ctk.BooleanVar(value=config.get('curbramp_trustworthy', True))
        checkbox = ctk.CTkCheckBox(scrollable_frame, text="", variable=trustworthy_var)
        checkbox.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.city_widgets['curbramp_trustworthy'] = trustworthy_var
        row += 1
        
        # File path fields
        file_fields = [
            ('parcels', 'Parcels File'),
            ('street_centerlines', 'Street Centerlines'),
            ('intersection_nodes', 'Intersection Nodes'),
            ('sidewalks', 'Sidewalks'),
            ('bikelanes', 'Bikelanes'),
            ('curb_ramps', 'Curb Ramps'),
            ('crosswalks', 'Crosswalks'),
            ('street_features', 'Street Features'),
            ('sidewalk_features', 'Sidewalk Features'),
            ('bikeway_features', 'Bikeway Features'),
        ]
        
        for key, label in file_fields:
            label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(label_frame, text=f"{label}:").pack(side="left")
            info_btn = create_info_button(label_frame, key)
            info_btn.pack(side="left", padx=3)
            
            entry = ctk.CTkEntry(scrollable_frame, width=300)
            entry.grid(row=row, column=1, padx=5, pady=5)
            
            path_value = config['government_data_paths'].get(key, '')
            if path_value:
                entry.insert(0, path_value)
            
            # Button frame for Browse and Column Config
            btn_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            btn_frame.grid(row=row, column=2, padx=5, pady=5)
            
            browse_btn = ctk.CTkButton(btn_frame, text="Browse...", 
                                       command=lambda e=entry, k=key, c=city_name: self.browse_file_with_validation(e, k, c),
                                       width=100)
            browse_btn.pack(side="left", padx=2)
            
            # Show Column Config for all data types that support column mapping
            if key in COLUMN_MAPPINGS:
                config_btn = ctk.CTkButton(btn_frame, text="Column Config", 
                                          command=lambda k=key, c=city_name: self.show_column_config(k, c),
                                          width=120,
                                          fg_color="transparent",
                                          border_width=2)
                
                # Initially hide if no file path set
                if not path_value or not Path(path_value).exists():
                    config_btn.pack_forget()
                else:
                    config_btn.pack(side="left", padx=2)
                
                # Store with city-specific key
                button_key = f"{city_name}_{key}"
                self.column_config_buttons[button_key] = config_btn
            
            self.city_widgets[key] = entry
            row += 1
        
        ctk.CTkButton(scrollable_frame, text="Save City Config", 
                  command=lambda: self.save_city_config(city_name), height=35).grid(
            row=row, column=0, columnspan=3, pady=20)
    
    def browse_file(self, entry_widget):
        """Open file browser and update entry widget."""
        filename = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("All files", "*.*"),
                ("GeoJSON files", "*.geojson"),
                ("JSON files", "*.json"),
                ("Shapefile", "*.shp"),
            ]
        )
        if filename:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, filename)
    
    def browse_file_with_validation(self, entry_widget, data_type, city_name):
        """Open file browser, update entry, and show/hide Column Config button."""
        filename = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("All files", "*.*"),
                ("GeoJSON files", "*.geojson"),
                ("JSON files", "*.json"),
                ("Shapefile", "*.shp"),
                ("CSV files", "*.csv"),
            ]
        )
        if filename:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, filename)
            
            # Show Column Config button if this data type supports it
            button_key = f"{city_name}_{data_type}"
            if button_key in self.column_config_buttons:
                file_path = Path(filename)
                if file_path.exists():
                    # Just show the button - validation will happen when user clicks it
                    self.column_config_buttons[button_key].pack(side="left", padx=2)
    
    def get_file_columns(self, file_path):
        """Extract column names from a geospatial file (headers/schema only, no data loading)."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"File does not exist: {file_path}")
                return None
            
            print(f"Reading column headers from: {file_path}")
            
            file_ext = file_path.suffix.lower()
            
            # Read only headers/schema based on file type
            if file_ext in ['.geojson', '.json']:
                # For GeoJSON, use streaming to find first feature without loading entire file
                import json
                columns = []
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Try to parse incrementally
                        buffer = ""
                        feature_found = False
                        
                        for line in f:
                            buffer += line
                            
                            # Look for first complete feature
                            if '"properties"' in buffer and not feature_found:
                                try:
                                    # Try to extract just the properties section
                                    import re
                                    match = re.search(r'"properties"\s*:\s*\{([^}]*)\}', buffer, re.DOTALL)
                                    if match:
                                        props_str = match.group(1)
                                        # Extract all keys
                                        keys = re.findall(r'"([^"]+)"\s*:', props_str)
                                        if keys:
                                            columns = keys
                                            feature_found = True
                                            break
                                except:
                                    pass
                            
                            # Stop after reading enough to find first feature
                            if len(buffer) > 100000:  # 100KB limit
                                break
                        
                        # If regex didn't work, try parsing the buffer as JSON
                        if not columns:
                            try:
                                data = json.loads(buffer)
                                if 'features' in data and len(data['features']) > 0:
                                    first_feature = data['features'][0]
                                    if 'properties' in first_feature:
                                        columns = list(first_feature['properties'].keys())
                            except:
                                pass
                    
                    if columns:
                        print(f"Found {len(columns)} columns in GeoJSON: {columns}")
                        return columns
                    else:
                        print("Could not extract columns from GeoJSON")
                        return []
                        
                except Exception as e:
                    print(f"Error reading GeoJSON: {e}")
                    return []
                
            elif file_ext == '.shp':
                # For Shapefile, use fiona to read schema without loading data
                try:
                    import fiona
                    with fiona.open(file_path) as src:
                        columns = list(src.schema['properties'].keys())
                        print(f"Found {len(columns)} columns in Shapefile (schema only): {columns}")
                        return columns
                except ImportError:
                    print("Fiona not available, using geopandas with minimal read")
                    # Fallback to geopandas with minimal rows
                    gdf = gpd.read_file(file_path, rows=1)
                    columns = [col for col in gdf.columns if col.lower() != 'geometry']
                    print(f"Found {len(columns)} columns in Shapefile: {columns}")
                    return columns
                except Exception as e:
                    print(f"Error reading Shapefile schema: {e}")
                    return None
                    
            elif file_ext == '.csv':
                # For CSV, just read the header row (no data)
                df = pd.read_csv(file_path, nrows=0, low_memory=False)
                columns = list(df.columns)
                print(f"Found {len(columns)} columns in CSV (header only): {columns}")
                return columns
                
            else:
                print(f"Unsupported file type: {file_ext}")
                return None
            
        except Exception as e:
            print(f"Error reading file columns from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_city_config(self, city_name):
        """Save current city configuration."""
        config = self.city_configs[city_name]
        
        config['curbramp_trustworthy'] = self.city_widgets['curbramp_trustworthy'].get()
        
        for key in ['parcels', 'street_centerlines', 'intersection_nodes', 
                    'sidewalks', 'bikelanes', 'curb_ramps', 'crosswalks',
                    'street_features', 'sidewalk_features', 'bikeway_features']:
            config['government_data_paths'][key] = self.city_widgets[key].get()
        
        # Saved successfully - no popup needed
    
    def show_global_settings(self):
        """Show global settings configuration window."""
        if self.current_city:
            self.save_city_config(self.current_city)
        
        for key, value in self.default_global_settings.items():
            if key not in self.global_settings:
                self.global_settings[key] = value
        
        global_window = ctk.CTkToplevel(self.root)
        global_window.title("Global Settings")
        
        screen_width = global_window.winfo_screenwidth()
        screen_height = global_window.winfo_screenheight()
        
        window_width = min(800, int(screen_width * 0.6))
        window_height = min(600, int(screen_height * 0.7))
        
        x = (screen_width - window_width) // 2
        y = max(0, (screen_height - window_height) // 2 - 30)
        
        global_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        global_window.minsize(600, 400)
        
        def on_close():
            global_window.grab_release()
            global_window.destroy()
        
        global_window.protocol("WM_DELETE_WINDOW", on_close)
        
        global_window.transient(self.root)
        global_window.grab_set()
        global_window.focus_force()
        global_window.lift()
        global_window.attributes('-topmost', True)
        global_window.after(100, lambda: global_window.attributes('-topmost', False))
        
        global_window.update_idletasks()
        
        main_container = ctk.CTkFrame(global_window, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        title_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            title_frame, 
            text="Global Settings", 
            font=("Inter", 18, "bold")
        ).pack()
        
        button_frame = ctk.CTkFrame(main_container, fg_color=("#2b2b2b", "#2b2b2b"), height=60)
        button_frame.pack(side="bottom", fill="x", pady=(10, 0))
        button_frame.pack_propagate(False)
        
        def save_global_settings():
            self.global_settings['output_dir'] = global_widgets['output_dir'].get()
            self.global_settings['default_max_speed'] = int(global_widgets['default_max_speed'].get())
            self.global_settings['curb_ramp_trustworthiness_outer_buffer'] = global_widgets['outer_buffer'].get()
            self.global_settings['curb_ramp_trustworthiness_inner_buffer'] = global_widgets['inner_buffer'].get()
            self.global_settings['python_env'] = global_widgets['python_env'].get()
            self.global_settings['use_gpu'] = global_widgets['use_gpu'].get()
            
            # Saved successfully - no popup needed
            global_window.grab_release()
            global_window.destroy()
        
        def cancel_global_settings():
            global_window.grab_release()
            global_window.destroy()
        
        save_btn = ctk.CTkButton(
            button_frame, 
            text="Save & Close", 
            command=save_global_settings,
            width=150,
            height=40,
            font=("Inter", 13, "bold"),
            corner_radius=8
        )
        save_btn.pack(side="left", padx=20, pady=10, expand=True)
        
        cancel_btn = ctk.CTkButton(
            button_frame, 
            text="Cancel", 
            command=cancel_global_settings,
            width=150,
            height=40,
            font=("Inter", 13, "bold"),
            fg_color="transparent",
            border_width=2,
            corner_radius=8
        )
        cancel_btn.pack(side="left", padx=20, pady=10, expand=True)
        
        scrollable_frame = ctk.CTkScrollableFrame(
            main_container, 
            fg_color="transparent"
        )
        scrollable_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        scrollable_frame.grid_columnconfigure(1, weight=1)
        
        global_widgets = {}
        row = 0
        
        # Output directory
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Output Directory:").pack(side="left")
        info_btn = create_info_button(label_frame, 'output_dir')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=300)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        entry.insert(0, self.global_settings.get('output_dir', 'Output'))
        
        btn = ctk.CTkButton(scrollable_frame, text="Browse...", 
                           command=lambda: self.browse_directory(entry))
        btn.grid(row=row, column=2, padx=5, pady=5)
        
        global_widgets['output_dir'] = entry
        row += 1
        
        # Default max speed
        ctk.CTkLabel(scrollable_frame, text="Default Max Speed (mph):").grid(
            row=row, column=0, sticky="w", padx=5, pady=5)
        
        speed_var = ctk.IntVar(value=self.global_settings.get('default_max_speed', 25))
        speed_entry = ctk.CTkEntry(scrollable_frame, width=100, textvariable=speed_var)
        speed_entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        global_widgets['default_max_speed'] = speed_var
        row += 1
        
        # Curb ramp buffers
        ctk.CTkLabel(scrollable_frame, text="Curb Ramp Outer Buffer (m):").grid(
            row=row, column=0, sticky="w", padx=5, pady=5)
        
        outer_var = ctk.DoubleVar(value=self.global_settings.get('curb_ramp_trustworthiness_outer_buffer', 15.0))
        outer_slider = ctk.CTkSlider(scrollable_frame, from_=5.0, to=30.0, variable=outer_var, width=250)
        outer_slider.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        
        outer_label = ctk.CTkLabel(scrollable_frame, text=f"{outer_var.get():.1f}m")
        outer_label.grid(row=row, column=2, padx=5, pady=5)
        
        def update_outer_label(*args):
            outer_label.configure(text=f"{outer_var.get():.1f}m")
        outer_var.trace('w', update_outer_label)
        
        global_widgets['outer_buffer'] = outer_var
        row += 1
        
        ctk.CTkLabel(scrollable_frame, text="Curb Ramp Inner Buffer (m):").grid(
            row=row, column=0, sticky="w", padx=5, pady=5)
        
        inner_var = ctk.DoubleVar(value=self.global_settings.get('curb_ramp_trustworthiness_inner_buffer', 5.0))
        inner_slider = ctk.CTkSlider(scrollable_frame, from_=1.0, to=15.0, variable=inner_var, width=250)
        inner_slider.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        
        inner_label = ctk.CTkLabel(scrollable_frame, text=f"{inner_var.get():.1f}m")
        inner_label.grid(row=row, column=2, padx=5, pady=5)
        
        def update_inner_label(*args):
            inner_label.configure(text=f"{inner_var.get():.1f}m")
        inner_var.trace('w', update_inner_label)
        
        global_widgets['inner_buffer'] = inner_var
        row += 1
        
        # Python environment
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Python Environment:").pack(side="left")
        
        python_env_entry = ctk.CTkEntry(scrollable_frame, width=300)
        python_env_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew", columnspan=2)
        python_env_entry.insert(0, self.global_settings.get('python_env', sys.executable))
        
        global_widgets['python_env'] = python_env_entry
        row += 1
        
        # GPU usage checkbox
        ctk.CTkLabel(scrollable_frame, text="Use GPU:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5)
        
        use_gpu_var = ctk.BooleanVar(value=self.global_settings.get('use_gpu', True))
        use_gpu_checkbox = ctk.CTkCheckBox(
            scrollable_frame, 
            text="Enable GPU acceleration (uncheck if screen flickering occurs)",
            variable=use_gpu_var
        )
        use_gpu_checkbox.grid(row=row, column=1, padx=5, pady=5, sticky="w", columnspan=2)
        global_widgets['use_gpu'] = use_gpu_var
        row += 1
    
    def browse_directory(self, entry_widget):
        """Open directory browser and update entry widget."""
        dirname = filedialog.askdirectory(title="Select Directory")
        if dirname:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, dirname)
    
    def show_column_config(self, data_type, city_name):
        """Show column mapping configuration dialog for a data type."""
        if city_name not in self.city_configs:
            messagebox.showerror("Error", f"City configuration not found: {city_name}")
            return
        
        config = self.city_configs[city_name]
        
        # Get the file path from the entry widget (current value)
        if data_type not in self.city_widgets:
            messagebox.showerror("Error", f"Widget not found for {data_type}")
            return
        
        file_path = self.city_widgets[data_type].get().strip()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a file first before configuring columns")
            return
        
        if not Path(file_path).exists():
            messagebox.showerror("Error", f"File not found: {file_path}\n\nPlease select a valid file first.")
            return
        
        # Read columns from the file
        available_columns = self.get_file_columns(file_path)
        if not available_columns:
            messagebox.showerror("Error", f"Could not read columns from the selected file.\n\nFile: {file_path}\n\nPlease ensure the file is a valid GeoJSON, Shapefile, or CSV.")
            return
        
        # Initialize column_mappings if not present
        if 'column_mappings' not in config:
            config['column_mappings'] = {}
        if data_type not in config['column_mappings']:
            config['column_mappings'][data_type] = {}
        
        # Get the column mapping template for this data type
        if data_type not in COLUMN_MAPPINGS:
            messagebox.showwarning("Warning", f"No column mapping template for {data_type}")
            return
        
        mapping_template = COLUMN_MAPPINGS[data_type]
        
        # Create dialog window
        dialog = ctk.CTkToplevel(self.root)
        dialog.title(f"Column Mapping - {data_type.replace('_', ' ').title()}")
        
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        window_width = min(900, int(screen_width * 0.7))
        window_height = min(700, int(screen_height * 0.8))
        
        x = (screen_width - window_width) // 2
        y = max(0, (screen_height - window_height) // 2 - 30)
        
        dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")
        dialog.minsize(700, 500)
        
        def on_close():
            dialog.grab_release()
            dialog.destroy()
        
        dialog.protocol("WM_DELETE_WINDOW", on_close)
        
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.focus_force()
        dialog.lift()
        dialog.attributes('-topmost', True)
        dialog.after(100, lambda: dialog.attributes('-topmost', False))
        
        dialog.update_idletasks()
        
        # Main container
        main_container = ctk.CTkFrame(dialog, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Title
        title_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            title_frame, 
            text=f"Column Mapping: {data_type.replace('_', ' ').title()}", 
            font=("Inter", 18, "bold")
        ).pack()
        
        ctk.CTkLabel(
            title_frame,
            text=f"Map columns from your government data file to the output schema\nCity: {city_name}",
            font=("Inter", 11),
            text_color=("#999999", "#999999")
        ).pack(pady=(5, 0))
        
        # File info
        file_info_label = ctk.CTkLabel(
            title_frame,
            text=f"File: {Path(file_path).name} ({len(available_columns)} columns available)",
            font=("Inter", 9),
            text_color=("#777777", "#777777")
        )
        file_info_label.pack(pady=(5, 0))
        
        # Button frame at bottom
        button_frame = ctk.CTkFrame(main_container, fg_color=("#2b2b2b", "#2b2b2b"), height=60)
        button_frame.pack(side="bottom", fill="x", pady=(10, 0))
        button_frame.pack_propagate(False)
        
        # Store mapping widgets
        mapping_widgets = {}
        
        def save_mappings():
            """Save the column mappings."""
            for output_col, dropdown_widget in mapping_widgets.items():
                input_col = dropdown_widget.get()
                if input_col and input_col != "(none)":
                    config['column_mappings'][data_type][output_col] = input_col
                elif output_col in config['column_mappings'][data_type]:
                    # Remove mapping if cleared
                    del config['column_mappings'][data_type][output_col]
            
            # Saved successfully - no popup needed
            dialog.grab_release()
            dialog.destroy()
        
        def cancel_mappings():
            dialog.grab_release()
            dialog.destroy()
        
        save_btn = ctk.CTkButton(
            button_frame, 
            text="Save Mappings", 
            command=save_mappings,
            width=150,
            height=40,
            font=("Inter", 13, "bold"),
            corner_radius=8
        )
        save_btn.pack(side="left", padx=20, pady=10, expand=True)
        
        cancel_btn = ctk.CTkButton(
            button_frame, 
            text="Cancel", 
            command=cancel_mappings,
            width=150,
            height=40,
            font=("Inter", 13, "bold"),
            fg_color="transparent",
            border_width=2,
            corner_radius=8
        )
        cancel_btn.pack(side="left", padx=20, pady=10, expand=True)
        
        # Scrollable frame for mappings
        scrollable_frame = ctk.CTkScrollableFrame(
            main_container, 
            fg_color="transparent"
        )
        scrollable_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        scrollable_frame.grid_columnconfigure(1, weight=1)
        
        # Instructions
        instructions = ctk.CTkLabel(
            scrollable_frame,
            text="Map data attribute columns from your file to the output schema.\n"
                 "Geometry is auto-detected - only data attributes need to be mapped.\n"
                 "Select '(none)' if a field is not available in your data.",
            font=("Inter", 10),
            text_color=("#888888", "#888888"),
            justify="left"
        )
        instructions.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 15))
        
        row = 1
        
        # Prepare dropdown values (add "(none)" option)
        dropdown_values = ["(none)"] + sorted(available_columns)
        
        # Create mapping fields
        for output_col, description in mapping_template.items():
            # Output column label
            label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=8)
            
            ctk.CTkLabel(
                label_frame, 
                text=f"{output_col}:",
                font=("Inter", 11, "bold")
            ).pack(side="left")
            
            # Description
            desc_label = ctk.CTkLabel(
                scrollable_frame,
                text=description,
                font=("Inter", 9),
                text_color=("#777777", "#777777"),
                anchor="w"
            )
            desc_label.grid(row=row+1, column=0, sticky="w", padx=20, pady=(0, 5))
            
            # Dropdown for column selection
            dropdown = ctk.CTkComboBox(
                scrollable_frame, 
                values=dropdown_values,
                width=300,
                state="readonly"
            )
            dropdown.grid(row=row, column=1, padx=5, pady=8, sticky="ew")
            
            # Pre-select if mapping exists
            if output_col in config['column_mappings'].get(data_type, {}):
                existing_mapping = config['column_mappings'][data_type][output_col]
                if existing_mapping in available_columns:
                    dropdown.set(existing_mapping)
                else:
                    dropdown.set("(none)")
            else:
                dropdown.set("(none)")
            
            mapping_widgets[output_col] = dropdown
            row += 2
    
    def save_config(self):
        """Save configuration to ProximityModel.py file."""
        if self.current_city:
            self.save_city_config(self.current_city)
        
        try:
            self.write_config_to_file()
            messagebox.showinfo("Success", "Configuration saved to ProximityModel.py")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
            import traceback
            traceback.print_exc()
    
    def write_config_to_file(self):
        """Write configuration back to ProximityModel.py."""
        script_dir = Path(__file__).parent
        proximity_model_file = script_dir / "ProximityModel.py"
        
        if not proximity_model_file.exists():
            raise FileNotFoundError("ProximityModel.py not found")
        
        # Read the file
        with open(proximity_model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the def main(): section
        main_start = content.find('def main():')
        if main_start == -1:
            raise ValueError("Could not find main() function in ProximityModel.py")
        
        # Find the end of main() - look for next function or if __name__
        main_end = content.find('\nif __name__ == "__main__":', main_start)
        if main_end == -1:
            main_end = len(content)
        
        # Generate new config code
        config_code = self.generate_config_code()
        
        # Replace main() function
        new_content = content[:main_start] + config_code + '\n\n' + content[main_end:]
        
        # Write back
        with open(proximity_model_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    def generate_config_code(self):
        """Generate Python code for configuration in ProximityModel.py format."""
        lines = ['def main():\n']
        lines.append('    """Main entry point."""\n')
        lines.append('    # Create output directory\n')
        lines.append(f'    os.makedirs("{self.global_settings["output_dir"]}", exist_ok=True)\n\n')
        lines.append('    # Define global configuration\n')
        lines.append('    GLOBAL_CONFIG = GlobalConfig(\n')
        lines.append(f'        output_dir="{self.global_settings["output_dir"]}",\n')
        lines.append(f'        default_max_speed={self.global_settings["default_max_speed"]},\n')
        lines.append(f'        curb_ramp_trustworthiness_outer_buffer={self.global_settings["curb_ramp_trustworthiness_outer_buffer"]},\n')
        lines.append(f'        curb_ramp_trustworthiness_inner_buffer={self.global_settings["curb_ramp_trustworthiness_inner_buffer"]},\n')
        lines.append(f'        use_gpu={self.global_settings.get("use_gpu", True)},\n')
        lines.append(f'        python_env="{self.global_settings.get("python_env", sys.executable)}",\n')
        lines.append('        cities=[\n')
        
        # Write city configs
        for city_name in self.cities:
            config = self.city_configs[city_name]
            lines.append('            CityConfig(\n')
            lines.append(f'                name="{city_name}",\n')
            lines.append('                government_data_paths=GovernmentDataPaths(\n')
            
            # Write government data paths
            gov_paths = config['government_data_paths']
            for key, value in gov_paths.items():
                if value:
                    lines.append(f'                    {key}="{value}",\n')
            
            lines.append('                ),\n')
            lines.append('                column_mappings=ColumnMappingConfig(\n')
            
            # Write column mappings - need to map from data_type.field to ColumnMappingConfig field
            col_mappings = config.get('column_mappings', {})
            
            # Map curb_ramps fields
            if 'curb_ramps' in col_mappings:
                cr_map = col_mappings['curb_ramps']
                if 'public_data_id' in cr_map:
                    lines.append(f'                    curbramp_id="{cr_map["public_data_id"]}",\n')
                if 'returnloc' in cr_map:
                    lines.append(f'                    curbramp_return_loc="{cr_map["returnloc"]}",\n')
                if 'returnposition' in cr_map:
                    lines.append(f'                    curbramp_position="{cr_map["returnposition"]}",\n')
                if 'condition_score' in cr_map:
                    lines.append(f'                    curbramp_condition="{cr_map["condition_score"]}",\n')
            
            # Map street_centerlines fields
            if 'street_centerlines' in col_mappings:
                st_map = col_mappings['street_centerlines']
                if 'public_data_id' in st_map:
                    lines.append(f'                    street_id="{st_map["public_data_id"]}",\n')
                if 'name' in st_map:
                    lines.append(f'                    street_name="{st_map["name"]}",\n')
                if 'highway' in st_map:
                    lines.append(f'                    street_highway="{st_map["highway"]}",\n')
                if 'maxspeed' in st_map:
                    lines.append(f'                    street_maxspeed="{st_map["maxspeed"]}",\n')
                if 'lanes' in st_map:
                    lines.append(f'                    street_lanes="{st_map["lanes"]}",\n')
                if 'surface' in st_map:
                    lines.append(f'                    street_surface="{st_map["surface"]}",\n')
            
            # Map sidewalks fields
            if 'sidewalks' in col_mappings:
                sw_map = col_mappings['sidewalks']
                if 'public_data_id' in sw_map:
                    lines.append(f'                    sidewalk_id="{sw_map["public_data_id"]}",\n')
                if 'surface' in sw_map:
                    lines.append(f'                    sidewalk_surface="{sw_map["surface"]}",\n')
                if 'width' in sw_map:
                    lines.append(f'                    sidewalk_width="{sw_map["width"]}",\n')
                if 'incline' in sw_map:
                    lines.append(f'                    sidewalk_incline="{sw_map["incline"]}",\n')
            
            # Map bikelanes fields
            if 'bikelanes' in col_mappings:
                bk_map = col_mappings['bikelanes']
                if 'public_data_id' in bk_map:
                    lines.append(f'                    bikelane_id="{bk_map["public_data_id"]}",\n')
                if 'type' in bk_map:
                    lines.append(f'                    bikelane_type="{bk_map["type"]}",\n')
                if 'surface' in bk_map:
                    lines.append(f'                    bikelane_surface="{bk_map["surface"]}",\n')
                if 'width' in bk_map:
                    lines.append(f'                    bikelane_width="{bk_map["width"]}",\n')
            
            # Map feature fields (street/sidewalk/bikeway features)
            for feature_type in ['street_features', 'sidewalk_features', 'bikeway_features']:
                if feature_type in col_mappings:
                    feat_map = col_mappings[feature_type]
                    if 'public_data_id' in feat_map:
                        lines.append(f'                    feature_id="{feat_map["public_data_id"]}",\n')
                    if 'feature_type' in feat_map:
                        lines.append(f'                    feature_type="{feat_map["feature_type"]}",\n')
                    break  # All feature types use the same fields in ColumnMappingConfig
            
            # Map crosswalk fields
            if 'crosswalks' in col_mappings:
                cw_map = col_mappings['crosswalks']
                # Crosswalks don't have specific column mappings in ColumnMappingConfig yet
                # This is a placeholder for future expansion
                pass
            
            lines.append('                ),\n')
            lines.append(f'                curbramp_trustworthy={config.get("curbramp_trustworthy", False)},\n')
            lines.append('            ),\n')
        
        lines.append('        ]\n')
        lines.append('    )\n\n')
        
        # Add execution code using new pipeline stages
        lines.append('    # Run pipeline for each city\n')
        lines.append('    for city in GLOBAL_CONFIG.cities:\n')
        lines.append('        print(f"\\n{\'=\'*80}")\n')
        lines.append('        print(f"Processing: {city.name}")\n')
        lines.append('        print(f"{\'=\'*80}\\n")\n')
        lines.append('        _display_gpu_status()\n')
        lines.append('        _reset_pipeline_counters()\n')
        lines.append('        network_output = os.path.join(\n')
        lines.append('            GLOBAL_CONFIG.output_dir,\n')
        lines.append('            f"{city.name.replace(\', \', \'_\').replace(\' \', \'_\')}_network.parquet"\n')
        lines.append('        )\n')
        lines.append('        try:\n')
        lines.append('            edges, crossings_cache = _stage_load_and_init(city, GLOBAL_CONFIG, network_output)\n')
        lines.append('            sanity_buffer          = _stage_sanity_buffer(edges, city, GLOBAL_CONFIG)\n')
        lines.append('            edges, gov_nodes       = _stage_government_data(edges, city, GLOBAL_CONFIG, network_output)\n')
        lines.append('            edges                  = _stage_tag_extraction(edges)\n')
        lines.append('            edges                  = _stage_deflection_split(edges, GLOBAL_CONFIG, city, network_output)\n')
        lines.append('            edges                  = _stage_offset_and_features(edges, sanity_buffer, city, GLOBAL_CONFIG)\n')
        lines.append('            edges                  = _stage_bearing_and_blocks(edges, gov_nodes)\n')
        lines.append('            edges                  = _stage_crosswalks(edges, crossings_cache, city)\n')
        lines.append('            edges                  = _stage_intersection_analysis(edges, sanity_buffer, GLOBAL_CONFIG)\n')
        lines.append('            edges                  = _stage_curb_ramps(edges, city, GLOBAL_CONFIG, network_output)\n')
        lines.append('            _stage_finalize(edges, city, GLOBAL_CONFIG, network_output)\n')
        lines.append('            print(f"\\n{\'=\'*80}")\n')
        lines.append('            print(f"✓ Successfully processed: {city.name}")\n')
        lines.append('            print(f"{\'=\'*80}\\n")\n')
        lines.append('        except Exception:\n')
        lines.append('            print(f"\\n{\'=\'*80}")\n')
        lines.append('            print(f"✗ Error processing {city.name}:")\n')
        lines.append('            print(f"{\'=\'*80}")\n')
        lines.append('            traceback.print_exc()\n')
        lines.append('            print()\n')
        
        return ''.join(lines)
    
    def run_analysis(self):
        """Save configuration and run analysis."""
        if self.current_city:
            self.save_city_config(self.current_city)
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        proximity_model_file = script_dir / "ProximityModel.py"
        
        # Save config to ProximityModel.py first
        try:
            self.write_config_to_file()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
            return
        
        # Confirm with user
        result = messagebox.askyesno(
            "Run Analysis",
            f"Configuration saved for {len(self.cities)} cities.\n\n"
            "The analysis will now run in a new console window.\n"
            "This may take several minutes to hours depending on the size of the data.\n\n"
            "Continue?"
        )
        
        if result:
            try:
                # Get Python executable from global settings
                python_exe = self.global_settings.get('python_env', sys.executable)
                
                # Run ProximityModel.py in a new console window
                if sys.platform == 'win32':
                    # Windows: Use start command to open new console
                    # Use proper escaping for paths with spaces
                    cmd = f'start "Proximity Model Analysis" cmd /k ""{python_exe}" "{proximity_model_file}""'
                    subprocess.Popen(
                        cmd,
                        shell=True,
                        cwd=str(script_dir)
                    )
                else:
                    # Unix-like: Use terminal emulator
                    subprocess.Popen(
                        [python_exe, str(proximity_model_file)],
                        cwd=str(script_dir)
                    )
                
                messagebox.showinfo(
                    "Analysis Starting",
                    "Configuration saved successfully.\n\n"
                    "The analysis is running in a separate console window.\n"
                    "Check the console window for progress.\n\n"
                    "This UI will now close."
                )
                
                # Close the UI
                self.root.quit()
                self.root.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start analysis: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point for the UI."""
    root = ctk.CTk()
    app = ProximityModelConfigUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
