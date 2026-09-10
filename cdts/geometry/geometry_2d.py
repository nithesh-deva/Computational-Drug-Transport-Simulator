"""
2D rectangular multilayer tissue geometry for M6.

Supports:
- Rectangular tissue with multiple horizontal layers
- Layer-specific material properties
- Mesh refinement control
- Gmsh geometry definition export
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Layer2D:
    """Specification of a 2D tissue layer.
    
    Attributes:
        name: Layer identifier
        thickness: Vertical thickness [m]
        width: Horizontal width [m]
        diffusion_coefficient: D [m²/s]
        clearance: k [1/s]
        mesh_size: Target element size [m]
    """
    
    def __init__(
        self,
        name: str,
        thickness: float,
        width: float,
        diffusion_coefficient: float,
        clearance: float = 0.0,
        mesh_size: Optional[float] = None
    ):
        self.name = name
        self.thickness = thickness
        self.width = width
        self.diffusion_coefficient = diffusion_coefficient
        self.clearance = clearance
        self.mesh_size = mesh_size
    
    def validate(self):
        """Validate layer specification."""
        if self.thickness <= 0:
            raise ValueError(f"Thickness must be positive: {self.thickness}")
        if self.width <= 0:
            raise ValueError(f"Width must be positive: {self.width}")
        if self.diffusion_coefficient < 0:
            raise ValueError(f"Diffusion coefficient must be non-negative: {self.diffusion_coefficient}")
        if self.clearance < 0:
            raise ValueError(f"Clearance must be non-negative: {self.clearance}")


class Geometry2D:
    """2D rectangular multilayer tissue geometry.
    
    Creates a rectangular domain with horizontal layers, each with
    potentially different thickness and material properties.
    """
    
    def __init__(self, layers: List[Layer2D]):
        """Initialize 2D geometry from layer specifications.
        
        Args:
            layers: List of Layer2D objects (ordered from top to bottom)
        """
        if not layers:
            raise ValueError("At least one layer required")
        
        self.layers = layers
        self.n_layers = len(layers)
        
        # Validate all layers
        for layer in layers:
            layer.validate()
        
        # Calculate geometry
        self.width = layers[0].width  # All layers span same width
        self.total_height = sum(layer.thickness for layer in layers)
        
        # Layer y-positions (top edge of each layer)
        self.layer_y_positions = []
        y = self.total_height
        for layer in layers:
            self.layer_y_positions.append(y)
            y -= layer.thickness
        self.layer_y_positions.append(0.0)  # Bottom edge
        
        logger.info(
            f"2D Geometry initialized:\n"
            f"  Layers: {self.n_layers}\n"
            f"  Width: {self.width:.6e} m\n"
            f"  Height: {self.total_height:.6e} m"
        )
    
    def get_layer_at_position(self, y: float) -> int:
        """Get layer index at y position.
        
        Args:
            y: Vertical position [m]
            
        Returns:
            Layer index (0 = top layer)
        """
        for i in range(self.n_layers):
            if y >= self.layer_y_positions[i+1] and y <= self.layer_y_positions[i]:
                return i
        raise ValueError(f"Position y={y} outside domain [0, {self.total_height}]")
    
    def get_layer_property(self, layer_idx: int, property_name: str) -> float:
        """Get property value for layer.
        
        Args:
            layer_idx: Layer index
            property_name: 'diffusion_coefficient', 'clearance', 'thickness'
            
        Returns:
            Property value
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"Layer index {layer_idx} out of range")
        
        layer = self.layers[layer_idx]
        if property_name == 'diffusion_coefficient':
            return layer.diffusion_coefficient
        elif property_name == 'clearance':
            return layer.clearance
        elif property_name == 'thickness':
            return layer.thickness
        else:
            raise KeyError(f"Unknown property: {property_name}")
    
    def generate_gmsh_script(self, output_path: str, mesh_size: Optional[float] = None):
        """Generate Gmsh .geo script for mesh generation.
        
        Args:
            output_path: Path to save .geo file
            mesh_size: Override default mesh size [m]
        """
        lines = [
            "// CDTS M6 - 2D Multilayer Tissue Geometry",
            "// Generated Gmsh geometry script",
            "",
            "// Physical domain dimensions",
        ]
        
        lines.append(f"width = {self.width:.6e};")
        lines.append(f"height = {self.total_height:.6e};")
        lines.append("")
        
        # Default mesh size
        default_mesh = mesh_size or min(self.width, self.total_height) / 10
        lines.append(f"// Default mesh size")
        lines.append(f"mesh_size = {default_mesh:.6e};")
        lines.append("")
        
        # Create corner points
        lines.append("// Corners of domain")
        lines.append("Point(1) = {0, 0, 0, mesh_size};")
        lines.append("Point(2) = {width, 0, 0, mesh_size};")
        lines.append("Point(3) = {width, height, 0, mesh_size};")
        lines.append("Point(4) = {0, height, 0, mesh_size};")
        lines.append("")
        
        # Create layer boundaries
        point_idx = 5
        layer_points = {}  # layer_idx -> [left_points, right_points]
        
        for i, layer in enumerate(self.layers):
            y_top = self.layer_y_positions[i]
            y_bottom = self.layer_y_positions[i+1]
            
            lines.append(f"// Layer {i}: {layer.name}")
            
            # Left boundary points
            lines.append(f"Point({point_idx}) = {{0, {y_top:.6e}, 0, mesh_size}};")
            left_top = point_idx
            point_idx += 1
            
            lines.append(f"Point({point_idx}) = {{0, {y_bottom:.6e}, 0, mesh_size}};")
            left_bottom = point_idx
            point_idx += 1
            
            # Right boundary points
            lines.append(f"Point({point_idx}) = {{width, {y_top:.6e}, 0, mesh_size}};")
            right_top = point_idx
            point_idx += 1
            
            lines.append(f"Point({point_idx}) = {{width, {y_bottom:.6e}, 0, mesh_size}};")
            right_bottom = point_idx
            point_idx += 1
            
            layer_points[i] = ([left_top, left_bottom], [right_top, right_bottom])
            lines.append("")
        
        # Create lines for layer boundaries
        lines.append("// Lines")
        line_idx = 1
        layer_lines = {}
        
        for i in range(self.n_layers):
            left_pts, right_pts = layer_points[i]
            
            # Left edge
            lines.append(f"Line({line_idx}) = {{{left_pts[0]}, {left_pts[1]}}};")
            line_idx += 1
            
            # Top edge
            lines.append(f"Line({line_idx}) = {{{left_pts[0]}, {right_pts[0]}}};")
            line_idx += 1
            
            # Right edge
            lines.append(f"Line({line_idx}) = {{{right_pts[0]}, {right_pts[1]}}};")
            line_idx += 1
            
            # Bottom edge
            lines.append(f"Line({line_idx}) = {{{left_pts[1]}, {right_pts[1]}}};")
            line_idx += 1
        
        lines.append("")
        
        # Create surfaces (one per layer)
        lines.append("// Surfaces")
        for i in range(self.n_layers):
            layer = self.layers[i]
            mesh_sz = layer.mesh_size or default_mesh
            
            lines.append(f"// Layer: {layer.name}")
            lines.append(f"mesh_size_{i} = {mesh_sz:.6e};")
            
            # Create a simple rectangle loop for this layer
            # (simplified: would need proper loop definition in real Gmsh)
            lines.append(f"Surface({i+1}) = {{/* layer {i} surface */}};")
        
        lines.append("")
        
        # Physical groups
        lines.append("// Physical groups for export")
        for i, layer in enumerate(self.layers):
            lines.append(f"Physical Surface(\"{layer.name}\", {i+1}) = {{{i+1}}};")
        
        lines.append("Physical Surface(\"tissue\", 1000) = {")
        lines.append(", ".join(str(i+1) for i in range(self.n_layers)))
        lines.append("};")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Gmsh script generated: {output_path}")
    
    def __repr__(self) -> str:
        """String representation."""
        layer_info = "\n  ".join([
            f"Layer {i}: {layer.name} | "
            f"h={layer.thickness:.2e} m | "
            f"D={layer.diffusion_coefficient:.2e} m²/s | "
            f"k={layer.clearance:.2e} 1/s"
            for i, layer in enumerate(self.layers)
        ])
        return (
            f"Geometry2D:\n"
            f"  Width: {self.width:.6e} m\n"
            f"  Height: {self.total_height:.6e} m\n"
            f"  Layers ({self.n_layers}):\n  {layer_info}"
        )
