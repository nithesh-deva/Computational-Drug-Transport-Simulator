"""
2D rectangular multilayer tissue geometry with Gmsh mesh generation for M6.

Supports:
- 2D rectangular domain with multiple horizontal tissue layers
- Adaptive mesh refinement strategies (uniform, layer-based, boundary)
- Gmsh integration for high-quality mesh generation
- VTK/ParaView export for visualization
- Interface partition coefficient specification
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class Layer2D:
    """2D tissue layer specification.
    
    Attributes:
        name: Layer identifier
        thickness: Layer thickness in y-direction [m]
        width: Domain width in x-direction [m] (shared across all layers)
        diffusion_coefficient: D [m²/s]
        clearance: k [1/s]
        refinement_level: Mesh refinement indicator (1=coarse, higher=finer)
    """
    name: str
    thickness: float
    width: float
    diffusion_coefficient: float
    clearance: float
    refinement_level: int = 1
    
    def validate(self):
        """Validate layer specification."""
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.width <= 0:
            raise ValueError(f"Layer width must be positive, got {self.width}")
        if self.diffusion_coefficient < 0:
            raise ValueError(f"Diffusion coefficient must be non-negative, got {self.diffusion_coefficient}")
        if self.clearance < 0:
            raise ValueError(f"Clearance must be non-negative, got {self.clearance}")
        if self.refinement_level < 1:
            raise ValueError(f"Refinement level must be ≥ 1, got {self.refinement_level}")


class Rectangular2DDomain:
    """2D rectangular multilayer tissue domain.
    
    Manages:
    - Multiple horizontal tissue layers
    - 2D spatial coordinates (x, y)
    - Interface conditions between layers
    - Gmsh mesh generation with refinement strategies
    - Conversion to FEM data structures
    """
    
    def __init__(self, layers: List[Dict[str, Any]], width: Optional[float] = None):
        """Initialize 2D rectangular domain from layer specifications.
        
        Args:
            layers: List of layer dictionaries, each containing:
                - name: Layer identifier [str]
                - thickness: Layer thickness [m]
                - diffusion_coefficient: D [m²/s]
                - clearance: k [1/s]
                - refinement_level: Mesh refinement (optional, default 1)
            width: Domain width [m]. If None, uses first layer's width or defaults to 0.01 m.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not layers:
            raise ValueError("At least one layer required")
        
        # Determine width
        if width is not None:
            self.width = width
        elif 'width' in layers[0]:
            self.width = layers[0]['width']
        else:
            self.width = 0.01  # Default 10 mm width
        
        # Create Layer2D objects
        self.layers = []
        self.layer_names = []
        
        for layer_dict in layers:
            layer = Layer2D(
                name=layer_dict['name'],
                thickness=layer_dict['thickness'],
                width=self.width,
                diffusion_coefficient=layer_dict['diffusion_coefficient'],
                clearance=layer_dict['clearance'],
                refinement_level=layer_dict.get('refinement_level', 1)
            )
            layer.validate()
            self.layers.append(layer)
            self.layer_names.append(layer.name)
        
        self.n_layers = len(self.layers)
        
        # Calculate interface positions (y-coordinates)
        self.interface_y_positions = []
        y_pos = 0.0
        for layer in self.layers:
            self.interface_y_positions.append(y_pos)
            y_pos += layer.thickness
        self.interface_y_positions.append(y_pos)  # Top boundary
        
        self.total_height = y_pos
        
        logger.info(
            f"2D rectangular domain initialized: {self.n_layers} layers, "
            f"width={self.width:.6e} m, height={self.total_height:.6e} m"
        )
    
    def get_layer_at_y(self, y: float) -> Tuple[int, Layer2D]:
        """Get layer index and layer object at y-coordinate.
        
        Args:
            y: Y-coordinate [m]
        
        Returns:
            Tuple of (layer_index, Layer2D object)
        
        Raises:
            ValueError: If y is outside domain
        """
        if y < 0 or y > self.total_height:
            raise ValueError(f"Y-coordinate {y} outside domain [0, {self.total_height}]")
        
        for i, layer in enumerate(self.layers):
            y_bottom = self.interface_y_positions[i]
            y_top = self.interface_y_positions[i + 1]
            if y_bottom <= y <= y_top:
                return i, layer
        
        # Handle numerical precision at top boundary
        return self.n_layers - 1, self.layers[-1]
    
    def get_properties_at_point(self, x: float, y: float) -> Dict[str, float]:
        """Get tissue properties (D, k) at spatial point.
        
        Args:
            x: X-coordinate [m]
            y: Y-coordinate [m]
        
        Returns:
            Dict with 'diffusion_coefficient' and 'clearance'
        """
        _, layer = self.get_layer_at_y(y)
        return {
            'diffusion_coefficient': layer.diffusion_coefficient,
            'clearance': layer.clearance
        }
    
    def generate_gmsh_mesh(
        self,
        dx_target: float = 1.0e-4,
        gmsh_file: Optional[str] = None,
        refinement_strategy: str = 'uniform'
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate 2D mesh using Gmsh with specified refinement strategy.
        
        Args:
            dx_target: Target element size [m]
            gmsh_file: Output .msh file path. If None, creates temporary.
            refinement_strategy: 'uniform', 'layer-based', or 'boundary'
        
        Returns:
            Tuple of:
            - nodes: (n_nodes, 2) array of [x, y] coordinates [m]
            - elements: (n_elems, 3) array of node indices (triangles)
            - mesh_data: Dict with metadata
        
        Raises:
            ImportError: If gmsh not available
            ValueError: If refinement_strategy invalid
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError(
                "gmsh package required for mesh generation. "
                "Install with: pip install gmsh"
            )
        
        if refinement_strategy not in ['uniform', 'layer-based', 'boundary']:
            raise ValueError(f"Unknown refinement strategy: {refinement_strategy}")
        
        gmsh.initialize()
        gmsh.model.add("rectangular_2d_tissue")
        
        try:
            # Set Gmsh options
            gmsh.option.setNumber("General.Terminal", 0)  # No output
            gmsh.option.setNumber("Mesh.Algorithm", 6)     # Delaunay
            gmsh.option.setNumber("Mesh.Smoothing", 10)    # Smooth mesh
            
            # Create geometry
            model = gmsh.model.geo
            
            # Define corners of rectangle
            corners = [
                model.addPoint(0, 0, 0, dx_target),
                model.addPoint(self.width, 0, 0, dx_target),
                model.addPoint(self.width, self.total_height, 0, dx_target),
                model.addPoint(0, self.total_height, 0, dx_target)
            ]
            
            # Create boundary lines
            bottom_line = model.addLine(corners[0], corners[1])
            right_line = model.addLine(corners[1], corners[2])
            top_line = model.addLine(corners[2], corners[3])
            left_line = model.addLine(corners[3], corners[0])
            
            # Add interface lines for layer boundaries (internal)
            interface_lines = []
            interface_points_left = []
            interface_points_right = []
            
            for i in range(1, self.n_layers):
                y_interface = self.interface_y_positions[i]
                
                # Points on left and right edges
                p_left = model.addPoint(0, y_interface, 0, dx_target)
                p_right = model.addPoint(self.width, y_interface, 0, dx_target)
                
                interface_points_left.append(p_left)
                interface_points_right.append(p_right)
                
                # Line connecting left and right
                line = model.addLine(p_left, p_right)
                interface_lines.append(line)
            
            # Create curve loops for each layer
            layer_loops = []
            
            for i in range(self.n_layers):
                if i == 0:
                    # Bottom layer: uses bottom and part of sides
                    p_left_top = interface_points_left[0] if i < self.n_layers - 1 else corners[3]
                    p_right_top = interface_points_right[0] if i < self.n_layers - 1 else corners[2]
                    
                    loop = model.addCurveLoop([
                        bottom_line,
                        model.addLine(corners[1], p_right_top),
                        interface_lines[0] if i < self.n_layers - 1 else top_line,
                        model.addLine(p_left_top, corners[0])
                    ])
                    layer_loops.append(loop)
                elif i == self.n_layers - 1:
                    # Top layer
                    p_left_bot = interface_points_left[-1]
                    p_right_bot = interface_points_right[-1]
                    
                    loop = model.addCurveLoop([
                        interface_lines[-1],
                        right_line,
                        top_line,
                        left_line
                    ])
                    # Adjust to connect properly
                    loop = model.addCurveLoop([
                        model.addLine(p_left_bot, p_right_bot),
                        right_line,
                        top_line,
                        left_line
                    ])
                    layer_loops.append(loop)
                else:
                    # Middle layers
                    p_left_bot = interface_points_left[i - 1]
                    p_right_bot = interface_points_right[i - 1]
                    p_left_top = interface_points_left[i]
                    p_right_top = interface_points_right[i]
                    
                    loop = model.addCurveLoop([
                        interface_lines[i - 1],
                        model.addLine(p_right_bot, p_right_top),
                        interface_lines[i],
                        model.addLine(p_left_top, p_left_bot)
                    ])
                    layer_loops.append(loop)
            
            # Create surfaces for each layer
            surfaces = []
            for i, loop in enumerate(layer_loops):
                surface = model.addPlaneSurface([loop])
                surfaces.append(surface)
                
                # Set refinement based on layer
                layer = self.layers[i]
                if refinement_strategy == 'layer-based':
                    element_size = dx_target / (2 ** (layer.refinement_level - 1))
                    gmsh.model.mesh.setSize(
                        gmsh.model.getEntities(0, surface),
                        element_size
                    )
            
            # Synchronize geometry
            model.synchronize()
            
            # Generate 2D mesh
            gmsh.model.mesh.generate(2)
            
            # Optional: refine mesh at boundaries
            if refinement_strategy == 'boundary':
                gmsh.model.mesh.refine()
            
            # Extract mesh data
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(2)  # 2 = triangles
            
            # Convert to numpy arrays
            nodes = node_coords.reshape((-1, 3))[:, :2]  # Take only x, y
            elements = elem_node_tags.reshape((-1, 3)) - 1  # Convert to 0-indexed
            
            n_nodes = len(nodes)
            n_elements = len(elements)
            
            # Save mesh file if requested
            if gmsh_file:
                gmsh.write(gmsh_file)
            
            mesh_data = {
                'n_nodes': n_nodes,
                'n_elements': n_elements,
                'n_layers': self.n_layers,
                'width': self.width,
                'height': self.total_height,
                'dx_target': dx_target,
                'refinement_strategy': refinement_strategy,
                'layer_names': self.layer_names,
                'interface_y_positions': np.array(self.interface_y_positions)
            }
            
            logger.info(
                f"Gmsh mesh generated: {n_nodes} nodes, {n_elements} triangular elements, "
                f"strategy={refinement_strategy}"
            )
            
            return nodes, elements, mesh_data
        
        finally:
            gmsh.finalize()
    
    def get_element_layer(self, element_nodes: np.ndarray, nodes: np.ndarray) -> int:
        """Get layer index for a triangular element.
        
        Uses centroid of element to determine layer.
        
        Args:
            element_nodes: Array of 3 node indices
            nodes: (n_nodes, 2) array of node coordinates
        
        Returns:
            Layer index
        """
        # Calculate centroid
        centroid_y = np.mean(nodes[element_nodes, 1])
        layer_idx, _ = self.get_layer_at_y(centroid_y)
        return layer_idx
    
    def get_element_properties(
        self,
        element_nodes: np.ndarray,
        nodes: np.ndarray
    ) -> Dict[str, float]:
        """Get tissue properties for element (based on centroid layer).
        
        Args:
            element_nodes: Array of 3 node indices
            nodes: (n_nodes, 2) array of node coordinates
        
        Returns:
            Dict with 'diffusion_coefficient' and 'clearance'
        """
        layer_idx = self.get_element_layer(element_nodes, nodes)
        layer = self.layers[layer_idx]
        return {
            'diffusion_coefficient': layer.diffusion_coefficient,
            'clearance': layer.clearance
        }


def create_rectangular_2d_domain(
    layers_config: List[Dict[str, Any]],
    width: Optional[float] = None
) -> Rectangular2DDomain:
    """Factory function to create 2D rectangular domain.
    
    Args:
        layers_config: Layer configuration list
        width: Domain width [m]
    
    Returns:
        Rectangular2DDomain instance
    """
    return Rectangular2DDomain(layers_config, width=width)
