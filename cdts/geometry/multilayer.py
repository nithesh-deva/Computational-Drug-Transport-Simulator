"""
Multilayer tissue domain representation and mesh generation for M3.

Supports arbitrary number of tissue layers with:
- Layer-specific diffusion coefficients
- Layer-specific clearance coefficients
- Interface partition coefficients
- Layer thickness specification
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayerInterface:
    """Specification of interface between two tissue layers.
    
    Attributes:
        layer_left: Name of left layer
        layer_right: Name of right layer
        partition_coefficient: Partition coefficient K (concentration ratio)
        position: Interface position [m]
        interface_type: 'continuity' or 'partition'
    """
    layer_left: str
    layer_right: str
    partition_coefficient: float = 1.0
    position: float = 0.0
    interface_type: str = 'partition'
    
    def validate(self):
        """Validate interface specification."""
        if self.partition_coefficient <= 0:
            raise ValueError(f"Partition coefficient must be positive, got {self.partition_coefficient}")
        if self.position < 0:
            raise ValueError(f"Interface position must be non-negative, got {self.position}")
        if self.interface_type not in ['continuity', 'partition']:
            raise ValueError(f"Invalid interface type: {self.interface_type}")


class MultilayerDomain:
    """Multilayer tissue domain representation.
    
    Manages:
    - Multiple tissue layers with different properties
    - Interface positions and conditions
    - Spatial mesh generation
    - Layer identification for each grid point
    """
    
    def __init__(self, layers: List[Dict[str, Any]]):
        """Initialize multilayer domain from layer specifications.
        
        Args:
            layers: List of layer dictionaries, each containing:
                - name: Layer identifier [str]
                - thickness: Layer thickness [m]
                - diffusion_coefficient: D [m²/s]
                - clearance: k [1/s]
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not layers:
            raise ValueError("At least one layer required")
        
        self.layers = layers
        self.n_layers = len(layers)
        
        # Calculate interface positions
        self.interface_positions = []
        position = 0.0
        for i, layer in enumerate(layers):
            self.interface_positions.append(position)
            position += layer['thickness']
        self.interface_positions.append(position)  # Right boundary
        
        self.total_length = position
        
        # Validate all layers
        for layer in layers:
            if layer['thickness'] <= 0:
                raise ValueError(f"Layer thickness must be positive: {layer['thickness']}")
            if layer['diffusion_coefficient'] < 0:
                raise ValueError(f"Diffusion coefficient must be non-negative: {layer['diffusion_coefficient']}")
            if layer['clearance'] < 0:
                raise ValueError(f"Clearance must be non-negative: {layer['clearance']}")
        
        logger.info(f"Multilayer domain initialized: {self.n_layers} layers, total length {self.total_length:.6e} m")
    
    def generate_mesh(self, dx: float) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Generate spatial mesh for multilayer domain.
        
        Mesh includes interface positions to ensure layer boundaries are resolved.
        
        Args:
            dx: Target spatial step [m]
            
        Returns:
            Tuple containing:
                - x: Spatial grid points [m]
                - layer_indices: Layer index for each grid point
                - interface_indices: Grid indices of interfaces
        """
        if dx <= 0:
            raise ValueError(f"Grid step must be positive, got {dx}")
        
        # Create base grid
        x_base = np.arange(0, self.total_length + dx/2, dx)
        
        # Add interface positions to ensure they're exactly represented
        x_all = np.concatenate([x_base, np.array(self.interface_positions)])
        x = np.sort(np.unique(x_all))
        
        # Identify which layer each point belongs to
        layer_indices = np.zeros(len(x), dtype=int)
        interface_indices = []
        
        for i, xi in enumerate(x):
            # Find which layer this point is in
            for layer_idx, (left_pos, right_pos) in enumerate(
                zip(self.interface_positions[:-1], self.interface_positions[1:])
            ):
                if left_pos <= xi < right_pos:
                    layer_indices[i] = layer_idx
                    break
                elif np.isclose(xi, right_pos) and layer_idx < self.n_layers - 1:
                    # At interface (not last layer boundary)
                    layer_indices[i] = layer_idx
                    if i not in interface_indices:
                        interface_indices.append(i)
                    break
                elif xi >= right_pos and layer_idx == self.n_layers - 1:
                    # At right boundary
                    layer_indices[i] = layer_idx
                    if i not in interface_indices:
                        interface_indices.append(i)
                    break
        
        logger.info(f"Mesh generated: {len(x)} grid points, {len(interface_indices)} interface points")
        
        return x, layer_indices, interface_indices
    
    def get_layer_property(self, layer_idx: int, property_name: str) -> float:
        """Get property value for specified layer.
        
        Args:
            layer_idx: Layer index (0-based)
            property_name: 'diffusion_coefficient', 'clearance', or 'thickness'
            
        Returns:
            Property value [float]
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.n_layers-1}]")
        
        if property_name not in self.layers[layer_idx]:
            raise KeyError(f"Property '{property_name}' not found in layer {layer_idx}")
        
        return self.layers[layer_idx][property_name]
    
    def get_layer_properties_at_points(
        self,
        x: np.ndarray,
        layer_indices: np.ndarray,
        property_name: str
    ) -> np.ndarray:
        """Get property values at all grid points.
        
        Args:
            x: Spatial grid points [m]
            layer_indices: Layer index for each grid point
            property_name: Property name
            
        Returns:
            Property values at each grid point
        """
        properties = np.zeros(len(x))
        
        for i, layer_idx in enumerate(layer_indices):
            properties[i] = self.get_layer_property(layer_idx, property_name)
        
        return properties
    
    def __repr__(self) -> str:
        """String representation of domain."""
        layer_info = "\n  ".join([
            f"Layer {i}: {layer['name']} "
            f"| thickness={layer['thickness']:.2e} m "
            f"| D={layer['diffusion_coefficient']:.2e} m²/s "
            f"| k={layer['clearance']:.2e} 1/s"
            for i, layer in enumerate(self.layers)
        ])
        return (
            f"MultilayerDomain:\n"
            f"  Total length: {self.total_length:.6e} m\n"
            f"  Layers ({self.n_layers}):\n  {layer_info}"
        )


def validate_partition_coefficient(K: float, layer_left_type: str, layer_right_type: str) -> None:
    """Validate partition coefficient for interface.
    
    Args:
        K: Partition coefficient (dimensionless)
        layer_left_type: Type/name of left layer
        layer_right_type: Type/name of right layer
        
    Raises:
        ValueError: If K is invalid
    """
    if K <= 0:
        raise ValueError(f"Partition coefficient must be positive, got {K}")
    if K > 100:
        logger.warning(f"Partition coefficient {K} is very large; check units")
    if K < 0.01:
        logger.warning(f"Partition coefficient {K} is very small; check units")


def calculate_interface_concentration(
    C_left: float,
    K: float,
    interface_type: str = 'partition'
) -> float:
    """Calculate concentration on right side of interface given left side.
    
    Interface conditions:
    - 'partition': C_right = K · C_left (partition coefficient)
    - 'continuity': C_right = C_left (no partition)
    
    Args:
        C_left: Concentration on left side of interface [mol/m³]
        K: Partition coefficient (dimensionless)
        interface_type: Type of interface condition
        
    Returns:
        Concentration on right side [mol/m³]
    """
    if interface_type == 'partition':
        return K * C_left
    elif interface_type == 'continuity':
        return C_left
    else:
        raise ValueError(f"Unknown interface type: {interface_type}")


def verify_flux_continuity(
    D_left: float,
    D_right: float,
    K: float,
    dC_dx_left: float,
    dC_dx_right: float,
    tolerance: float = 0.05
) -> Tuple[bool, float]:
    """Verify flux continuity at interface (excluding partition effect).
    
    For physical consistency, diffusive flux should be continuous:
        D_left · (∂C/∂x)|_left = D_right · (∂C/∂x)|_right
    
    When partition coefficient K ≠ 1, concentration is discontinuous but
    flux should still follow continuity with concentration jump.
    
    Args:
        D_left: Diffusion coefficient in left layer [m²/s]
        D_right: Diffusion coefficient in right layer [m²/s]
        K: Partition coefficient at interface
        dC_dx_left: Concentration gradient in left layer [mol/m⁴]
        dC_dx_right: Concentration gradient in right layer [mol/m⁴]
        tolerance: Acceptable relative error in flux
        
    Returns:
        Tuple:
            - is_continuous: Whether flux is continuous within tolerance
            - relative_error: Relative flux error
    """
    flux_left = D_left * dC_dx_left
    flux_right = D_right * dC_dx_right
    
    # For non-unity partition coefficients, we need to account for
    # the concentration discontinuity
    max_flux = max(abs(flux_left), abs(flux_right), 1e-14)
    relative_error = abs(flux_left - flux_right) / max_flux
    
    is_continuous = relative_error <= tolerance
    
    return is_continuous, relative_error
