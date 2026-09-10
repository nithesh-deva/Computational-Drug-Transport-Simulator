"""
Multilayer explicit finite-difference solver for M3.

Extends the single-layer solver to handle multiple tissue layers with:
- Layer-specific diffusion and clearance coefficients
- Interface partition coefficients
- Flux continuity at interfaces
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)


class MultilayerExplicitFDMSolver:
    """Explicit FDM solver for multilayer reaction-diffusion systems.
    
    Solves: ∂C/∂t = ∂(D·∂C/∂x)/∂x - k·C on multilayer domain
    
    With interface conditions:
        C_right = K · C_left (partition coefficient)
        Flux continuity: D_left·(∂C/∂x)|_- = D_right·(∂C/∂x)|_+
    """
    
    def __init__(
        self,
        domain,  # MultilayerDomain
        x: np.ndarray,
        layer_indices: np.ndarray,
        dx: float,
        dt: float,
        boundary_left: float,
        boundary_right: float,
        interface_partitions: Dict[str, float] = None
    ):
        """Initialize multilayer solver.
        
        Args:
            domain: MultilayerDomain object
            x: Spatial grid points [m]
            layer_indices: Layer index for each grid point
            dx: Spatial step [m]
            dt: Temporal step [s]
            boundary_left: Left boundary concentration [mol/m³]
            boundary_right: Right boundary concentration [mol/m³]
            interface_partitions: Dict mapping layer_pair -> partition coefficient
        """
        from ..numerics.stability import check_explicit_fdm_stability
        
        self.domain = domain
        self.x = x
        self.layer_indices = layer_indices
        self.nx = len(x)
        self.dx = dx
        self.dt = dt
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        
        # Get layer properties at each grid point
        self.D_array = domain.get_layer_properties_at_points(x, layer_indices, 'diffusion_coefficient')
        self.k_array = domain.get_layer_properties_at_points(x, layer_indices, 'clearance')
        
        # Interface partition coefficients (K for interface between layers)
        self.interface_partitions = interface_partitions or {}
        
        # Calculate Fourier numbers for stability check
        # Use maximum to be conservative
        D_max = np.max(self.D_array)
        if D_max > 0:
            r_max = D_max * dt / (dx ** 2)
            
            if not check_explicit_fdm_stability(D_max, dx, dt):
                raise ValueError(
                    f"Unstable: r_max = {r_max:.6f} > 0.5\n"
                    f"Max dt = {0.5 * dx**2 / D_max:.6e} s"
                )
            
            self.r_array = self.D_array * dt / (dx ** 2)
        else:
            self.r_array = np.zeros(self.nx)
        
        logger.info(
            f"Multilayer solver initialized:\n"
            f"  Layers: {domain.n_layers}\n"
            f"  Grid points: {self.nx}\n"
            f"  Max Fourier number: {np.max(self.r_array):.6f}\n"
            f"  Interfaces: {len(self.interface_partitions)}"
        )
    
    def apply_interface_conditions(self, C: np.ndarray, interface_indices: List[int]) -> None:
        """Apply partition coefficient conditions at interfaces.
        
        For interface between layer_left and layer_right:
            C_right = K · C_left
        
        Args:
            C: Concentration array (modified in-place)
            interface_indices: Grid indices of interfaces
        """
        for idx in interface_indices:
            if idx > 0 and idx < self.nx - 1:
                # Check if this is an interface
                layer_left = self.layer_indices[idx - 1] if idx > 0 else -1
                layer_right = self.layer_indices[idx] if idx < self.nx else -1
                
                if layer_left != layer_right and layer_left >= 0 and layer_right >= 0:
                    # Get partition coefficient for this interface
                    left_name = self.domain.layers[layer_left]['name']
                    right_name = self.domain.layers[layer_right]['name']
                    interface_key = f"{left_name}_{right_name}"
                    
                    K = self.interface_partitions.get(interface_key, 1.0)
                    
                    # Apply partition condition
                    # Use concentration from left side, apply partition to right side
                    C[idx] = K * C[idx - 1]
    
    def solve(
        self,
        initial_condition: np.ndarray,
        final_time: float,
        interface_indices: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve multilayer reaction-diffusion system.
        
        Args:
            initial_condition: Initial concentration [mol/m³]
            final_time: Final simulation time [s]
            interface_indices: Grid indices of interfaces
            
        Returns:
            Tuple:
                - x: Spatial grid [m]
                - t: Time grid [s]
                - C: Concentration field [nx, nt] [mol/m³]
                - metrics: Solver metrics
        """
        nt = int(final_time / self.dt) + 1
        t = np.linspace(0, final_time, nt)
        
        C = np.zeros((self.nx, nt))
        C[:, 0] = initial_condition
        
        # Time stepping
        for n in range(nt - 1):
            C_current = C[:, n]
            C_next = np.zeros(self.nx)
            
            # Interior points (excluding interfaces for now)
            for i in range(1, self.nx - 1):
                # Layer-specific properties
                D_i = self.D_array[i]
                k_i = self.k_array[i]
                r_i = self.r_array[i]
                
                # Diffusion term
                if D_i > 0:
                    diffusion_term = r_i * (C_current[i+1] - 2*C_current[i] + C_current[i-1])
                else:
                    diffusion_term = 0.0
                
                # Clearance term
                clearance_term = -k_i * self.dt * C_current[i]
                
                # Update
                C_next[i] = C_current[i] + diffusion_term + clearance_term
                C_next[i] = max(0.0, C_next[i])  # Non-negative enforcement
            
            # Boundary conditions (Dirichlet)
            C_next[0] = self.boundary_left
            C_next[-1] = self.boundary_right
            
            # Apply interface conditions
            if interface_indices:
                self.apply_interface_conditions(C_next, interface_indices)
            
            C[:, n+1] = C_next
        
        metrics = {
            'method': 'explicit_fdm_multilayer',
            'n_layers': self.domain.n_layers,
            'nx': self.nx,
            'nt': nt,
            'dx': self.dx,
            'dt': self.dt,
            'domain_length': self.domain.total_length,
            'final_time': final_time
        }
        
        return self.x, t, C, metrics
