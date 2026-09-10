"""
Explicit finite-difference solver for 1D reaction-diffusion equation.

M1 Governing Equation (Pure Diffusion):
    ∂C/∂t = D ∂²C/∂x²

M2 Governing Equation (Reaction-Diffusion with Clearance):
    ∂C/∂t = D ∂²C/∂x² - k·C
    
where:
    C(x,t) = concentration [mol/m³]
    D = diffusion coefficient [m²/s]
    k = first-order clearance coefficient [1/s]
    x = spatial coordinate [m]
    t = time [s]

Spatial discretization (centered differences):
    ∂²C/∂x² ≈ (C[i+1] - 2·C[i] + C[i-1]) / Δx²

Temporal discretization (forward Euler):
    ∂C/∂t ≈ (C[i,n+1] - C[i,n]) / Δt

Update formula (M1 - pure diffusion):
    C[i,n+1] = C[i,n] + r·(C[i+1,n] - 2·C[i,n] + C[i-1,n])

Update formula (M2 - with clearance):
    C[i,n+1] = C[i,n] + r·(C[i+1,n] - 2·C[i,n] + C[i-1,n]) - k·Δt·C[i,n]
             = C[i,n]·(1 - k·Δt) + r·(C[i+1,n] - 2·C[i,n] + C[i-1,n])
    
where r = D·Δt/Δx² (Fourier number)

Stability requirement: r ≤ 0.5 (diffusive stability)

Mass conservation:
    - M1 (k=0): Total mass conserved at zero-flux boundaries
    - M2 (k>0): Mass decays exponentially: dM/dt = -k·M
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any
from ..numerics.stability import check_explicit_fdm_stability, get_max_stable_dt

logger = logging.getLogger(__name__)


class ExplicitFDMSolver:
    """Explicit finite-difference solver for 1D diffusion.
    
    Attributes:
        domain_length: Total domain length [m]
        diffusion_coeff: Diffusion coefficient [m²/s]
        dx: Spatial step size [m]
        dt: Temporal step size [s]
        boundary_left: Left boundary condition (Dirichlet value)
        boundary_right: Right boundary condition (Dirichlet value)
    """
    
    def __init__(
        self,
        domain_length: float,
        diffusion_coeff: float,
        dx: float,
        dt: float,
        boundary_left: float,
        boundary_right: float,
        clearance: float = 0.0
    ):
        """Initialize solver.
        
        Args:
            domain_length: Total tissue thickness [m]
            diffusion_coeff: Diffusion coefficient [m²/s]
            dx: Spatial discretization [m]
            dt: Temporal discretization [s]
            boundary_left: Left boundary value (Dirichlet) [mol/m³]
            boundary_right: Right boundary value (Dirichlet) [mol/m³]
            clearance: First-order clearance coefficient [1/s]
        """
        self.domain_length = domain_length
        self.diffusion_coeff = diffusion_coeff
        self.dx = dx
        self.dt = dt
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.clearance = clearance
        
        # Spatial grid
        self.x = np.arange(0, domain_length + dx/2, dx)
        self.nx = len(self.x)
        
        # Fourier number
        self.r = diffusion_coeff * dt / (dx ** 2)
        
        # Stability check
        if not check_explicit_fdm_stability(diffusion_coeff, dx, dt):
            raise ValueError(
                f"Unstable configuration: r = {self.r:.6f} > 0.5\n"
                f"Max stable dt = {get_max_stable_dt(diffusion_coeff, dx):.6e} s"
            )
        
        logger.info(f"Solver initialized: nx={self.nx}, r={self.r:.6f}, clearance={clearance:.6e} 1/s")
    
    def solve(
        self,
        initial_condition: np.ndarray,
        final_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve diffusion equation from t=0 to t=final_time.
        
        Args:
            initial_condition: Initial concentration at all grid points [mol/m³]
            final_time: Final simulation time [s]
            
        Returns:
            Tuple containing:
                - x: Spatial grid points [m]
                - t: Time points [s]
                - C: Concentration field [nx, nt] [mol/m³]
                - metrics: Dictionary with solver metrics
        """
        nt = int(final_time / self.dt) + 1
        t = np.linspace(0, final_time, nt)
        
        # Initialize concentration array
        C = np.zeros((self.nx, nt))
        C[:, 0] = initial_condition
        
        # Time stepping
        for n in range(nt - 1):
            C_current = C[:, n]
            C_next = np.zeros(self.nx)
            
            # Interior points
            for i in range(1, self.nx - 1):
                diffusion_term = self.r * (C_current[i+1] - 2*C_current[i] + C_current[i-1])
                clearance_term = -self.clearance * self.dt * C_current[i]
                C_next[i] = C_current[i] + diffusion_term + clearance_term
                C_next[i] = max(0.0, C_next[i])  # Ensure non-negative
            
            # Boundary conditions (Dirichlet)
            C_next[0] = self.boundary_left
            C_next[-1] = self.boundary_right
            
            C[:, n+1] = C_next
        
        metrics = {
            'method': 'explicit_fdm',
            'nx': self.nx,
            'nt': nt,
            'dx': self.dx,
            'dt': self.dt,
            'r': self.r,
            'diffusion_coeff': self.diffusion_coeff,
            'clearance': self.clearance,
            'domain_length': self.domain_length,
            'final_time': final_time,
            'stability_stable': True
        }
        
        return self.x, t, C, metrics
