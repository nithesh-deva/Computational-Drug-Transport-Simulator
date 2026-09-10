"""
Crank-Nicolson implicit solver for 1D reaction-diffusion systems (M4).

The Crank-Nicolson method is a second-order implicit scheme that is
unconditionally stable for parabolic PDEs, allowing larger time steps
than explicit methods without stability constraints.

Governing Equation:
    ∂C/∂t = D ∂²C/∂x² - k·C

Crank-Nicolson Discretization:
    [I + (r/2)·L - (k·Δt/2)·I]·C^(n+1) = [I - (r/2)·L + (k·Δt/2)·I]·C^n
    
where L is the discrete Laplacian (finite differences):
    L[i,j] = -2 (diagonal), 1 (off-diagonals)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any
from scipy.linalg import solve_banded

logger = logging.getLogger(__name__)


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system using Thomas algorithm (TDMA).
    
    Solves: a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]
    
    More efficient than general LU for tridiagonal systems.
    
    Args:
        a: Lower diagonal [n] (a[0] unused)
        b: Main diagonal [n]
        c: Upper diagonal [n] (c[n-1] unused)
        d: Right-hand side [n]
        
    Returns:
        x: Solution vector [n]
    """
    n = len(d)
    x = np.zeros(n)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i-1]
        if abs(denominator) < 1e-14:
            raise ValueError(f"Singular tridiagonal matrix at row {i}")
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator
    
    # Back substitution
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


class CrankNicolsonSolver:
    """Crank-Nicolson implicit solver for 1D reaction-diffusion.
    
    Features:
    - Unconditionally stable (no r ≤ 0.5 constraint)
    - Second-order accurate in space and time
    - Tridiagonal system solve per time step
    - Layer-specific properties support
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
        """Initialize Crank-Nicolson solver.
        
        Args:
            domain_length: Total domain length [m]
            diffusion_coeff: Diffusion coefficient [m²/s]
            dx: Spatial step [m]
            dt: Temporal step [s]
            boundary_left: Left boundary (Dirichlet) [mol/m³]
            boundary_right: Right boundary (Dirichlet) [mol/m³]
            clearance: First-order clearance [1/s]
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
        
        # Fourier number (for information only, not stability constraint)
        self.r = diffusion_coeff * dt / (dx ** 2)
        
        logger.info(
            f"Crank-Nicolson solver initialized:\n"
            f"  Grid points: {self.nx}\n"
            f"  Fourier number r: {self.r:.6f} (unconditionally stable)\n"
            f"  Clearance: {clearance:.6e} 1/s"
        )
    
    def solve(
        self,
        initial_condition: np.ndarray,
        final_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve using Crank-Nicolson implicit scheme.
        
        Args:
            initial_condition: Initial concentration [mol/m³]
            final_time: Final simulation time [s]
            
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
        
        # Precalculate coefficients
        r = self.r
        k = self.clearance
        
        # LHS coefficients: [I - (r/2)·L + (k·Δt/2)·I]
        # Main diagonal: 1 - (r/2)·(-2) + (k·Δt/2) = 1 + r + k·Δt/2
        lhs_main = np.ones(self.nx) + r + (k * self.dt / 2)
        # Off-diagonals: -(r/2)·1 = -r/2
        lhs_off = np.ones(self.nx - 1) * (-r / 2)
        
        # RHS coefficients: [I + (r/2)·L - (k·Δt/2)·I]
        # Main diagonal: 1 + (r/2)·(-2) - (k·Δt/2) = 1 - r - k·Δt/2
        rhs_main = np.ones(self.nx) - r - (k * self.dt / 2)
        # Off-diagonals: (r/2)·1 = r/2
        rhs_off = np.ones(self.nx - 1) * (r / 2)
        
        # Time stepping
        for n in range(nt - 1):
            C_current = C[:, n]
            
            # Compute RHS: [I - (r/2)·L + (k·Δt/2)·I]·C^n
            rhs = np.zeros(self.nx)
            rhs[0] = self.boundary_left  # BC at left
            rhs[-1] = self.boundary_right  # BC at right
            
            # Interior points
            for i in range(1, self.nx - 1):
                rhs[i] = (rhs_main[i] * C_current[i] + 
                         rhs_off[i-1] * C_current[i-1] + 
                         rhs_off[i] * C_current[i+1])
            
            # Solve tridiagonal system: [I + (r/2)·L - (k·Δt/2)·I]·C^(n+1) = rhs
            # System: a·x[i-1] + b·x[i] + c·x[i+1] = d[i]
            a = np.zeros(self.nx)
            a[1:] = lhs_off  # Lower diagonal
            b = lhs_main.copy()  # Main diagonal
            c = np.zeros(self.nx)
            c[:-1] = lhs_off  # Upper diagonal
            d = rhs.copy()
            
            # Enforce boundary conditions
            b[0] = 1.0
            b[-1] = 1.0
            c[0] = 0.0
            a[-1] = 0.0
            d[0] = self.boundary_left
            d[-1] = self.boundary_right
            
            # Solve tridiagonal system
            C_next = thomas_algorithm(a, b, c, d)
            C_next = np.maximum(C_next, 0.0)  # Non-negative enforcement
            
            C[:, n+1] = C_next
        
        metrics = {
            'method': 'crank_nicolson',
            'nx': self.nx,
            'nt': nt,
            'dx': self.dx,
            'dt': self.dt,
            'r': self.r,
            'diffusion_coeff': self.diffusion_coeff,
            'clearance': self.clearance,
            'domain_length': self.domain_length,
            'final_time': final_time,
            'stability': 'unconditionally_stable'
        }
        
        return self.x, t, C, metrics

