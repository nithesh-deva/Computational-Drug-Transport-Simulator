"""
M6 Verification Benchmark: Analytical Solution Framework

This module implements a simplified reaction-diffusion problem with known
analytical solution for rigorous numerical method validation.

Problem: 1D finite domain, reaction-diffusion with homogeneous Dirichlet BCs

    ∂C/∂t = D ∂²C/∂x² - k·C    in [0, L]
    
    Initial condition: C(x,0) = C_init
    Boundary conditions: C(0,t) = 0, C(L,t) = 0 for all t > 0

This is a simplified version of M2 that admits known analytical solutions
for validation. The problem combines:
- Spatial diffusion (D)
- First-order clearance (k)
- Finite domain with zero boundary conditions

Analytical Solution (Eigenvalue Expansion):

    C(x,t) = Σ_n B_n · sin(n·π·x/L) · exp(-λ_n·t)
    
    where:
        λ_n = D·(n·π/L)² + k  (combined diffusion + clearance decay rate)
        B_n = expansion coefficients from initial condition

For uniform initial condition C(x,0) = C_init:
    B_n = (4·C_init)/(n·π)  for odd n
    B_n = 0  for even n
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AnalyticalBenchmark:
    """Reference solution for verification benchmark problem."""
    
    def __init__(
        self,
        domain_length: float,
        diffusion_coeff: float,
        clearance: float,
        initial_concentration: float,
        n_fourier_terms: int = 100
    ):
        """Initialize benchmark problem.
        
        Args:
            domain_length: Domain length L [m]
            diffusion_coeff: Diffusion coefficient D [m²/s]
            clearance: First-order clearance k [1/s]
            initial_concentration: Initial uniform concentration C_init [mol/m³]
            n_fourier_terms: Number of Fourier terms in eigenvalue expansion
        """
        self.L = domain_length
        self.D = diffusion_coeff
        self.k = clearance
        self.C_init = initial_concentration
        self.n_fourier = n_fourier_terms
        
        # Characteristic parameters
        self.tau_diffusion = domain_length ** 2 / diffusion_coeff
        self.tau_clearance = 1.0 / clearance if clearance > 0 else np.inf
        self.damkohler = (clearance * domain_length ** 2) / diffusion_coeff
        
        logger.info(
            f"Benchmark initialized:\n"
            f"  Domain: [0, {self.L:.4e}] m\n"
            f"  D = {self.D:.4e} m²/s, k = {self.k:.4e} 1/s\n"
            f"  Diffusion timescale: {self.tau_diffusion:.4e} s\n"
            f"  Clearance timescale: {self.tau_clearance:.4e} s\n"
            f"  Damköhler number: {self.damkohler:.4e}\n"
            f"  Fourier terms: {self.n_fourier}"
        )
    
    def solution(
        self,
        x: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """Compute analytical solution.
        
        Args:
            x: Spatial positions [m], shape (nx,)
            t: Time points [s], shape (nt,)
            
        Returns:
            C_analytical: Concentration field [nx, nt] [mol/m³]
        """
        x = np.asarray(x)
        t = np.asarray(t)
        
        nx = len(x) if x.ndim == 1 else x.shape[0]
        nt = len(t) if t.ndim == 1 else t.shape[0]
        
        C = np.zeros((nx, nt))
        
        for n_time, time_val in enumerate(t):
            for n in range(1, self.n_fourier + 1):
                # Eigenvalue: λ_n = D·(n·π/L)² + k
                eigenvalue = self.D * (n * np.pi / self.L) ** 2 + self.k
                
                # Amplitude for odd n (uniform initial condition)
                if n % 2 == 1:
                    amplitude = (4.0 * self.C_init) / (n * np.pi)
                else:
                    amplitude = 0.0
                
                # Temporal decay
                temporal_decay = np.exp(-eigenvalue * time_val)
                
                # Spatial mode: sin(n·π·x/L)
                spatial_mode = np.sin(n * np.pi * x / self.L)
                
                C[:, n_time] += amplitude * temporal_decay * spatial_mode
        
        return C
    
    def solution_at_time(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute analytical solution at single time.
        
        Args:
            x: Spatial positions [m]
            t: Time point [s]
            
        Returns:
            C: Concentration at all x [mol/m³]
        """
        t_array = np.array([t])
        C = self.solution(x, t_array)
        return C[:, 0]
    
    def total_mass(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute total integrated mass M(t) = ∫_0^L C(x,t) dx.
        
        For this benchmark, analytical mass is:
            M(t) = Σ_n (8·C_init/(n²·π²)) · exp(-λ_n·t)
            
        for odd n, summed over all odd n.
        
        Args:
            x: Spatial grid [m]
            t: Time array [s]
            
        Returns:
            total_mass: Integrated mass at each time [mol/m]
        """
        t = np.asarray(t)
        M = np.zeros_like(t)
        
        for n in range(1, self.n_fourier + 1, 2):  # Odd n only
            eigenvalue = self.D * (n * np.pi / self.L) ** 2 + self.k
            
            # Integral of sin(n·π·x/L) over [0,L] is 2L/(n·π)
            # So: ∫ B_n·sin(n·π·x/L) dx = B_n · 2L/(n·π)
            # where B_n = (4·C_init)/(n·π)
            # Result: (8·C_init·L)/(n²·π²)
            integral_amplitude = (8.0 * self.C_init * self.L) / (n ** 2 * np.pi ** 2)
            
            M += integral_amplitude * np.exp(-eigenvalue * t)
        
        return M
    
    def total_mass_rate(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute total mass loss rate dM/dt.
        
        Analytically: dM/dt = Σ_n (-λ_n)·(8·C_init/(n²·π²))·exp(-λ_n·t)
        
        Args:
            x: Spatial grid [m]
            t: Time array [s]
            
        Returns:
            mass_rate: Rate of mass change [mol/(m·s)]
        """
        t = np.asarray(t)
        dM_dt = np.zeros_like(t)
        
        for n in range(1, self.n_fourier + 1, 2):  # Odd n only
            eigenvalue = self.D * (n * np.pi / self.L) ** 2 + self.k
            integral_amplitude = (8.0 * self.C_init * self.L) / (n ** 2 * np.pi ** 2)
            
            dM_dt -= eigenvalue * integral_amplitude * np.exp(-eigenvalue * t)
        
        return dM_dt


def create_benchmark_problem(
    regime: str = 'moderate'
) -> Tuple[AnalyticalBenchmark, Dict]:
    """Create predefined benchmark problem configurations.
    
    Three regimes for different physical scenarios:
    
    1. Diffusion-dominated (D_a << 1):
       - Slow clearance, diffusion governs
       - Easy to resolve numerically
       
    2. Moderate (D_a ~ 1):
       - Comparable diffusion and clearance
       - Reasonable numerical challenge
       
    3. Clearance-dominated (D_a >> 1):
       - Fast clearance dominates
       - Steep gradients, harder to resolve
    
    Args:
        regime: 'diffusion', 'moderate', or 'clearance'
        
    Returns:
        Tuple:
            - benchmark: AnalyticalBenchmark instance
            - params: Parameter dictionary for reference
    """
    if regime == 'diffusion':
        # Diffusion-dominated: slow clearance
        L = 0.001  # 1 mm
        D = 1.0e-10  # m²/s
        k = 0.0001  # 1/s, slow clearance
        C_init = 1.0  # mol/m³
        
    elif regime == 'moderate':
        # Moderate: D_a ~ 1
        L = 0.001  # 1 mm
        D = 1.0e-10  # m²/s
        k = 0.01  # 1/s, moderate clearance
        C_init = 1.0  # mol/m³
        
    elif regime == 'clearance':
        # Clearance-dominated: fast clearance
        L = 0.001  # 1 mm
        D = 1.0e-10  # m²/s
        k = 1.0  # 1/s, fast clearance
        C_init = 1.0  # mol/m³
        
    else:
        raise ValueError(f"Unknown regime: {regime}")
    
    benchmark = AnalyticalBenchmark(
        domain_length=L,
        diffusion_coeff=D,
        clearance=k,
        initial_concentration=C_init,
        n_fourier_terms=100
    )
    
    params = {
        'regime': regime,
        'domain_length': L,
        'diffusion_coeff': D,
        'clearance': k,
        'initial_concentration': C_init,
        'damkohler_number': benchmark.damkohler,
        'tau_diffusion': benchmark.tau_diffusion,
        'tau_clearance': benchmark.tau_clearance,
    }
    
    return benchmark, params
