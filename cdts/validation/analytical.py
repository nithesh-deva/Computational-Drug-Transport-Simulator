"""
Analytical solutions for validation and benchmarking.
"""

import numpy as np
from scipy import special
from typing import Tuple


def semi_infinite_domain_solution(
    x: np.ndarray,
    t: np.ndarray,
    D: float,
    C0: float
) -> np.ndarray:
    """Analytical solution for semi-infinite domain with constant boundary condition.
    
    Problem: Pure diffusion in semi-infinite domain [0, ∞)
        ∂C/∂t = D ∂²C/∂x²
        
    Initial condition: C(x,0) = 0 for x > 0
    Boundary condition: C(0,t) = C0 for all t > 0
    
    Analytical solution:
        C(x,t) = C0 · (1 - erf(x / (2·√(D·t))))
    
    where erf is the error function.
    
    Args:
        x: Spatial positions [m]
        t: Time points [s]
        D: Diffusion coefficient [m²/s]
        C0: Boundary concentration [mol/m³]
        
    Returns:
        C_analytical: Analytical solution [len(x), len(t)] [mol/m³]
    """
    C_analytical = np.zeros((len(x), len(t)))
    
    for n, time in enumerate(t):
        if time > 0:
            # Dimensionless distance
            eta = x / (2 * np.sqrt(D * time))
            # Analytical solution using complementary error function
            C_analytical[:, n] = C0 * special.erfc(eta)
        else:
            # At t=0, apply initial condition
            C_analytical[:, n] = 0.0
            C_analytical[0, n] = C0  # Boundary value
    
    return C_analytical


def infinite_domain_impulse_solution(
    x: np.ndarray,
    t: np.ndarray,
    D: float,
    M0: float
) -> np.ndarray:
    """Analytical solution for infinite domain with impulse (Dirac delta) injection.
    
    Problem: Pure diffusion in infinite domain (-∞, ∞)
        ∂C/∂t = D ∂²C/∂x²
        
    Initial condition: C(x,0) = M0·δ(x) [mol/m³]
    (impulse injection of total mass M0 at origin)
    
    Analytical solution:
        C(x,t) = (M0 / √(4π·D·t)) · exp(-x²/(4·D·t))
    
    Args:
        x: Spatial positions [m]
        t: Time points [s]
        D: Diffusion coefficient [m²/s]
        M0: Total injected mass [mol/m]
        
    Returns:
        C_analytical: Analytical solution [len(x), len(t)] [mol/m³]
    """
    C_analytical = np.zeros((len(x), len(t)))
    
    for n, time in enumerate(t):
        if time > 0:
            denominator = np.sqrt(4 * np.pi * D * time)
            exponent = -x**2 / (4 * D * time)
            C_analytical[:, n] = (M0 / denominator) * np.exp(exponent)
        else:
            # At t=0, solution is singular (Dirac delta)
            # Set very small time to approximate
            pass
    
    return C_analytical


def finite_domain_solution(
    x: np.ndarray,
    t: np.ndarray,
    D: float,
    L: float,
    C_left: float,
    C_right: float,
    C_init: float
) -> np.ndarray:
    """Analytical solution for finite domain with constant boundary conditions.
    
    Problem: Pure diffusion in finite domain [0, L]
        ∂C/∂t = D ∂²C/∂x²
        
    Initial condition: C(x,0) = C_init
    Boundary conditions: C(0,t) = C_left, C(L,t) = C_right for all t > 0
    
    Analytical solution (by separation of variables):
        C(x,t) = C_ss(x) + Σ_n A_n·exp(-λ_n·D·t)·sin(n·π·x/L)
        
    where C_ss is the steady-state solution and A_n are expansion coefficients.
    
    For C_left = C_right = 0 and C_init = 1:
        C(x,t) = (4/π) · Σ_(n odd) (1/n)·sin(n·π·x/L)·exp(-(n·π)²·D·t/L²)
    
    Args:
        x: Spatial positions [m] (should be in [0, L])
        t: Time points [s]
        D: Diffusion coefficient [m²/s]
        L: Domain length [m]
        C_left: Left boundary value [mol/m³]
        C_right: Right boundary value [mol/m³]
        C_init: Initial concentration [mol/m³]
        
    Returns:
        C_analytical: Analytical solution [len(x), len(t)] [mol/m³]
    """
    C_analytical = np.zeros((len(x), len(t)))
    
    # For symmetric case: C_left = C_right = 0, C_init = 1
    if abs(C_left) < 1e-12 and abs(C_right) < 1e-12:
        n_terms = 50  # Number of Fourier terms
        
        for n_t, time in enumerate(t):
            C_analytical[:, n_t] = 0.0
            
            for n in range(1, n_terms + 1, 2):  # Odd n only
                eigenvalue = (n * np.pi) ** 2 * D / (L ** 2)
                amplitude = (4.0 / np.pi) * (1.0 / n) * C_init
                temporal_decay = np.exp(-eigenvalue * time)
                spatial_mode = np.sin(n * np.pi * x / L)
                C_analytical[:, n_t] += amplitude * temporal_decay * spatial_mode
    else:
        # General case with non-zero boundary conditions
        # Steady state + transient
        C_ss = C_left + (C_right - C_left) * x / L
        n_terms = 50
        
        for n_t, time in enumerate(t):
            C_analytical[:, n_t] = C_ss.copy()
            
            for n in range(1, n_terms + 1):
                eigenvalue = (n * np.pi) ** 2 * D / (L ** 2)
                
                # Coefficients for transient part
                b_n = (2.0 / L) * np.trapz(
                    (C_init - C_ss) * np.sin(n * np.pi * x / L),
                    x
                )
                
                temporal_decay = np.exp(-eigenvalue * time)
                spatial_mode = np.sin(n * np.pi * x / L)
                C_analytical[:, n_t] += b_n * temporal_decay * spatial_mode
    
    return C_analytical
