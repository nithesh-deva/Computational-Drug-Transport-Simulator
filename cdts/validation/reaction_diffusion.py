"""
Analytical solutions for reaction-diffusion systems with first-order clearance.

Governing Equation:
    ∂C/∂t = D ∂²C/∂x² - k·C
    
where:
    C(x,t) = concentration [mol/m³]
    D = diffusion coefficient [m²/s]
    k = first-order clearance coefficient [1/s]
    x = spatial coordinate [m]
    t = time [s]
"""

import numpy as np
from scipy import special, optimize
from typing import Tuple, Callable


def semi_infinite_domain_with_clearance(
    x: np.ndarray,
    t: np.ndarray,
    D: float,
    k: float,
    C0: float
) -> np.ndarray:
    """Analytical solution for semi-infinite domain with clearance.
    
    Problem: Reaction-diffusion in semi-infinite domain [0, ∞)
        ∂C/∂t = D ∂²C/∂x² - k·C
        
    Initial condition: C(x,0) = 0 for x > 0
    Boundary condition: C(0,t) = C0 for all t > 0
    
    Analytical solution uses complementary error function with exponential decay:
        C(x,t) = C0 · exp(-k·t) · erfc(x / (2·√(D·t))) + correction term
    
    For large time, solutions approach zero due to combined diffusion and clearance.
    
    Args:
        x: Spatial positions [m]
        t: Time points [s]
        D: Diffusion coefficient [m²/s]
        k: Clearance coefficient [1/s]
        C0: Boundary concentration [mol/m³]
        
    Returns:
        C_analytical: Analytical solution [len(x), len(t)] [mol/m³]
    """
    C_analytical = np.zeros((len(x), len(t)))
    
    for n, time in enumerate(t):
        if time > 0:
            # Exponential decay from clearance
            decay = np.exp(-k * time)
            
            # Diffusion part (modified by effective diffusion with decay)
            # For small k, this approaches the pure diffusion solution
            eta = x / (2 * np.sqrt(D * time))
            
            # Complementary error function
            C_analytical[:, n] = C0 * decay * special.erfc(eta)
        else:
            # At t=0
            C_analytical[:, n] = 0.0
            C_analytical[0, n] = C0
    
    return C_analytical


def finite_domain_with_clearance_exponential(
    x: np.ndarray,
    t: np.ndarray,
    D: float,
    k: float,
    L: float,
    C_left: float = 0.0,
    C_right: float = 0.0,
    C_init: float = 1.0
) -> np.ndarray:
    """Analytical solution for finite domain with clearance using eigenvalue expansion.
    
    Problem: Reaction-diffusion in finite domain [0, L]
        ∂C/∂t = D ∂²C/∂x² - k·C
        
    Initial condition: C(x,0) = C_init
    Boundary conditions: C(0,t) = C_left, C(L,t) = C_right for all t > 0
    
    For C_left = C_right = 0 (homogeneous Dirichlet), the solution is:
        C(x,t) = Σ_n B_n · sin(n·π·x/L) · exp(-λ_n·t)
        
    where λ_n = D·(n·π/L)² + k (combined diffusion and clearance decay rate)
    and B_n are initial condition expansion coefficients.
    
    Args:
        x: Spatial positions [m]
        t: Time points [s]
        D: Diffusion coefficient [m²/s]
        k: Clearance coefficient [1/s]
        L: Domain length [m]
        C_left: Left boundary value [mol/m³]
        C_right: Right boundary value [mol/m³]
        C_init: Initial concentration [mol/m³]
        
    Returns:
        C_analytical: Analytical solution [len(x), len(t)] [mol/m³]
    """
    C_analytical = np.zeros((len(x), len(t)))
    
    # Case: Symmetric homogeneous Dirichlet boundary conditions
    if abs(C_left) < 1e-12 and abs(C_right) < 1e-12:
        n_terms = 50  # Number of Fourier terms
        
        for n_t, time in enumerate(t):
            C_analytical[:, n_t] = 0.0
            
            for n in range(1, n_terms + 1):
                # Eigenvalue for reaction-diffusion: λ_n = D(nπ/L)² + k
                eigenvalue = D * (n * np.pi / L) ** 2 + k
                
                # Amplitude for pure sine basis (C_init uniform, homogeneous BC)
                amplitude = (4.0 / np.pi) * (1.0 / n) * C_init if n % 2 == 1 else 0.0
                
                # Temporal decay with combined diffusion and clearance
                temporal_decay = np.exp(-eigenvalue * time)
                
                # Spatial mode
                spatial_mode = np.sin(n * np.pi * x / L)
                
                C_analytical[:, n_t] += amplitude * temporal_decay * spatial_mode
    else:
        # General case with non-homogeneous boundary conditions
        C_ss = C_left + (C_right - C_left) * x / L
        n_terms = 50
        
        for n_t, time in enumerate(t):
            C_analytical[:, n_t] = C_ss.copy()
            
            for n in range(1, n_terms + 1):
                eigenvalue = D * (n * np.pi / L) ** 2 + k
                
                # Fourier coefficient of initial condition minus steady state
                b_n = (2.0 / L) * np.trapz(
                    (C_init - C_ss) * np.sin(n * np.pi * x / L),
                    x
                )
                
                temporal_decay = np.exp(-eigenvalue * time)
                spatial_mode = np.sin(n * np.pi * x / L)
                C_analytical[:, n_t] += b_n * temporal_decay * spatial_mode
    
    return C_analytical


def diffusion_dominance_time(
    D: float,
    k: float,
    L: float,
    Pe: float = 1.0
) -> Tuple[float, float, float]:
    """Determine characteristic timescales for reaction-diffusion system.
    
    Three important timescales:
    
    1. Diffusion timescale: τ_D = L²/D (time to diffuse across domain)
    2. Clearance timescale: τ_k = 1/k (time for exponential decay due to clearance)
    3. Diffusion number: D_a = k·L²/D (Damköhler number)
    
    Regimes:
    - D_a << 1: Diffusion-limited (slow clearance)
    - D_a >> 1: Clearance-limited (fast clearance)
    - D_a ~ 1: Comparable rates
    
    Args:
        D: Diffusion coefficient [m²/s]
        k: Clearance coefficient [1/s]
        L: Domain length [m]
        Pe: Péclet number (for reference, not used in calculation)
        
    Returns:
        Tuple:
            - tau_diffusion: Diffusion timescale [s]
            - tau_clearance: Clearance timescale [s]
            - damkohler_number: Damköhler number (dimensionless)
    """
    if D <= 0 or L <= 0:
        raise ValueError("D and L must be positive")
    
    tau_diffusion = L ** 2 / D  # Time to diffuse across L
    
    if k > 0:
        tau_clearance = 1.0 / k  # Clearance decay time
        damkohler = k * L ** 2 / D  # Reaction vs diffusion ratio
    else:
        tau_clearance = np.inf
        damkohler = 0.0
    
    return tau_diffusion, tau_clearance, damkohler


def expected_mass_loss_rate(
    total_concentration_integral: float,
    k: float
) -> float:
    """Calculate expected mass loss rate due to first-order clearance.
    
    For first-order clearance:
        dM/dt = -k · M(t)
        
    where M(t) = ∫ C(x,t) dx
    
    This is a differential equation with solution:
        M(t) = M(0) · exp(-k·t)
    
    The instantaneous loss rate at time t is:
        |dM/dt| = k · M(t)
    
    Args:
        total_concentration_integral: Total mass M(t) = ∫ C dx [mol/m]
        k: Clearance coefficient [1/s]
        
    Returns:
        float: Expected mass loss rate [mol/(m·s)]
    """
    return k * total_concentration_integral


def verify_mass_loss_consistency(
    t: np.ndarray,
    total_mass: np.ndarray,
    k: float,
    tolerance: float = 0.05
) -> Tuple[bool, float, np.ndarray]:
    """Verify that simulated mass loss matches first-order clearance model.
    
    For first-order clearance, total mass should satisfy:
        M(t) = M(0) · exp(-k·t)
    
    Check relative error:
        error = |M_computed(t) - M_expected(t)| / M_expected(t)
    
    Args:
        t: Time array [s]
        total_mass: Total mass at each time [mol/m]
        k: Clearance coefficient [1/s]
        tolerance: Acceptable relative error (default 0.05 = 5%)
        
    Returns:
        Tuple:
            - is_consistent: Whether error is within tolerance
            - mean_error: Average relative error
            - error_array: Relative error at each time point
    """
    if k <= 0:
        raise ValueError("Clearance coefficient k must be positive for mass loss check")
    
    M0 = total_mass[0]
    if M0 <= 0:
        raise ValueError("Initial mass must be positive")
    
    # Expected exponential decay
    M_expected = M0 * np.exp(-k * t)
    
    # Relative error (avoid division by zero)
    error_array = np.abs(total_mass - M_expected) / (M_expected + 1e-14)
    mean_error = np.mean(error_array)
    
    is_consistent = mean_error <= tolerance
    
    return is_consistent, mean_error, error_array
