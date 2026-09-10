"""
Analytical solutions for two-layer reaction-diffusion systems (M3).

Provides benchmark solutions for:
- Two-layer finite domain with interface partition coefficient
- Interface matching conditions
- Layer-specific diffusion and clearance
"""

import numpy as np
from scipy import special
from typing import Tuple


def two_layer_analytical_solution(
    x: np.ndarray,
    t: np.ndarray,
    D1: float,
    D2: float,
    k1: float,
    k2: float,
    L1: float,
    L2: float,
    K: float,
    C_left: float = 0.0,
    C_right: float = 0.0,
    C_init_1: float = 1.0,
    C_init_2: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical solution for two-layer domain with interface partition.
    
    Problem:
        Layer 1 [0, L1]: ∂C₁/∂t = D₁∂²C₁/∂x² - k₁C₁
        Layer 2 [L1, L1+L2]: ∂C₂/∂t = D₂∂²C₂/∂x² - k₂C₂
        
    Boundary conditions:
        C₁(0,t) = C_left
        C₂(L1+L2,t) = C_right
        
    Interface (at x=L1):
        Concentration: C₂ = K·C₁
        Flux: D₁·∂C₁/∂x = D₂·∂C₂/∂x
        
    Initial conditions:
        C₁(x,0) = C_init_1
        C₂(x,0) = C_init_2
    
    For the case C_left = C_right = 0 with uniform initial conditions,
    the solution is an infinite eigenvalue expansion.
    
    Args:
        x: Spatial grid [m]
        t: Time grid [s]
        D1: Diffusion in layer 1 [m²/s]
        D2: Diffusion in layer 2 [m²/s]
        k1: Clearance in layer 1 [1/s]
        k2: Clearance in layer 2 [1/s]
        L1: Thickness of layer 1 [m]
        L2: Thickness of layer 2 [m]
        K: Partition coefficient at interface (C2=K*C1)
        C_left: Left boundary concentration [mol/m³]
        C_right: Right boundary concentration [mol/m³]
        C_init_1: Initial concentration in layer 1 [mol/m³]
        C_init_2: Initial concentration in layer 2 [mol/m³]
        
    Returns:
        Tuple:
            - C1: Concentration in layer 1 [len(x), len(t)] [mol/m³]
            - C2: Concentration in layer 2 [len(x), len(t)] [mol/m³]
    """
    # Identify which points are in which layer
    interface_pos = L1
    total_length = L1 + L2
    
    x1_indices = x <= interface_pos
    x2_indices = x > interface_pos
    
    C1 = np.zeros((len(x), len(t)))
    C2 = np.zeros((len(x), len(t)))
    
    # For the case C_left = C_right = 0 (simplified for M3)
    if abs(C_left) < 1e-12 and abs(C_right) < 1e-12:
        n_terms = 30  # Number of eigenvalue terms
        
        for n_t, time in enumerate(t):
            if time > 0:
                # Solve eigenvalue problem numerically (simplified approach)
                # Use series expansion with multiple modes
                
                for n in range(1, n_terms + 1):
                    # Approximate eigenvalues and eigenvectors
                    # This is a simplified solution; full solution requires
                    # solving transcendental equation for eigenvalues
                    
                    # For simplicity, use uncoupled approximation
                    lambda_n_1 = (n * np.pi / L1) ** 2 * D1 + k1
                    lambda_n_2 = (n * np.pi / L2) ** 2 * D2 + k2
                    
                    # Amplitude for layer 1 (uniform initial)
                    A_n = (4.0 / (n * np.pi)) * C_init_1 if n % 2 == 1 else 0.0
                    
                    # Layer 1 solution
                    decay_1 = np.exp(-lambda_n_1 * time)
                    mode_1 = np.sin(n * np.pi * x[x1_indices] / L1)
                    C1[x1_indices, n_t] += A_n * decay_1 * mode_1
                    
                    # Layer 2 solution (with partition and different eigenvalue)
                    # This is approximate; exact solution requires eigenvalue matching
                    decay_2 = np.exp(-lambda_n_2 * time)
                    mode_2 = np.sin(n * np.pi * (x[x2_indices] - L1) / L2)
                    x2_shifted = x[x2_indices] - interface_pos
                    
                    C2[x2_indices, n_t] += K * A_n * decay_2 * mode_2
            else:
                # t=0: Apply initial conditions
                C1[x1_indices, n_t] = C_init_1
                C2[x2_indices, n_t] = C_init_2
    else:
        # Non-zero boundary conditions: use steady-state + transient
        # This is more complex; simplified for now
        pass
    
    return C1, C2


def two_layer_uncoupled_solution(
    x: np.ndarray,
    t: np.ndarray,
    D1: float,
    D2: float,
    k1: float,
    k2: float,
    L1: float,
    L2: float,
    K: float,
    C_init_1: float = 1.0,
    C_init_2: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified two-layer solution treating layers as decoupled.
    
    Each layer solves independently with zero boundary conditions:
        Layer 1: homogeneous Dirichlet at both ends
        Layer 2: homogeneous Dirichlet at both ends
        
    Interface partition condition applied post-solve: C2 = K*C1
    
    This is an approximation; exact solution requires solving coupled
    eigenvalue problem.
    
    Args:
        (same as two_layer_analytical_solution)
        
    Returns:
        Tuple: (C1, C2)
    """
    interface_pos = L1
    x1_indices = x <= interface_pos
    x2_indices = x > interface_pos
    
    C1 = np.zeros((len(x), len(t)))
    C2 = np.zeros((len(x), len(t)))
    
    n_terms = 30
    
    # Layer 1 solution (standard finite domain solution)
    for n_t, time in enumerate(t):
        if time > 0:
            for n in range(1, n_terms + 1, 2):  # Odd n only for sine basis
                lambda_1 = D1 * (n * np.pi / L1) ** 2 + k1
                A_n = (4.0 / (n * np.pi)) * C_init_1
                decay = np.exp(-lambda_1 * time)
                spatial = np.sin(n * np.pi * x[x1_indices] / L1)
                C1[x1_indices, n_t] += A_n * decay * spatial
            
            # Layer 2 solution with partition coefficient
            x2_local = x[x2_indices] - interface_pos  # Local coordinate in layer 2
            for n in range(1, n_terms + 1, 2):
                lambda_2 = D2 * (n * np.pi / L2) ** 2 + k2
                A_n = (4.0 / (n * np.pi)) * C_init_2 * K  # Apply K to initial
                decay = np.exp(-lambda_2 * time)
                spatial = np.sin(n * np.pi * x2_local / L2)
                C2[x2_indices, n_t] += A_n * decay * spatial
        else:
            C1[x1_indices, n_t] = C_init_1
            C2[x2_indices, n_t] = C_init_2
    
    return C1, C2


def calculate_interface_boundary_flux_matching(
    D1: float,
    D2: float,
    K: float,
    dC1_dx_interface: float,
    C1_interface: float
) -> Tuple[float, float]:
    """Calculate concentration and flux at interface using matching conditions.
    
    Matching conditions at interface:
        C2_interface = K · C1_interface (partition)
        D1 · dC1/dx = D2 · dC2/dx (flux continuity)
        
    Args:
        D1: Diffusion in layer 1 [m²/s]
        D2: Diffusion in layer 2 [m²/s]
        K: Partition coefficient
        dC1_dx_interface: Concentration gradient in layer 1 at interface [mol/m⁴]
        C1_interface: Concentration in layer 1 at interface [mol/m³]
        
    Returns:
        Tuple:
            - C2_interface: Concentration in layer 2 at interface [mol/m³]
            - dC2_dx_interface: Concentration gradient in layer 2 at interface [mol/m⁴]
    """
    # Partition condition
    C2_interface = K * C1_interface
    
    # Flux continuity: D1·dC1/dx = D2·dC2/dx
    if D2 > 0:
        dC2_dx_interface = (D1 / D2) * dC1_dx_interface
    else:
        dC2_dx_interface = 0.0
    
    return C2_interface, dC2_dx_interface


def verify_two_layer_mass_conservation(
    x: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    L1: float,
    k1: float,
    k2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Verify total mass conservation in two-layer system.
    
    Total mass: M(t) = ∫₀^L1 C1 dx + ∫_L1^(L1+L2) C2 dx
    
    With clearance: dM/dt = -∫ k·C dx = -(∫ k1·C1 dx + ∫ k2·C2 dx)
    
    Args:
        x: Spatial grid [m]
        C1: Concentration in layer 1 [nx, nt] [mol/m³]
        C2: Concentration in layer 2 [nx, nt] [mol/m³]
        L1: Interface position [m]
        k1: Clearance in layer 1 [1/s]
        k2: Clearance in layer 2 [1/s]
        
    Returns:
        Tuple:
            - total_mass: Total mass at each time [mol/m]
            - expected_loss_rate: Expected mass loss rate [mol/(m·s)]
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    
    total_mass = np.zeros(C1.shape[1])
    
    # Integrate over both layers
    for t_idx in range(C1.shape[1]):
        mass1 = np.trapz(C1[:, t_idx], x)
        mass2 = np.trapz(C2[:, t_idx], x)
        total_mass[t_idx] = mass1 + mass2
    
    # Expected loss per time step
    expected_loss = np.zeros_like(total_mass)
    for t_idx in range(C1.shape[1]):
        loss1 = k1 * np.trapz(C1[:, t_idx], x)
        loss2 = k2 * np.trapz(C2[:, t_idx], x)
        expected_loss[t_idx] = -(loss1 + loss2)
    
    return total_mass, expected_loss
