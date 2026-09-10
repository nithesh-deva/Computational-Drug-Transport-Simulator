"""
Error metrics and validation analysis.
"""

import numpy as np
from typing import Dict, Tuple


def rmse(C_numerical: np.ndarray, C_analytical: np.ndarray) -> float:
    """Root Mean Squared Error.
    
    RMSE = √(1/N · Σ(C_num - C_ana)²)
    
    Args:
        C_numerical: Numerical solution [mol/m³]
        C_analytical: Analytical solution [mol/m³]
        
    Returns:
        float: RMSE value [mol/m³]
    """
    diff = C_numerical - C_analytical
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def relative_error(C_numerical: np.ndarray, C_analytical: np.ndarray) -> np.ndarray:
    """Pointwise relative error.
    
    E_rel = |C_num - C_ana| / (|C_ana| + ε)
    
    where ε is a small threshold to avoid division by zero.
    
    Args:
        C_numerical: Numerical solution [mol/m³]
        C_analytical: Analytical solution [mol/m³]
        
    Returns:
        np.ndarray: Relative error at each point
    """
    epsilon = 1e-14  # Threshold for near-zero concentrations
    denominator = np.abs(C_analytical) + epsilon
    relative_err = np.abs(C_numerical - C_analytical) / denominator
    return relative_err


def maximum_absolute_error(C_numerical: np.ndarray, C_analytical: np.ndarray) -> float:
    """Maximum absolute error across domain.
    
    E_max = max|C_num - C_ana|
    
    Args:
        C_numerical: Numerical solution [mol/m³]
        C_analytical: Analytical solution [mol/m³]
        
    Returns:
        float: Maximum absolute error [mol/m³]
    """
    return np.max(np.abs(C_numerical - C_analytical))


def l2_norm_error(C_numerical: np.ndarray, C_analytical: np.ndarray, dx: float) -> float:
    """L² norm error over spatial domain.
    
    ||E||_L² = √(∫ (C_num - C_ana)² dx)
    
    Args:
        C_numerical: Numerical solution at single time [mol/m³]
        C_analytical: Analytical solution at single time [mol/m³]
        dx: Spatial step size [m]
        
    Returns:
        float: L² norm of error [mol/m³]
    """
    diff = C_numerical - C_analytical
    l2 = np.sqrt(np.sum(diff ** 2) * dx)
    return l2


def mass_conservation_check(
    x: np.ndarray,
    C: np.ndarray,
    dx: float,
    clearance: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Check mass conservation during simulation.
    
    Total mass: M(t) = ∫ C(x,t) dx
    
    For pure diffusion (clearance=0) with zero-flux boundaries:
        M(t) = constant (conservation)
    
    For with clearance:
        dM/dt = -∫ k·C dx
    
    Args:
        x: Spatial grid [m]
        C: Concentration field [nx, nt] [mol/m³]
        dx: Spatial step size [m]
        clearance: Clearance coefficient [1/s]
        
    Returns:
        Tuple:
            - total_mass: Mass at each time step [mol/m]
            - mass_loss_rate: Expected mass loss rate [mol/(m·s)]
    """
    # Trapezoidal integration over space
    total_mass = np.trapz(C, x, axis=0)
    
    # Expected loss per time step due to clearance
    if clearance > 0:
        # Mass loss = -k · ∫ C dx · Δt
        mass_loss_per_step = -clearance * total_mass
    else:
        mass_loss_per_step = np.zeros_like(total_mass)
    
    return total_mass, mass_loss_per_step


def convergence_order_analysis(
    errors: np.ndarray,
    refinement_factors: np.ndarray
) -> Tuple[float, float]:
    """Estimate convergence order from error vs mesh size data.
    
    If error ~ h^p, then:
        log(error) = p · log(h) + const
        p = d(log(error)) / d(log(h))
    
    Args:
        errors: Array of errors at different refinements
        refinement_factors: Array of mesh sizes or refinement factors
        
    Returns:
        Tuple:
            - convergence_order: Estimated order p
            - r_squared: R² value of linear fit (0-1)
    """
    if len(errors) < 2:
        raise ValueError("Need at least 2 error measurements for convergence analysis")
    
    log_h = np.log(refinement_factors)
    log_error = np.log(errors)
    
    # Linear regression
    coeffs = np.polyfit(log_h, log_error, 1)
    convergence_order = coeffs[0]
    
    # R² calculation
    y_pred = np.polyval(coeffs, log_h)
    ss_res = np.sum((log_error - y_pred) ** 2)
    ss_tot = np.sum((log_error - np.mean(log_error)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return convergence_order, r_squared


def validation_report(
    C_numerical: np.ndarray,
    C_analytical: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    dx: float,
    dt: float,
    clearance: float = 0.0
) -> Dict:
    """Generate comprehensive validation report.
    
    Args:
        C_numerical: Numerical solution [nx, nt] [mol/m³]
        C_analytical: Analytical solution [nx, nt] [mol/m³]
        x: Spatial grid [m]
        t: Time grid [s]
        dx: Spatial step [m]
        dt: Temporal step [s]
        clearance: Clearance coefficient [1/s]
        
    Returns:
        Dict: Validation metrics
    """
    # Spatial and temporal errors
    rmse_val = rmse(C_numerical, C_analytical)
    max_err = maximum_absolute_error(C_numerical, C_analytical)
    
    # Mean relative error (at non-zero locations)
    rel_err = relative_error(C_numerical, C_analytical)
    mean_rel_err = np.mean(rel_err[C_analytical > 1e-14])
    
    # Final time L² error
    l2_final = l2_norm_error(C_numerical[:, -1], C_analytical[:, -1], dx)
    
    # Mass conservation
    total_mass, _ = mass_conservation_check(x, C_numerical, dx, clearance)
    mass_initial = total_mass[0]
    mass_final = total_mass[-1]
    mass_change = abs(mass_final - mass_initial) / (mass_initial + 1e-14)
    
    report = {
        'rmse': rmse_val,
        'max_absolute_error': max_err,
        'mean_relative_error': mean_rel_err,
        'l2_error_final_time': l2_final,
        'mass_initial': mass_initial,
        'mass_final': mass_final,
        'mass_change_fraction': mass_change,
        'grid_resolution': dx,
        'time_step': dt,
        'spatial_points': len(x),
        'time_points': len(t)
    }
    
    return report
