"""
Stability analysis for finite-difference methods.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def check_explicit_fdm_stability(diffusion_coeff: float, dx: float, dt: float) -> bool:
    """Check stability of explicit finite-difference method.
    
    For the 1D diffusion equation with explicit FDM:
        ∂C/∂t = D ∂²C/∂x²
    
    The stability condition (CFL/Fourier number) is:
        r = D·Δt/Δx² ≤ 0.5
    
    Args:
        diffusion_coeff: Diffusion coefficient D [m²/s]
        dx: Spatial step size Δx [m]
        dt: Temporal step size Δt [s]
        
    Returns:
        bool: True if stable, False if unstable
        
    Raises:
        ValueError: If parameters are invalid
    """
    if diffusion_coeff < 0:
        raise ValueError(f"Diffusion coefficient must be non-negative, got {diffusion_coeff}")
    if dx <= 0:
        raise ValueError(f"Spatial step must be positive, got {dx}")
    if dt <= 0:
        raise ValueError(f"Temporal step must be positive, got {dt}")
    
    # Fourier number (dimensionless)
    r = diffusion_coeff * dt / (dx ** 2)
    
    stability_limit = 0.5
    is_stable = r <= stability_limit
    
    logger.info(f"Stability check: r = {r:.6f}, limit = {stability_limit}")
    
    if not is_stable:
        logger.warning(
            f"Explicit FDM is UNSTABLE: r = {r:.6f} > {stability_limit}\n"
            f"Reduce time step or increase spatial step.\n"
            f"Suggested max dt = {stability_limit * dx**2 / diffusion_coeff:.6e} s"
        )
    else:
        logger.info(f"Explicit FDM is STABLE: r = {r:.6f} ≤ {stability_limit}")
    
    return is_stable


def get_max_stable_dt(diffusion_coeff: float, dx: float) -> float:
    """Calculate maximum stable time step for explicit FDM.
    
    Args:
        diffusion_coeff: Diffusion coefficient D [m²/s]
        dx: Spatial step size Δx [m]
        
    Returns:
        float: Maximum stable time step [s]
    """
    if diffusion_coeff <= 0:
        raise ValueError(f"Diffusion coefficient must be positive for stability analysis, got {diffusion_coeff}")
    if dx <= 0:
        raise ValueError(f"Spatial step must be positive, got {dx}")
    
    stability_limit = 0.5
    dt_max = stability_limit * dx ** 2 / diffusion_coeff
    
    return dt_max
