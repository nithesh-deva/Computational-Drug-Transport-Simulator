"""
Visualization for CDTS results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def plot_concentration_profiles(
    x: np.ndarray,
    C: np.ndarray,
    t: np.ndarray,
    time_indices: Optional[list] = None,
    output_path: Optional[str] = None,
    title: str = "Drug Concentration Profiles"
) -> None:
    """Plot concentration profiles at selected time points.
    
    Args:
        x: Spatial grid [m]
        C: Concentration field [nx, nt] [mol/m³]
        t: Time array [s]
        time_indices: Indices of times to plot (default: [0, 25%, 50%, 75%, 100%])
        output_path: Path to save figure
        title: Plot title
    """
    if time_indices is None:
        nt = C.shape[1]
        time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx in time_indices:
        ax.plot(x * 1e3, C[:, idx] / np.max(C), marker='o', markersize=3, 
                label=f't = {t[idx]:.2e} s', linewidth=1.5)
    
    ax.set_xlabel('Position x [mm]', fontsize=12)
    ax.set_ylabel('Normalized Concentration C/C_max', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x[0]*1e3, x[-1]*1e3])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved concentration profile plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_analysis(
    x: np.ndarray,
    C_numerical: np.ndarray,
    C_analytical: np.ndarray,
    t: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Numerical Error Analysis"
) -> None:
    """Plot absolute and relative error between numerical and analytical solutions.
    
    Args:
        x: Spatial grid [m]
        C_numerical: Numerical solution [nx, nt] [mol/m³]
        C_analytical: Analytical solution [nx, nt] [mol/m³]
        t: Time array [s]
        output_path: Path to save figure
        title: Plot title
    """
    absolute_error = np.abs(C_numerical - C_analytical)
    
    # Plot at final time and intermediate times
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute error
    time_indices = [C_numerical.shape[1]//4, C_numerical.shape[1]//2, C_numerical.shape[1]-1]
    for idx in time_indices:
        ax1.semilogy(x * 1e3, absolute_error[:, idx] + 1e-16, marker='o', 
                     markersize=3, label=f't = {t[idx]:.2e} s', linewidth=1.5)
    
    ax1.set_xlabel('Position x [mm]', fontsize=12)
    ax1.set_ylabel('Absolute Error |C_num - C_ana| [mol/m³]', fontsize=12)
    ax1.set_title('Absolute Error', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Numerical vs analytical at final time
    ax2.plot(x * 1e3, C_numerical[:, -1], 'b-', linewidth=2, label='Numerical')
    ax2.plot(x * 1e3, C_analytical[:, -1], 'r--', linewidth=2, label='Analytical')
    ax2.fill_between(x * 1e3, C_numerical[:, -1], C_analytical[:, -1], 
                     alpha=0.2, color='gray', label='Error region')
    
    ax2.set_xlabel('Position x [mm]', fontsize=12)
    ax2.set_ylabel('Concentration [mol/m³]', fontsize=12)
    ax2.set_title(f'Comparison at t = {t[-1]:.2e} s', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error analysis plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_mass_conservation(
    t: np.ndarray,
    total_mass: np.ndarray,
    expected_loss: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Mass Conservation"
) -> None:
    """Plot total mass evolution during simulation.
    
    Args:
        t: Time array [s]
        total_mass: Total mass at each time point [mol/m]
        expected_loss: Expected mass loss (for clearance systems)
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize to initial mass
    M0 = total_mass[0]
    normalized_mass = total_mass / M0
    
    ax.plot(t, normalized_mass, 'b-', linewidth=2, label='Computed Mass')
    
    if np.any(np.abs(expected_loss) > 1e-14):
        # For clearance systems, plot expected behavior
        ax.plot(t, normalized_mass, 'r--', linewidth=1.5, alpha=0.7, label='Expected (with clearance)')
    
    ax.set_xlabel('Time t [s]', fontsize=12)
    ax.set_ylabel('Normalized Total Mass M(t)/M(0)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # For pure diffusion, mass should be constant
    if np.max(np.abs(expected_loss)) < 1e-14:
        ax.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='Ideal conservation')
        ax.set_ylim([0.95, 1.05])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved mass conservation plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_convergence_analysis(
    grid_sizes: np.ndarray,
    errors: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Grid Convergence Study"
) -> None:
    """Plot convergence of numerical error with grid refinement.
    
    Args:
        grid_sizes: Array of spatial step sizes [m]
        errors: Corresponding RMSE values [mol/m³]
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Log-log plot
    ax.loglog(grid_sizes, errors, 'bo-', markersize=8, linewidth=2, label='RMSE')
    
    # Theoretical convergence lines
    reference_point = (grid_sizes[0], errors[0])
    for order, color, label in [(1, 'g', 'O(Δx)'), (2, 'r', 'O(Δx²)')]:
        x_line = grid_sizes
        y_line = reference_point[1] * (x_line / reference_point[0]) ** order
        ax.loglog(x_line, y_line, '--', color=color, alpha=0.6, linewidth=1.5, label=label)
    
    ax.set_xlabel('Grid Size Δx [m]', fontsize=12)
    ax.set_ylabel('RMSE [mol/m³]', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved convergence plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_solver_metrics(
    metrics_dict: Dict,
    output_path: Optional[str] = None,
    title: str = "Validation Metrics Summary"
) -> None:
    """Create a summary figure of validation metrics.
    
    Args:
        metrics_dict: Dictionary of metric name -> value
        output_path: Path to save figure
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 2, figure=fig)
    
    # Extract metrics
    rmse_val = metrics_dict.get('rmse', 0)
    max_err = metrics_dict.get('max_absolute_error', 0)
    mean_rel_err = metrics_dict.get('mean_relative_error', 0)
    l2_final = metrics_dict.get('l2_error_final_time', 0)
    mass_change = metrics_dict.get('mass_change_fraction', 0)
    
    # Create text display
    ax = fig.add_subplot(gs[:, :])
    ax.axis('off')
    
    metrics_text = f"""
    {title}
    
    Root Mean Squared Error (RMSE):           {rmse_val:.6e} mol/m³
    Maximum Absolute Error:                    {max_err:.6e} mol/m³
    Mean Relative Error:                       {mean_rel_err:.6e}
    L² Norm Error (final time):                {l2_final:.6e} mol/m³
    
    Mass Conservation:
        Initial Mass:                          {metrics_dict.get('mass_initial', 0):.6e} mol/m
        Final Mass:                            {metrics_dict.get('mass_final', 0):.6e} mol/m
        Relative Change:                       {mass_change:.6e}
    
    Numerical Configuration:
        Spatial Resolution Δx:                 {metrics_dict.get('grid_resolution', 0):.6e} m
        Temporal Resolution Δt:                {metrics_dict.get('time_step', 0):.6e} s
        Spatial Points:                        {metrics_dict.get('spatial_points', 0)}
        Temporal Points:                       {metrics_dict.get('time_points', 0)}
    """
    
    ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics summary plot: {output_path}")
    else:
        plt.show()
    
    plt.close()
