"""
M2 simulation runner for reaction-diffusion with clearance.
Runs solver with analytical benchmark comparison and mass loss verification.
"""

import sys
import logging
import numpy as np
from pathlib import Path
import json
import time as time_module

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cdts.config import load_config_from_yaml
from cdts.solvers.explicit_fdm import ExplicitFDMSolver
from cdts.validation.reaction_diffusion import (
    finite_domain_with_clearance_exponential,
    diffusion_dominance_time,
    verify_mass_loss_consistency
)
from cdts.validation.metrics import (
    rmse, maximum_absolute_error, relative_error,
    mass_conservation_check, validation_report
)
from cdts.visualization.plots import (
    plot_concentration_profiles, plot_error_analysis,
    plot_mass_conservation, plot_solver_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_clearance_comparison(
    x: np.ndarray,
    C_low: np.ndarray,
    C_medium: np.ndarray,
    C_high: np.ndarray,
    t: np.ndarray,
    output_path: str,
    title: str = "Clearance Effect Comparison"
) -> None:
    """Plot concentration comparison across clearance regimes.
    
    Args:
        x: Spatial grid
        C_low: Concentration with low clearance
        C_medium: Concentration with medium clearance
        C_high: Concentration with high clearance
        t: Time array
        output_path: Output file path
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select time indices to plot
    n_times = C_low.shape[1]
    time_indices = [n_times//4, n_times//2, 3*n_times//4, n_times-1]
    
    # Plot 1: Low clearance profiles
    ax = axes[0, 0]
    for idx in time_indices:
        ax.plot(x*1e3, C_low[:, idx], marker='o', markersize=2, linewidth=1.5,
                label=f't = {t[idx]:.1e} s')
    ax.set_xlabel('Position x [mm]', fontsize=11)
    ax.set_ylabel('Concentration [mol/m³]', fontsize=11)
    ax.set_title('Low Clearance (k=10⁻⁴ 1/s)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Medium clearance profiles
    ax = axes[0, 1]
    for idx in time_indices:
        ax.plot(x*1e3, C_medium[:, idx], marker='o', markersize=2, linewidth=1.5,
                label=f't = {t[idx]:.1e} s')
    ax.set_xlabel('Position x [mm]', fontsize=11)
    ax.set_ylabel('Concentration [mol/m³]', fontsize=11)
    ax.set_title('Medium Clearance (k=10⁻³ 1/s)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: High clearance profiles
    ax = axes[1, 0]
    for idx in time_indices:
        ax.plot(x*1e3, C_high[:, idx], marker='o', markersize=2, linewidth=1.5,
                label=f't = {t[idx]:.1e} s')
    ax.set_xlabel('Position x [mm]', fontsize=11)
    ax.set_ylabel('Concentration [mol/m³]', fontsize=11)
    ax.set_title('High Clearance (k=10⁻² 1/s)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Total mass comparison
    ax = axes[1, 1]
    M_low, _ = mass_conservation_check(x, C_low, x[1]-x[0], 1e-4)
    M_med, _ = mass_conservation_check(x, C_medium, x[1]-x[0], 1e-3)
    M_high, _ = mass_conservation_check(x, C_high, x[1]-x[0], 1e-2)
    
    ax.semilogy(t, M_low/M_low[0], 'b-', linewidth=2, label='Low clearance')
    ax.semilogy(t, M_med/M_med[0], 'g-', linewidth=2, label='Medium clearance')
    ax.semilogy(t, M_high/M_high[0], 'r-', linewidth=2, label='High clearance')
    
    # Theoretical exponential decay curves
    k_values = [1e-4, 1e-3, 1e-2]
    colors = ['b', 'g', 'r']
    for k, color in zip(k_values, colors):
        ax.semilogy(t, np.exp(-k*t), '--', color=color, alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time t [s]', fontsize=11)
    ax.set_ylabel('Normalized Mass M(t)/M(0)', fontsize=11)
    ax.set_title('Mass Decay Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved clearance comparison plot: {output_path}")
    plt.close()


def run_m2_experiment(config_path: str) -> None:
    """Run single M2 reaction-diffusion experiment.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config_from_yaml(config_path)
    logger.info(f"Experiment ID: {config.experiment_id}")
    logger.info(f"Experiment Name: {config.name}")
    
    # Create output directory
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tissue parameters
    tissue_layer = config.tissue.layers[0]
    domain_length = tissue_layer.thickness
    D = tissue_layer.diffusion_coefficient
    k = tissue_layer.clearance
    
    logger.info("\n" + "=" * 70)
    logger.info("M2 REACTION-DIFFUSION SIMULATION")
    logger.info("=" * 70)
    logger.info(f"Domain length:                      {domain_length:.6e} m ({domain_length*1e3:.3f} mm)")
    logger.info(f"Diffusion coefficient D:            {D:.6e} m²/s")
    logger.info(f"Clearance coefficient k:            {k:.6e} 1/s")
    
    # Calculate timescales
    if k > 0:
        tau_d, tau_k, Da = diffusion_dominance_time(D, k, domain_length)
        logger.info(f"\nCharacteristic Timescales:")
        logger.info(f"  Diffusion timescale τ_D:          {tau_d:.6e} s")
        logger.info(f"  Clearance timescale τ_k:          {tau_k:.6e} s")
        logger.info(f"  Damköhler number Da:              {Da:.6f}")
        
        if Da < 0.1:
            regime = "DIFFUSION-LIMITED"
        elif Da > 10:
            regime = "CLEARANCE-LIMITED"
        else:
            regime = "INTERMEDIATE"
        logger.info(f"  Regime:                           {regime}")
    
    logger.info(f"\nNumerical Configuration:")
    logger.info(f"  Spatial step Δx:                  {config.simulation.spatial_step:.6e} m")
    logger.info(f"  Temporal step Δt:                 {config.simulation.time_step:.6e} s")
    logger.info(f"  Final time:                       {config.simulation.final_time:.6e} s")
    
    # Initialize solver
    logger.info("\n" + "=" * 70)
    logger.info("Solver Initialization")
    logger.info("=" * 70)
    
    try:
        solver = ExplicitFDMSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=config.simulation.spatial_step,
            dt=config.simulation.time_step,
            boundary_left=config.boundary_left.value,
            boundary_right=config.boundary_right.value,
            clearance=k
        )
        logger.info(f"Solver initialized: explicit FDM with clearance")
        logger.info(f"Grid points:                        {solver.nx}")
        logger.info(f"Fourier number r:                  {solver.r:.6f}")
    except ValueError as e:
        logger.error(f"Solver initialization failed: {e}")
        return
    
    # Initial condition
    C_init = np.full(solver.nx, config.drug.initial_concentration)
    
    # Run solver
    logger.info("\n" + "=" * 70)
    logger.info("Simulation Execution")
    logger.info("=" * 70)
    
    t_start = time_module.time()
    x, t, C_num, metrics = solver.solve(C_init, config.simulation.final_time)
    t_elapsed = time_module.time() - t_start
    
    logger.info(f"Simulation completed in {t_elapsed:.3f} seconds")
    logger.info(f"Time steps:                         {len(t)}")
    logger.info(f"Total grid cells:                   {len(x) * len(t)}")
    
    # Generate analytical solution
    logger.info("\n" + "=" * 70)
    logger.info("Analytical Solution")
    logger.info("=" * 70)
    
    C_ana = finite_domain_with_clearance_exponential(
        x=x,
        t=t,
        D=D,
        k=k,
        L=domain_length,
        C_left=config.boundary_left.value,
        C_right=config.boundary_right.value,
        C_init=config.drug.initial_concentration
    )
    logger.info("Analytical solution generated")
    
    # Validation metrics
    logger.info("\n" + "=" * 70)
    logger.info("Validation Metrics")
    logger.info("=" * 70)
    
    rmse_val = rmse(C_num, C_ana)
    max_err = maximum_absolute_error(C_num, C_ana)
    rel_err = relative_error(C_num, C_ana)
    mean_rel_err = np.mean(rel_err[C_ana > 1e-14])
    
    logger.info(f"Root Mean Squared Error (RMSE):      {rmse_val:.6e} mol/m³")
    logger.info(f"Maximum Absolute Error:              {max_err:.6e} mol/m³")
    logger.info(f"Mean Relative Error:                 {mean_rel_err:.6e}")
    
    # Mass conservation with clearance
    logger.info("\n" + "=" * 70)
    logger.info("Mass Conservation & Clearance Analysis")
    logger.info("=" * 70)
    
    total_mass_num, _ = mass_conservation_check(x, C_num, config.simulation.spatial_step, k)
    total_mass_ana, _ = mass_conservation_check(x, C_ana, config.simulation.spatial_step, k)
    
    logger.info(f"Numerical Solution:")
    logger.info(f"  Initial mass:                     {total_mass_num[0]:.6e} mol/m")
    logger.info(f"  Final mass:                       {total_mass_num[-1]:.6e} mol/m")
    logger.info(f"  Remaining fraction:               {total_mass_num[-1]/total_mass_num[0]:.6e}")
    
    logger.info(f"\nAnalytical Solution:")
    logger.info(f"  Initial mass:                     {total_mass_ana[0]:.6e} mol/m")
    logger.info(f"  Final mass:                       {total_mass_ana[-1]:.6e} mol/m")
    logger.info(f"  Remaining fraction:               {total_mass_ana[-1]/total_mass_ana[0]:.6e}")
    
    # Verify mass loss consistency
    if k > 0 and total_mass_num[0] > 0:
        is_consistent, mean_error, _ = verify_mass_loss_consistency(
            t, total_mass_num, k, tolerance=0.1
        )
        logger.info(f"\nMass Loss Verification:")
        logger.info(f"  Expected: M(t) = M(0)·exp(-k·t)")
        logger.info(f"  Mean error:                       {mean_error:.6e}")
        logger.info(f"  Consistent (±10%):                {'YES' if is_consistent else 'NO'}")
    
    # Save metrics
    report = {
        'rmse': float(rmse_val),
        'max_absolute_error': float(max_err),
        'mean_relative_error': float(mean_rel_err),
        'mass_initial': float(total_mass_num[0]),
        'mass_final': float(total_mass_num[-1]),
        'mass_remaining_fraction': float(total_mass_num[-1] / (total_mass_num[0] + 1e-14)),
        'diffusion_coefficient': D,
        'clearance_coefficient': k,
        'grid_resolution': config.simulation.spatial_step,
        'time_step': config.simulation.time_step,
        'spatial_points': len(x),
        'time_points': len(t),
        'regime': 'DIFFUSION-LIMITED' if (k > 0 and k * domain_length**2 / D < 0.1) else \
                  'CLEARANCE-LIMITED' if (k > 0 and k * domain_length**2 / D > 10) else \
                  'INTERMEDIATE' if k > 0 else 'NO_CLEARANCE'
    }
    
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nMetrics saved to: {metrics_file}")
    
    # Generate visualizations
    logger.info("\n" + "=" * 70)
    logger.info("Visualization Generation")
    logger.info("=" * 70)
    
    plot_concentration_profiles(
        x=x, C=C_num, t=t,
        output_path=str(output_dir / 'concentration_profiles_numerical.png'),
        title='Numerical Solution: Reaction-Diffusion with Clearance'
    )
    
    plot_concentration_profiles(
        x=x, C=C_ana, t=t,
        output_path=str(output_dir / 'concentration_profiles_analytical.png'),
        title='Analytical Solution: Reaction-Diffusion with Clearance'
    )
    
    plot_error_analysis(
        x=x, C_numerical=C_num, C_analytical=C_ana, t=t,
        output_path=str(output_dir / 'error_analysis.png'),
        title='Numerical Error vs Analytical Solution'
    )
    
    plot_mass_conservation(
        t=t, total_mass=total_mass_num, expected_loss=np.zeros_like(total_mass_num),
        output_path=str(output_dir / 'mass_decay.png'),
        title='Total Mass Decay (Numerical vs Theoretical exp(-k·t))'
    )
    
    plot_solver_metrics(
        metrics_dict=report,
        output_path=str(output_dir / 'metrics_summary.png'),
        title='M2 Validation Metrics Summary'
    )
    
    logger.info("All visualizations saved to output directory")
    
    # Export data
    logger.info("\n" + "=" * 70)
    logger.info("Data Export")
    logger.info("=" * 70)
    
    import csv
    csv_file = output_dir / 'final_profile.csv'
    
    # Recalculate relative error for final time
    rel_err_final = relative_error(C_num[:, -1], C_ana[:, -1])
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Position_m', 'Position_mm', 'Numerical_mol_m3', 'Analytical_mol_m3', 
                        'Absolute_Error_mol_m3', 'Relative_Error'])
        for i in range(len(x)):
            writer.writerow([
                f"{x[i]:.6e}",
                f"{x[i]*1e3:.6f}",
                f"{C_num[i, -1]:.6e}",
                f"{C_ana[i, -1]:.6e}",
                f"{abs(C_num[i, -1] - C_ana[i, -1]):.6e}",
                f"{rel_err_final[i]:.6e}" if C_ana[i, -1] > 1e-14 else "N/A"
            ])
    logger.info(f"CSV data saved to: {csv_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("M2 SIMULATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_m2_simulation.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_m2_experiment(config_path)
