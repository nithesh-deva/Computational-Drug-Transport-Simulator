"""
Main simulation runner for M1 validation experiment.
Runs pure diffusion solver with analytical benchmark comparison.
"""

import sys
import logging
import numpy as np
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cdts.config import load_config_from_yaml
from cdts.solvers.explicit_fdm import ExplicitFDMSolver
from cdts.validation.analytical import finite_domain_solution
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


def run_simulation(config_path: str) -> None:
    """Run complete M1 validation simulation.
    
    Args:
        config_path: Path to YAML configuration file
    """
    logger.info("=" * 70)
    logger.info("CDTS M1 — Verified 1D Pure Diffusion Solver")
    logger.info("=" * 70)
    
    # Load and validate configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config_from_yaml(config_path)
    logger.info(f"Experiment ID: {config.experiment_id}")
    logger.info(f"Experiment Name: {config.name}")
    
    # Create output directory
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration to output
    with open(output_dir / 'config.yaml', 'w') as f:
        import yaml
        yaml.dump({
            'experiment': {
                'id': config.experiment_id,
                'name': config.name
            },
            'drug': {
                'initial_concentration': config.drug.initial_concentration,
                'unit': config.drug.unit
            },
            'tissue': {
                'layers': [
                    {
                        'name': layer.name,
                        'thickness': layer.thickness,
                        'diffusion_coefficient': layer.diffusion_coefficient,
                        'clearance': layer.clearance
                    }
                    for layer in config.tissue.layers
                ]
            },
            'simulation': {
                'final_time': config.simulation.final_time,
                'spatial_step': config.simulation.spatial_step,
                'time_step': config.simulation.time_step
            },
            'solver': {'method': config.solver.method},
            'boundary': {
                'left': {
                    'type': config.boundary_left.type,
                    'value': config.boundary_left.value
                },
                'right': {
                    'type': config.boundary_right.type,
                    'value': config.boundary_right.value
                }
            }
        }, f)
    
    # Get tissue parameters
    tissue_layer = config.tissue.layers[0]  # M1 uses single layer
    domain_length = tissue_layer.thickness
    D = tissue_layer.diffusion_coefficient
    clearance = tissue_layer.clearance
    
    logger.info("\n" + "=" * 70)
    logger.info("Physical Parameters")
    logger.info("=" * 70)
    logger.info(f"Domain length (tissue thickness):    {domain_length:.6e} m = {domain_length*1e3:.3f} mm")
    logger.info(f"Diffusion coefficient D:             {D:.6e} m²/s")
    logger.info(f"Clearance coefficient k:             {clearance:.6e} 1/s")
    logger.info(f"Initial concentration:               {config.drug.initial_concentration:.6e} mol/m³")
    logger.info(f"Boundary condition (left):           {config.boundary_left.value:.6e} mol/m³")
    logger.info(f"Boundary condition (right):          {config.boundary_right.value:.6e} mol/m³")
    
    logger.info("\n" + "=" * 70)
    logger.info("Numerical Configuration")
    logger.info("=" * 70)
    logger.info(f"Spatial discretization Δx:          {config.simulation.spatial_step:.6e} m")
    logger.info(f"Temporal discretization Δt:         {config.simulation.time_step:.6e} s")
    logger.info(f"Final simulation time:              {config.simulation.final_time:.6e} s")
    
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
            clearance=clearance
        )
        logger.info(f"Solver initialized: explicit FDM")
        logger.info(f"Grid points:                        {solver.nx}")
        logger.info(f"Fourier number r = D·Δt/Δx²:       {solver.r:.6f}")
        logger.info(f"Stability status:                   STABLE (r ≤ 0.5)")
    except ValueError as e:
        logger.error(f"Solver initialization failed: {e}")
        return
    
    # Initial condition
    C_init = np.full(solver.nx, config.drug.initial_concentration)
    
    # Run solver
    logger.info("\n" + "=" * 70)
    logger.info("Simulation Execution")
    logger.info("=" * 70)
    
    import time
    t_start = time.time()
    x, t, C, metrics = solver.solve(C_init, config.simulation.final_time)
    t_elapsed = time.time() - t_start
    
    logger.info(f"Simulation completed in {t_elapsed:.2f} seconds")
    logger.info(f"Time steps:                         {len(t)}")
    logger.info(f"Spatial points:                     {len(x)}")
    logger.info(f"Total grid cells:                   {len(x) * len(t)}")
    
    # Calculate analytical solution
    logger.info("\n" + "=" * 70)
    logger.info("Analytical Solution Generation")
    logger.info("=" * 70)
    
    C_ana = finite_domain_solution(
        x=x,
        t=t,
        D=D,
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
    
    report = validation_report(
        C_numerical=C,
        C_analytical=C_ana,
        x=x,
        t=t,
        dx=config.simulation.spatial_step,
        dt=config.simulation.time_step,
        clearance=clearance
    )
    
    logger.info(f"Root Mean Squared Error (RMSE):      {report['rmse']:.6e} mol/m³")
    logger.info(f"Maximum Absolute Error:              {report['max_absolute_error']:.6e} mol/m³")
    logger.info(f"Mean Relative Error:                 {report['mean_relative_error']:.6e}")
    logger.info(f"L² Norm Error (final time):          {report['l2_error_final_time']:.6e} mol/m³")
    
    logger.info("\nMass Conservation Check:")
    logger.info(f"  Initial total mass:                {report['mass_initial']:.6e} mol/m")
    logger.info(f"  Final total mass:                  {report['mass_final']:.6e} mol/m")
    logger.info(f"  Relative mass change:              {report['mass_change_fraction']:.6e}")
    
    # Save metrics to JSON
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nMetrics saved to: {metrics_file}")
    
    # Generate visualizations
    logger.info("\n" + "=" * 70)
    logger.info("Visualization Generation")
    logger.info("=" * 70)
    
    # Concentration profiles
    plot_concentration_profiles(
        x=x,
        C=C,
        t=t,
        output_path=str(output_dir / 'concentration_profiles.png'),
        title='Numerical Solution: Drug Concentration Profiles'
    )
    
    # Error analysis
    plot_error_analysis(
        x=x,
        C_numerical=C,
        C_analytical=C_ana,
        t=t,
        output_path=str(output_dir / 'error_analysis.png'),
        title='Numerical Error vs Analytical Solution'
    )
    
    # Mass conservation
    total_mass, mass_loss = mass_conservation_check(x, C, config.simulation.spatial_step, clearance)
    plot_mass_conservation(
        t=t,
        total_mass=total_mass,
        expected_loss=mass_loss,
        output_path=str(output_dir / 'mass_conservation.png'),
        title='Mass Conservation Check'
    )
    
    # Metrics summary
    plot_solver_metrics(
        metrics_dict=report,
        output_path=str(output_dir / 'metrics_summary.png'),
        title='M1 Validation Metrics Summary'
    )
    
    logger.info("All visualizations saved to output directory")
    
    # Save numerical and analytical data
    logger.info("\n" + "=" * 70)
    logger.info("Data Export")
    logger.info("=" * 70)
    
    # Save as CSV for reference
    import csv
    csv_file = output_dir / 'concentration_final.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Position_m', 'Position_mm', 'Numerical_mol_m3', 'Analytical_mol_m3', 'Absolute_Error_mol_m3'])
        for i in range(len(x)):
            writer.writerow([
                f"{x[i]:.6e}",
                f"{x[i]*1e3:.6f}",
                f"{C[i, -1]:.6e}",
                f"{C_ana[i, -1]:.6e}",
                f"{abs(C[i, -1] - C_ana[i, -1]):.6e}"
            ])
    logger.info(f"CSV data saved to: {csv_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("M1 VALIDATION SIMULATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_m1_simulation.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_simulation(config_path)
