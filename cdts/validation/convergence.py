"""
Convergence Analysis for M6 Verification

Implements systematic convergence studies:
1. Grid convergence (spatial refinement)
2. Time-step convergence (temporal refinement)
3. Convergence order analysis
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceStudyResult:
    """Single convergence study result."""
    grid_size: float  # Δx or Δt
    n_points: int
    rmse: float
    relative_error: float
    max_error: float
    runtime: float
    mass_balance_error: float = 0.0
    
    def __repr__(self):
        return (f"ConvergenceResult(size={self.grid_size:.4e}, "
                f"points={self.n_points}, rmse={self.rmse:.4e})")


class GridConvergenceStudy:
    """Systematic spatial grid refinement study."""
    
    def __init__(self, solver_factory, benchmark, reference_solution):
        """Initialize grid convergence study.
        
        Args:
            solver_factory: Callable that creates solver with given dx, dt
            benchmark: AnalyticalBenchmark instance
            reference_solution: Precomputed reference analytical solution
        """
        self.solver_factory = solver_factory
        self.benchmark = benchmark
        self.reference_solution = reference_solution
        self.results: List[ConvergenceStudyResult] = []
    
    def run(
        self,
        initial_dx: float,
        refinement_factor: float = 2.0,
        n_refinements: int = 4,
        dt_factor: float = 0.1,
        final_time: float = 1e-3
    ) -> List[ConvergenceStudyResult]:
        """Execute grid refinement study.
        
        Args:
            initial_dx: Coarsest grid spacing [m]
            refinement_factor: Factor for grid refinement (default 2)
            n_refinements: Number of refinement levels
            dt_factor: Time step relative to stability limit
            final_time: Simulation time [s]
            
        Returns:
            List of convergence results
        """
        self.results = []
        
        for level in range(n_refinements):
            # Compute grid size for this level
            dx = initial_dx / (refinement_factor ** level)
            
            # Create and run solver
            try:
                solver = self.solver_factory(dx=dx, dt_factor=dt_factor)
            except Exception as e:
                logger.warning(f"Solver creation failed at level {level}: {e}")
                break
            
            logger.info(f"Grid refinement level {level}: dx={dx:.4e}, nx={solver.nx}")
            
            # Run solver
            import time
            t0 = time.time()
            x, t, C_numerical, metrics = solver.solve(
                initial_condition=self.benchmark.C_init * np.ones(solver.nx),
                final_time=final_time
            )
            runtime = time.time() - t0
            
            # Interpolate reference solution to current grid
            # (reference solution should be computed at finest resolution)
            if C_numerical.shape[0] != self.reference_solution.shape[0]:
                # Interpolate reference to current grid
                C_ref_interp = np.interp(
                    x,
                    np.linspace(0, self.benchmark.L, self.reference_solution.shape[0]),
                    self.reference_solution,
                    axis=0
                )
            else:
                C_ref_interp = self.reference_solution
            
            # Compute error metrics
            from ..validation.metrics import (
                rmse, relative_error, maximum_absolute_error,
                mass_conservation_check
            )
            
            rmse_val = rmse(C_numerical, C_ref_interp)
            rel_err = np.mean(relative_error(C_numerical, C_ref_interp))
            max_err = maximum_absolute_error(C_numerical, C_ref_interp)
            
            # Mass conservation
            total_mass, _ = mass_conservation_check(
                x, C_numerical, dx, self.benchmark.k
            )
            M_initial = total_mass[0]
            M_final = total_mass[-1]
            mass_error = abs(M_final - M_initial) / (M_initial + 1e-14)
            
            result = ConvergenceStudyResult(
                grid_size=dx,
                n_points=solver.nx,
                rmse=rmse_val,
                relative_error=rel_err,
                max_error=max_err,
                runtime=runtime,
                mass_balance_error=mass_error
            )
            
            self.results.append(result)
            logger.info(f"  RMSE={rmse_val:.4e}, RelErr={rel_err:.4e}, Runtime={runtime:.3f}s")
        
        return self.results
    
    def convergence_order(self) -> Tuple[float, float]:
        """Analyze convergence order from RMSE vs grid size.
        
        Returns:
            Tuple:
                - order: Convergence order (slope in log-log plot)
                - r_squared: Goodness of fit
        """
        if len(self.results) < 2:
            raise ValueError("Need at least 2 convergence points")
        
        grid_sizes = np.array([r.grid_size for r in self.results])
        rmse_vals = np.array([r.rmse for r in self.results])
        
        # Linear regression in log-log space
        log_h = np.log(grid_sizes)
        log_error = np.log(rmse_vals)
        
        coeffs = np.polyfit(log_h, log_error, 1)
        order = coeffs[0]
        
        # R² calculation
        y_pred = np.polyval(coeffs, log_h)
        ss_res = np.sum((log_error - y_pred) ** 2)
        ss_tot = np.sum((log_error - np.mean(log_error)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        logger.info(f"Convergence order: {order:.2f} (R²={r_squared:.4f})")
        
        return order, r_squared
    
    def expected_convergence_order(self, method: str) -> float:
        """Return expected convergence order for method.
        
        Args:
            method: 'explicit_fdm', 'crank_nicolson', 'fem'
            
        Returns:
            Expected spatial convergence order
        """
        orders = {
            'explicit_fdm': 2.0,  # Second-order spatial
            'crank_nicolson': 2.0,  # Second-order spatial
            'fem': 2.0,  # Linear elements, second-order
        }
        return orders.get(method, 2.0)


class TimeConvergenceStudy:
    """Systematic temporal refinement study."""
    
    def __init__(self, solver_factory, benchmark, reference_solution):
        """Initialize time convergence study.
        
        Args:
            solver_factory: Callable that creates solver with given dx, dt
            benchmark: AnalyticalBenchmark instance
            reference_solution: Reference solution at finest resolution
        """
        self.solver_factory = solver_factory
        self.benchmark = benchmark
        self.reference_solution = reference_solution
        self.results: List[ConvergenceStudyResult] = []
    
    def run(
        self,
        dx: float,
        initial_dt: float,
        refinement_factor: float = 2.0,
        n_refinements: int = 4,
        final_time: float = 1e-3
    ) -> List[ConvergenceStudyResult]:
        """Execute temporal refinement study.
        
        Args:
            dx: Spatial grid size [m]
            initial_dt: Coarsest time step [s]
            refinement_factor: Factor for time step refinement
            n_refinements: Number of refinement levels
            final_time: Simulation time [s]
            
        Returns:
            List of convergence results
        """
        self.results = []
        
        for level in range(n_refinements):
            # Compute time step for this level
            dt = initial_dt / (refinement_factor ** level)
            
            # Create and run solver
            try:
                solver = self.solver_factory(dx=dx, dt=dt)
            except Exception as e:
                logger.warning(f"Solver creation failed at level {level}: {e}")
                break
            
            logger.info(f"Time refinement level {level}: dt={dt:.4e}, nt≈{int(final_time/dt)}")
            
            # Run solver
            import time
            t0 = time.time()
            x, t, C_numerical, metrics = solver.solve(
                initial_condition=self.benchmark.C_init * np.ones(solver.nx),
                final_time=final_time
            )
            runtime = time.time() - t0
            
            # Compute error metrics relative to reference
            from ..validation.metrics import (
                rmse, relative_error, maximum_absolute_error,
                mass_conservation_check
            )
            
            rmse_val = rmse(C_numerical, self.reference_solution)
            rel_err = np.mean(relative_error(C_numerical, self.reference_solution))
            max_err = maximum_absolute_error(C_numerical, self.reference_solution)
            
            # Mass conservation
            total_mass, _ = mass_conservation_check(
                x, C_numerical, dx, self.benchmark.k
            )
            M_initial = total_mass[0]
            M_final = total_mass[-1]
            mass_error = abs(M_final - M_initial) / (M_initial + 1e-14)
            
            result = ConvergenceStudyResult(
                grid_size=dt,
                n_points=len(t),
                rmse=rmse_val,
                relative_error=rel_err,
                max_error=max_err,
                runtime=runtime,
                mass_balance_error=mass_error
            )
            
            self.results.append(result)
            logger.info(f"  RMSE={rmse_val:.4e}, RelErr={rel_err:.4e}, Runtime={runtime:.3f}s")
        
        return self.results
    
    def convergence_order(self) -> Tuple[float, float]:
        """Analyze convergence order from RMSE vs time step.
        
        Returns:
            Tuple:
                - order: Temporal convergence order
                - r_squared: Goodness of fit
        """
        if len(self.results) < 2:
            raise ValueError("Need at least 2 convergence points")
        
        time_steps = np.array([r.grid_size for r in self.results])
        rmse_vals = np.array([r.rmse for r in self.results])
        
        # Linear regression in log-log space
        log_dt = np.log(time_steps)
        log_error = np.log(rmse_vals)
        
        coeffs = np.polyfit(log_dt, log_error, 1)
        order = coeffs[0]
        
        # R² calculation
        y_pred = np.polyval(coeffs, log_dt)
        ss_res = np.sum((log_error - y_pred) ** 2)
        ss_tot = np.sum((log_error - np.mean(log_error)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        logger.info(f"Temporal convergence order: {order:.2f} (R²={r_squared:.4f})")
        
        return order, r_squared
