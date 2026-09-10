"""
Unit tests for M4 Crank-Nicolson implicit solver.
Tests implicit time integration, stability, and solver comparison.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdts.solvers.crank_nicolson import CrankNicolsonSolver, thomas_algorithm
from cdts.solvers.explicit_fdm import ExplicitFDMSolver
from cdts.validation.metrics import rmse


class TestThomasAlgorithm:
    """Test tridiagonal solver."""
    
    def test_thomas_simple_system(self):
        """Test Thomas algorithm on simple tridiagonal system."""
        # System: 2x - y = 1, -x + 2y - z = 0, -y + 2z = 1
        a = np.array([0, -1, -1])
        b = np.array([2, 2, 2])
        c = np.array([-1, -1, 0])
        d = np.array([1, 0, 1])
        
        x = thomas_algorithm(a, b, c, d)
        
        # Verify solution
        assert np.isclose(2*x[0] - x[1], 1.0)
        assert np.isclose(-x[0] + 2*x[1] - x[2], 0.0)
        assert np.isclose(-x[1] + 2*x[2], 1.0)
    
    def test_thomas_diagonally_dominant(self):
        """Test on diagonally dominant matrix."""
        n = 100
        a = np.ones(n) * (-1)
        b = np.ones(n) * 4
        c = np.ones(n) * (-1)
        d = np.ones(n)
        
        x = thomas_algorithm(a, b, c, d)
        
        # Check all solutions are reasonable
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)


class TestCrankNicolsonBasic:
    """Test basic Crank-Nicolson functionality."""
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = CrankNicolsonSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=100.0,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        assert solver.nx == 101
        assert solver.dt == 100.0
        assert solver.clearance == 0.0
    
    def test_solver_execution(self):
        """Test solver basic execution."""
        solver = CrankNicolsonSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0
        )
        
        C_init = np.zeros(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)
        
        # Check output shapes
        assert C.shape == (solver.nx, len(t))
        assert np.isclose(C[0, -1], 1.0)  # Left BC
        assert np.isclose(C[-1, -1], 0.0)  # Right BC
        assert np.all(C >= -1e-12)  # Non-negative
    
    def test_solver_with_clearance(self):
        """Test Crank-Nicolson with clearance term."""
        solver = CrankNicolsonSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=0.0,
            boundary_right=0.0,
            clearance=0.001
        )
        
        # Uniform initial
        C_init = np.ones(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)
        
        # Mass should decrease due to clearance
        total_mass_init = np.trapz(C_init, x)
        total_mass_final = np.trapz(C[:, -1], x)
        
        assert total_mass_final < total_mass_init
        assert metrics['clearance'] == 0.001


class TestCrankNicolsonVsExplicit:
    """Compare Crank-Nicolson vs explicit FDM."""
    
    def test_small_timestep_equivalence(self):
        """With small time steps, both should give similar results."""
        D = 1e-10
        dx = 0.0001
        dt_small = 5.0  # Small, stable for explicit (r = 0.05)
        domain_length = 0.01
        
        # Crank-Nicolson
        solver_cn = CrankNicolsonSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=dx,
            dt=dt_small,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        # Explicit
        solver_exp = ExplicitFDMSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=dx,
            dt=dt_small,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver_cn.nx)
        
        x_cn, t_cn, C_cn, _ = solver_cn.solve(C_init, final_time=100.0)
        x_exp, t_exp, C_exp, _ = solver_exp.solve(C_init, final_time=100.0)
        
        # Compare profiles using RMSE over non-zero region
        mask = C_exp[:, -1] > 1e-12
        assert np.sum(mask) > 0, "Explicit solution should have non-zero values"
        
        from cdts.validation.metrics import rmse
        error = rmse(C_cn[mask, -1], C_exp[mask, -1])
        max_val = np.max(C_exp[mask, -1])
        assert error < max_val * 0.25  # Within 25% RMSE
    
    def test_large_timestep_implicit_stable(self):
        """Large time step only stable for implicit."""
        D = 1e-10
        dx = 0.0001
        dt_large = 200.0  # Unstable for explicit (r = 2.0 > 0.5)
        domain_length = 0.01
        
        # Crank-Nicolson should work
        solver_cn = CrankNicolsonSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=dx,
            dt=dt_large,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver_cn.nx)
        x_cn, t_cn, C_cn, _ = solver_cn.solve(C_init, final_time=400.0)
        
        # Implicit should give reasonable solution
        assert np.all(np.isfinite(C_cn))
        assert np.isclose(C_cn[0, -1], 1.0)
        assert np.isclose(C_cn[-1, -1], 0.0)
        
        # Explicit should fail
        with pytest.raises(ValueError):
            solver_exp = ExplicitFDMSolver(
                domain_length=domain_length,
                diffusion_coeff=D,
                dx=dx,
                dt=dt_large,
                boundary_left=1.0,
                boundary_right=0.0
            )


class TestCrankNicolsonStability:
    """Test stability properties."""
    
    def test_unconditional_stability_high_fourier(self):
        """Test that high Fourier numbers don't cause instability."""
        D = 1e-10
        dx = 0.0001
        dt = 500.0  # Very large, r = 5.0
        domain_length = 0.01
        
        solver = CrankNicolsonSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=dx,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=1000.0)
        
        # Should not have NaN or Inf
        assert np.all(np.isfinite(C))
        # Solution should be bounded
        assert np.all(C >= -1e-10)
        assert np.all(C <= 1.0 + 1e-10)
    
    def test_monotone_behavior(self):
        """Test that solution is physically monotone."""
        D = 1e-10
        dx = 0.0001
        dt = 100.0
        domain_length = 0.01
        
        solver = CrankNicolsonSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=dx,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.nx)
        x, t, C, _ = solver.solve(C_init, final_time=300.0)
        
        # At final time, concentration should be monotone decreasing
        C_final = C[:, -1]
        # Check that gradient is mostly negative (diffusion-driven)
        grad = np.diff(C_final)
        negative_grad = np.sum(grad < 0)
        assert negative_grad > len(grad) * 0.7  # >70% of gradient points are negative


class TestCrankNicolsonAccuracy:
    """Test accuracy of Crank-Nicolson."""
    
    def test_solution_smoothness(self):
        """Test that solution is smooth (expected for implicit methods)."""
        D = 1e-10
        dx = 0.0001
        dt = 50.0
        domain_length = 0.01
        
        solver = CrankNicolsonSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=dx,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.nx)
        x, t, C, _ = solver.solve(C_init, final_time=200.0)
        
        # Check smoothness: second derivative should be smooth
        C_final = C[:, -1]
        second_deriv = np.diff(C_final, n=2)
        
        # Variance in second derivative indicates oscillations
        variance = np.var(second_deriv)
        assert variance < 0.01  # Should have low variance (smooth)


class TestCrankNicolsonConvergence:
    """Test convergence properties."""
    
    def test_time_step_convergence(self):
        """Test convergence with decreasing time steps."""
        D = 1e-10
        dx = 0.0001
        domain_length = 0.01
        final_time = 200.0
        
        # Run with different time steps
        dt_values = [50.0, 25.0, 12.5]
        solutions = []
        
        for dt in dt_values:
            solver = CrankNicolsonSolver(
                domain_length=domain_length,
                diffusion_coeff=D,
                dx=dx,
                dt=dt,
                boundary_left=1.0,
                boundary_right=0.0
            )
            
            C_init = np.zeros(solver.nx)
            x, t, C, _ = solver.solve(C_init, final_time=final_time)
            solutions.append(C[:, -1])
        
        # Check that solutions converge (difference decreases)
        diff_1 = rmse(solutions[0], solutions[1])
        diff_2 = rmse(solutions[1], solutions[2])
        
        # Difference should decrease with finer time steps
        assert diff_2 < diff_1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

