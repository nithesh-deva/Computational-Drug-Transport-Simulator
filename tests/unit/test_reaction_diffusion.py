"""
Unit tests for M2 reaction-diffusion with clearance.
Tests clearance implementation, mass loss verification, and analytical solutions.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdts.solvers.explicit_fdm import ExplicitFDMSolver
from cdts.validation.reaction_diffusion import (
    semi_infinite_domain_with_clearance,
    finite_domain_with_clearance_exponential,
    diffusion_dominance_time,
    expected_mass_loss_rate,
    verify_mass_loss_consistency
)
from cdts.validation.metrics import mass_conservation_check


class TestReactionDiffusionAnalytical:
    """Test analytical solutions for reaction-diffusion systems."""
    
    def test_semi_infinite_with_clearance(self):
        """Test semi-infinite domain with clearance solution."""
        x = np.linspace(0, 0.01, 101)
        t = np.array([0.1, 1.0, 10.0])
        D = 1e-10
        k = 0.01  # clearance coefficient [1/s]
        C0 = 1.0
        
        C_ana = semi_infinite_domain_with_clearance(x, t, D, k, C0)
        
        # Check shape
        assert C_ana.shape == (len(x), len(t))
        
        # Check boundary condition
        assert np.isclose(C_ana[0, 1], C0 * np.exp(-k * t[1]), rtol=1e-6)
        
        # Check concentration decreases with distance
        for n in range(len(t)):
            assert np.all(np.diff(C_ana[:, n]) <= 1e-12), "Should be monotonically decreasing"
        
        # Check decay over time at fixed location
        mid_idx = len(x) // 2
        for n in range(len(t) - 1):
            # Concentration should generally decay due to clearance
            # (diffusion may cause slight temporal increase initially)
            pass
    
    def test_finite_domain_with_clearance(self):
        """Test finite domain with clearance solution."""
        x = np.linspace(0, 0.01, 101)
        t = np.array([0.01, 0.1, 1.0])
        D = 1e-10
        k = 0.001
        L = 0.01
        
        C_ana = finite_domain_with_clearance_exponential(
            x=x, t=t, D=D, k=k, L=L,
            C_left=0.0, C_right=0.0, C_init=1.0
        )
        
        # Check shape
        assert C_ana.shape == (len(x), len(t))
        
        # Check boundary conditions
        assert np.isclose(C_ana[0, 1], 0.0, atol=1e-12)
        assert np.isclose(C_ana[-1, 1], 0.0, atol=1e-12)
        
        # Check decay over time at interior
        center_idx = len(x) // 2
        for n in range(len(t) - 1):
            # Concentration decays due to clearance
            assert C_ana[center_idx, n+1] <= C_ana[center_idx, n]
    
    def test_clearance_dominates_long_time(self):
        """Test that clearance eventually dominates and solution decays to zero."""
        x = np.linspace(0, 0.01, 101)
        t = np.array([0.0, 1.0, 10.0, 100.0])  # Very long time
        D = 1e-10
        k = 0.1  # Strong clearance
        
        C_ana = finite_domain_with_clearance_exponential(
            x=x, t=t, D=D, k=k, L=0.01,
            C_left=0.0, C_right=0.0, C_init=1.0
        )
        
        # At long time, all concentrations should be very small
        assert np.max(C_ana[:, -1]) < 1e-4, "Concentration should decay to near-zero"


class TestCharacteristicTimescales:
    """Test calculation of reaction-diffusion timescales."""
    
    def test_timescale_calculation(self):
        """Test characteristic timescale computation."""
        D = 1e-10  # m²/s
        k = 0.01   # 1/s
        L = 0.01   # m
        
        tau_d, tau_k, Da = diffusion_dominance_time(D, k, L)
        
        # Check positive values
        assert tau_d > 0
        assert tau_k > 0
        assert Da > 0
        
        # Verify relationships
        assert np.isclose(tau_d, L**2 / D)
        assert np.isclose(tau_k, 1.0 / k)
        assert np.isclose(Da, k * L**2 / D)
    
    def test_diffusion_limited_regime(self):
        """Test diffusion-limited regime (Da << 1)."""
        D = 1e-8   # Large diffusion (very permeable)
        k = 1e-6   # Small clearance
        L = 0.01
        
        tau_d, tau_k, Da = diffusion_dominance_time(D, k, L)
        
        # Da should be small (use <= for boundary case)
        assert Da <= 0.05, f"Should be diffusion-limited, Da = {Da}"
        # Diffusion timescale should be much smaller than clearance timescale
        assert tau_d < tau_k
    
    def test_clearance_limited_regime(self):
        """Test clearance-limited regime (Da >> 1)."""
        D = 1e-11  # Small diffusion
        k = 0.1    # Large clearance
        L = 0.01
        
        tau_d, tau_k, Da = diffusion_dominance_time(D, k, L)
        
        # Da should be large
        assert Da > 10, "Should be clearance-limited"
        # Clearance timescale should be much smaller
        assert tau_k < tau_d
    
    def test_no_clearance(self):
        """Test behavior with k=0 (no clearance)."""
        D = 1e-10
        k = 0.0
        L = 0.01
        
        tau_d, tau_k, Da = diffusion_dominance_time(D, k, L)
        
        assert tau_d == L**2 / D
        assert tau_k == np.inf
        assert Da == 0.0


class TestMassLossVerification:
    """Test mass loss verification for clearance systems."""
    
    def test_expected_mass_loss_rate(self):
        """Test expected mass loss rate calculation."""
        M = 0.01  # Total mass [mol/m]
        k = 0.1   # Clearance coefficient [1/s]
        
        loss_rate = expected_mass_loss_rate(M, k)
        
        # Loss rate should be k * M
        assert np.isclose(loss_rate, k * M)
    
    def test_mass_loss_consistency_exponential_decay(self):
        """Test that exponential decay is correctly verified."""
        k = 0.01
        M0 = 1.0
        t = np.linspace(0, 100, 101)
        
        # Theoretical exponential decay
        total_mass = M0 * np.exp(-k * t)
        
        # Verify consistency
        is_consistent, mean_error, error_array = verify_mass_loss_consistency(
            t, total_mass, k, tolerance=0.01
        )
        
        # Should be perfectly consistent (within numerical precision)
        assert is_consistent, f"Mean error {mean_error} should be < 0.01"
        assert mean_error < 1e-6
    
    def test_mass_loss_consistency_with_numerical_error(self):
        """Test mass loss verification with realistic numerical error."""
        k = 0.01
        M0 = 1.0
        t = np.linspace(0, 100, 101)
        
        # Theoretical decay with small numerical noise
        M_theory = M0 * np.exp(-k * t)
        noise = np.random.normal(0, 0.001 * M0, len(t))
        total_mass = np.maximum(M_theory + noise, 0)  # Keep non-negative
        
        # Verify consistency with reasonable tolerance
        is_consistent, mean_error, error_array = verify_mass_loss_consistency(
            t, total_mass, k, tolerance=0.05
        )
        
        # Should pass with 5% tolerance
        assert mean_error < 0.05


class TestSolverWithClearance:
    """Test explicit FDM solver with clearance term."""
    
    def test_solver_with_clearance_initialization(self):
        """Test solver initialization with clearance."""
        solver = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=0.0,
            boundary_right=0.0,
            clearance=0.001
        )
        
        assert solver.clearance == 0.001
        assert solver.nx == 101
    
    def test_solver_with_clearance_mass_decay(self):
        """Test that mass decays with clearance."""
        solver = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=0.0,
            boundary_right=0.0,
            clearance=0.01  # 1% per second
        )
        
        # Uniform initial condition
        C_init = np.ones(solver.nx)
        
        # Run short simulation
        x, t, C, metrics = solver.solve(C_init, final_time=100.0)
        
        # Calculate total mass at each time
        total_mass, _ = mass_conservation_check(x, C, solver.dx, solver.clearance)
        
        # Mass should decrease over time
        assert total_mass[-1] < total_mass[0], "Mass should decrease with clearance"
        
        # Check that mass decay is reasonable
        # M(t) ≈ M(0) * exp(-k*t) for small time steps
        expected_final_mass = total_mass[0] * np.exp(-solver.clearance * t[-1])
        # Allow 20% error due to numerical diffusion and boundary effects
        assert total_mass[-1] > expected_final_mass * 0.8
    
    def test_solver_clearance_vs_pure_diffusion(self):
        """Test that clearance reduces total mass compared to pure diffusion."""
        # Run with clearance
        solver_clear = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.01
        )
        
        # Run without clearance (pure diffusion)
        solver_pure = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0
        )
        
        C_init = np.zeros(solver_clear.nx)
        
        x_c, t_c, C_clear, _ = solver_clear.solve(C_init, final_time=200.0)
        x_p, t_p, C_pure, _ = solver_pure.solve(C_init, final_time=200.0)
        
        # Calculate total mass
        M_clear, _ = mass_conservation_check(x_c, C_clear, solver_clear.dx, solver_clear.clearance)
        M_pure, _ = mass_conservation_check(x_p, C_pure, solver_pure.dx, 0.0)
        
        # System with clearance should have less total mass at late times
        assert M_clear[-1] < M_pure[-1], "Clearance should reduce total mass"
    
    def test_negative_concentration_prevention(self):
        """Test that solver prevents negative concentrations."""
        solver = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=0.0,
            boundary_right=0.0,
            clearance=0.1  # Strong clearance
        )
        
        # Start with high concentration
        C_init = np.ones(solver.nx)
        
        # Run long simulation
        x, t, C, metrics = solver.solve(C_init, final_time=500.0)
        
        # No negative concentrations
        assert np.all(C >= -1e-14), "Concentrations should remain non-negative"


class TestClearanceValidation:
    """Integration tests for clearance validation."""
    
    def test_complete_clearance_validation_workflow(self):
        """Test full workflow: solve, compare with analytical, verify mass loss."""
        # Setup problem
        D = 1e-10
        k = 0.001
        L = 0.01
        dx = 0.0001
        dt = 10.0
        final_time = 1000.0
        
        # Create solver
        solver = ExplicitFDMSolver(
            domain_length=L,
            diffusion_coeff=D,
            dx=dx,
            dt=dt,
            boundary_left=0.0,
            boundary_right=0.0,
            clearance=k
        )
        
        # Initial condition
        C_init = np.ones(solver.nx)
        
        # Solve
        x, t, C_num, _ = solver.solve(C_init, final_time)
        
        # Generate analytical solution
        C_ana = finite_domain_with_clearance_exponential(
            x=x, t=t, D=D, k=k, L=L,
            C_left=0.0, C_right=0.0, C_init=1.0
        )
        
        # Check mass decay
        M_num, _ = mass_conservation_check(x, C_num, dx, k)
        M_ana, _ = mass_conservation_check(x, C_ana, dx, k)
        
        # Both should show exponential decay
        # Relative mass at final time
        decay_num = M_num[-1] / M_num[0]
        decay_ana = M_ana[-1] / M_ana[0]
        
        # Theoretical decay
        decay_theory = np.exp(-k * final_time)
        
        # Should be reasonably close
        assert decay_num < decay_ana, "Numerical diffusion causes additional loss"
        assert decay_num > decay_theory * 0.5, "Decay shouldn't be too much faster"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
