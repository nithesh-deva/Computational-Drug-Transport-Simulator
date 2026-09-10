"""
Unit tests for M5 Finite Element Method solver.
Validates FEM implementation, convergence, and comparison with FDM/CN.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdts.solvers.fem_1d import FEMSolver1D, FEMAssembler1D, LinearLagrangeBasis1D
from cdts.solvers.explicit_fdm import ExplicitFDMSolver
from cdts.solvers.crank_nicolson import CrankNicolsonSolver
from cdts.validation.analytical import finite_domain_solution
from cdts.validation.metrics import rmse, maximum_absolute_error


class TestFEMBasis:
    """Test finite element basis functions."""
    
    def test_linear_lagrange_basis_properties(self):
        """Test linear Lagrange basis function properties."""
        nodes = np.array([0.0, 0.1, 0.2, 0.3])
        
        # Basis at node 1 (middle)
        basis = LinearLagrangeBasis1D(1, nodes)
        
        # Should be 1.0 at node 1
        assert np.isclose(basis.evaluate(np.array([0.1]))[0], 1.0)
        
        # Should be 0.0 at other nodes
        assert np.isclose(basis.evaluate(np.array([0.0]))[0], 0.0)
        assert np.isclose(basis.evaluate(np.array([0.2]))[0], 0.0)
    
    def test_linear_lagrange_derivative(self):
        """Test derivative of linear basis functions."""
        nodes = np.array([0.0, 0.1, 0.2])
        
        # Basis at node 0 (left)
        basis0 = LinearLagrangeBasis1D(0, nodes)
        
        # Derivative should be constant on element [0, 0.1]
        x_test = np.array([0.05])
        deriv = basis0.evaluate(x_test, derivatives=1)[0]
        
        # For linear basis on [0, 0.1], derivative should be -1/0.1 = -10
        assert np.isclose(deriv, -10.0)


class TestFEMAssembler:
    """Test FEM matrix assembly."""
    
    def test_mass_matrix_assembly(self):
        """Test mass matrix assembly."""
        nodes = np.array([0.0, 1.0])  # Single element
        assembler = FEMAssembler1D(nodes)
        M = assembler.assemble_mass_matrix()
        
        # For h=1: M_local = (1/6) * [2 1; 1 2]
        expected = (1.0/6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        
        assert np.allclose(M, expected)
    
    def test_stiffness_matrix_assembly(self):
        """Test stiffness matrix assembly."""
        nodes = np.array([0.0, 1.0])
        assembler = FEMAssembler1D(nodes)
        K = assembler.assemble_stiffness_matrix(diffusion_coeff=1.0)
        
        # For h=1, D=1: K_local = (1/1) * [1 -1; -1 1]
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        assert np.allclose(K, expected)
    
    def test_matrix_symmetry(self):
        """Test that assembled matrices are symmetric."""
        nodes = np.linspace(0, 1, 11)
        assembler = FEMAssembler1D(nodes)
        
        M = assembler.assemble_mass_matrix()
        K = assembler.assemble_stiffness_matrix(1.0)
        R = assembler.assemble_reaction_matrix(0.01)
        
        # All should be symmetric
        assert np.allclose(M, M.T)
        assert np.allclose(K, K.T)
        assert np.allclose(R, R.T)


class TestFEMSolver1D:
    """Test 1D FEM solver."""
    
    def test_solver_initialization(self):
        """Test FEM solver initialization."""
        solver = FEMSolver1D(
            domain_length=1.0,
            n_elements=10,
            diffusion_coeff=1.0,
            dt=0.01,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        assert solver.n_nodes == 11
        assert len(solver.x) == 11
        assert np.isclose(solver.x[0], 0.0)
        assert np.isclose(solver.x[-1], 1.0)
    
    def test_solver_execution(self):
        """Test basic FEM solver execution."""
        solver = FEMSolver1D(
            domain_length=0.01,
            n_elements=20,
            diffusion_coeff=1e-10,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.n_nodes)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)
        
        # Check output
        assert C.shape == (solver.n_nodes, len(t))
        assert np.isclose(C[0, -1], 1.0)  # Left BC
        assert np.isclose(C[-1, -1], 0.0)  # Right BC
        assert np.all(C >= -1e-12)  # Non-negative
    
    def test_solver_with_clearance(self):
        """Test FEM solver with clearance."""
        solver = FEMSolver1D(
            domain_length=0.01,
            n_elements=20,
            diffusion_coeff=1e-10,
            dt=50.0,
            boundary_left=0.0,
            boundary_right=0.0,
            clearance=0.001
        )
        
        # Uniform initial
        C_init = np.ones(solver.n_nodes)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)
        
        # Mass should decrease
        M_init = np.trapz(C_init, x)
        M_final = np.trapz(C[:, -1], x)
        
        assert M_final < M_init


class TestFEMConvergence:
    """Test FEM convergence properties."""
    
    def test_spatial_convergence(self):
        """Test spatial convergence with mesh refinement."""
        D = 1e-10
        dt = 50.0
        domain_length = 0.01
        final_time = 200.0
        
        # Run with different mesh sizes
        n_elements_list = [10, 20, 40]
        solutions = []
        
        for n_elem in n_elements_list:
            solver = FEMSolver1D(
                domain_length=domain_length,
                n_elements=n_elem,
                diffusion_coeff=D,
                dt=dt,
                boundary_left=1.0,
                boundary_right=0.0
            )
            
            C_init = np.zeros(solver.n_nodes)
            x, t, C, _ = solver.solve(C_init, final_time=final_time)
            solutions.append((x, C[:, -1]))
        
        # Interpolate to common grid
        x_ref = solutions[-1][0]
        C_10 = np.interp(x_ref, solutions[0][0], solutions[0][1])
        C_20 = np.interp(x_ref, solutions[1][0], solutions[1][1])
        C_40 = solutions[-1][1]
        
        # Error should decrease with refinement
        error_10 = rmse(C_10, C_40)
        error_20 = rmse(C_20, C_40)
        
        assert error_20 < error_10


class TestFEMvsFDMComparison:
    """Compare FEM with FDM and Crank-Nicolson."""
    
    def test_fem_vs_explicit_fdm(self):
        """Compare FEM with explicit FDM at small time step."""
        D = 1e-10
        dx = 0.0001
        dt = 5.0
        domain_length = 0.01
        
        # FEM solver
        n_elements_fem = 100  # ~same resolution as FDM
        solver_fem = FEMSolver1D(
            domain_length=domain_length,
            n_elements=n_elements_fem,
            diffusion_coeff=D,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        # FDM solver
        solver_fdm = ExplicitFDMSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=dx,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init_fem = np.zeros(solver_fem.n_nodes)
        C_init_fdm = np.zeros(solver_fdm.nx)
        
        x_fem, t_fem, C_fem, _ = solver_fem.solve(C_init_fem, final_time=100.0)
        x_fdm, t_fdm, C_fdm, _ = solver_fdm.solve(C_init_fdm, final_time=100.0)
        
        # Interpolate FEM to FDM grid
        C_fem_interp = np.interp(x_fdm, x_fem, C_fem[:, -1])
        
        # Should be similar
        error = rmse(C_fem_interp, C_fdm[:, -1])
        assert error < np.max(C_fdm[:, -1]) * 0.2  # < 20% error
    
    def test_fem_vs_crank_nicolson(self):
        """Compare FEM with Crank-Nicolson at large time step."""
        D = 1e-10
        dt = 100.0
        domain_length = 0.01
        
        # FEM solver
        solver_fem = FEMSolver1D(
            domain_length=domain_length,
            n_elements=50,
            diffusion_coeff=D,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        # Crank-Nicolson solver
        solver_cn = CrankNicolsonSolver(
            domain_length=domain_length,
            diffusion_coeff=D,
            dx=domain_length / 50,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init_fem = np.zeros(solver_fem.n_nodes)
        C_init_cn = np.zeros(solver_cn.nx)
        
        x_fem, t_fem, C_fem, _ = solver_fem.solve(C_init_fem, final_time=300.0)
        x_cn, t_cn, C_cn, _ = solver_cn.solve(C_init_cn, final_time=300.0)
        
        # Interpolate FEM to CN grid
        C_fem_interp = np.interp(x_cn, x_fem, C_fem[:, -1])
        
        # Should be similar
        error = rmse(C_fem_interp, C_cn[:, -1])
        assert error < np.max(C_cn[:, -1]) * 0.3  # < 30% error


class TestFEMAccuracy:
    """Test FEM accuracy against analytical solutions."""
    
    def test_analytical_comparison(self):
        """Compare FEM with analytical solution."""
        D = 1e-10
        domain_length = 0.01
        
        # FEM solver
        solver = FEMSolver1D(
            domain_length=domain_length,
            n_elements=50,
            diffusion_coeff=D,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.n_nodes)
        x, t, C_fem, _ = solver.solve(C_init, final_time=200.0)
        
        # Analytical solution
        C_ana = finite_domain_solution(x, t, D, domain_length, 1.0, 0.0, 0.0)
        
        # Compare
        error = rmse(C_fem[:, -1], C_ana[:, -1])
        max_error = maximum_absolute_error(C_fem[:, -1], C_ana[:, -1])
        
        # FEM should be reasonably accurate
        assert error < np.max(C_ana[:, -1]) * 0.15


class TestFEMStability:
    """Test FEM stability properties."""
    
    def test_stability_large_timestep(self):
        """Test that FEM is stable with large time steps."""
        D = 1e-10
        dt = 500.0  # Very large
        domain_length = 0.01
        
        solver = FEMSolver1D(
            domain_length=domain_length,
            n_elements=30,
            diffusion_coeff=D,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.n_nodes)
        x, t, C, _ = solver.solve(C_init, final_time=1000.0)
        
        # Should not have NaN/Inf
        assert np.all(np.isfinite(C))
        # Should be bounded
        assert np.all(C >= -1e-10)
        assert np.all(C <= 1.0 + 1e-10)
    
    def test_monotone_behavior(self):
        """Test physically monotone FEM solution."""
        D = 1e-10
        dt = 50.0
        domain_length = 0.01
        
        solver = FEMSolver1D(
            domain_length=domain_length,
            n_elements=40,
            diffusion_coeff=D,
            dt=dt,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.n_nodes)
        x, t, C, _ = solver.solve(C_init, final_time=500.0)
        
        # Final solution should be monotone
        C_final = C[:, -1]
        grad = np.diff(C_final)
        
        # >70% of gradients should be negative (diffusion-driven)
        assert np.sum(grad < 0) > len(grad) * 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
