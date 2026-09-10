"""
Unit tests for CDTS core modules.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdts.config import (
    ExperimentConfig, DrugSpecification, TissueSpecification, TissueLayer,
    SimulationParameters, SolverSpecification, BoundaryCondition,
    load_config_from_yaml
)
from cdts.numerics.stability import check_explicit_fdm_stability, get_max_stable_dt
from cdts.validation.analytical import (
    semi_infinite_domain_solution, finite_domain_solution
)
from cdts.validation.metrics import (
    rmse, relative_error, maximum_absolute_error, mass_conservation_check,
    convergence_order_analysis
)
from cdts.solvers.explicit_fdm import ExplicitFDMSolver


class TestConfiguration:
    """Test configuration parsing and validation."""
    
    def test_boundary_condition_validation(self):
        """Test boundary condition validation."""
        # Valid conditions
        bc1 = BoundaryCondition(type='dirichlet', value=1.0)
        bc1.validate()
        
        bc2 = BoundaryCondition(type='neumann', value=0.0)
        bc2.validate()
        
        # Invalid type
        with pytest.raises(ValueError):
            bc3 = BoundaryCondition(type='invalid', value=0.0)
            bc3.validate()
        
        # Negative value
        with pytest.raises(ValueError):
            bc4 = BoundaryCondition(type='dirichlet', value=-1.0)
            bc4.validate()
    
    def test_drug_specification_validation(self):
        """Test drug specification validation."""
        drug = DrugSpecification(initial_concentration=1.0)
        drug.validate()
        
        # Negative concentration
        with pytest.raises(ValueError):
            drug_bad = DrugSpecification(initial_concentration=-1.0)
            drug_bad.validate()
    
    def test_tissue_layer_validation(self):
        """Test tissue layer validation."""
        layer = TissueLayer(
            name='test_layer',
            thickness=0.01,
            diffusion_coefficient=1e-10,
            clearance=0.001
        )
        layer.validate()
        
        # Invalid thickness
        with pytest.raises(ValueError):
            layer_bad = TissueLayer(
                name='bad_layer',
                thickness=-0.01,
                diffusion_coefficient=1e-10
            )
            layer_bad.validate()
        
        # Negative diffusion coefficient
        with pytest.raises(ValueError):
            layer_bad2 = TissueLayer(
                name='bad_layer2',
                thickness=0.01,
                diffusion_coefficient=-1e-10
            )
            layer_bad2.validate()
    
    def test_yaml_configuration_loading(self):
        """Test YAML configuration loading."""
        config_data = {
            'experiment': {'id': 'TEST001', 'name': 'Test Experiment'},
            'drug': {'initial_concentration': 1.0, 'unit': 'mol/m3'},
            'tissue': {
                'layers': [
                    {
                        'name': 'layer1',
                        'thickness': 0.01,
                        'diffusion_coefficient': 1e-10,
                        'clearance': 0.0
                    }
                ]
            },
            'simulation': {
                'final_time': 3600.0,
                'spatial_step': 0.0001,
                'time_step': 10.0
            },
            'solver': {'method': 'explicit_fdm'},
            'boundary': {
                'left': {'type': 'dirichlet', 'value': 1.0},
                'right': {'type': 'dirichlet', 'value': 0.0}
            },
            'output': {'directory': 'results/TEST001'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            config = load_config_from_yaml(yaml_path)
            assert config.experiment_id == 'TEST001'
            assert config.name == 'Test Experiment'
            assert config.drug.initial_concentration == 1.0
            assert len(config.tissue.layers) == 1
        finally:
            Path(yaml_path).unlink()


class TestStabilityAnalysis:
    """Test numerical stability checks."""
    
    def test_stability_check_stable(self):
        """Test stable configuration."""
        D = 1e-10  # m²/s
        dx = 0.0001  # m
        dt = 10.0  # s
        
        is_stable = check_explicit_fdm_stability(D, dx, dt)
        assert is_stable, "Configuration should be stable"
    
    def test_stability_check_unstable(self):
        """Test unstable configuration."""
        D = 1e-10  # m²/s
        dx = 0.0001  # m
        dt = 1000.0  # s (too large)
        
        is_stable = check_explicit_fdm_stability(D, dx, dt)
        assert not is_stable, "Configuration should be unstable"
    
    def test_max_stable_dt(self):
        """Test maximum stable time step calculation."""
        D = 1e-10  # m²/s
        dx = 0.0001  # m
        
        dt_max = get_max_stable_dt(D, dx)
        
        # Check that dt_max gives r = 0.5
        r = D * dt_max / (dx ** 2)
        assert np.isclose(r, 0.5), f"Expected r=0.5, got r={r}"


class TestAnalyticalSolutions:
    """Test analytical solution implementations."""
    
    def test_semi_infinite_domain_solution(self):
        """Test semi-infinite domain analytical solution."""
        x = np.linspace(0, 0.01, 101)
        t = np.array([0.1, 1.0, 10.0])
        D = 1e-10
        C0 = 1.0
        
        C_ana = semi_infinite_domain_solution(x, t, D, C0)
        
        # Check shape
        assert C_ana.shape == (len(x), len(t))
        
        # Check boundary condition
        assert np.isclose(C_ana[0, 1], C0), "Boundary condition not satisfied"
        
        # Check concentration decreases with distance
        for n in range(len(t)):
            assert np.all(np.diff(C_ana[:, n]) <= 0), "Concentration should decrease with distance"
    
    def test_finite_domain_solution(self):
        """Test finite domain analytical solution."""
        x = np.linspace(0, 0.01, 101)
        t = np.array([0.01, 0.1, 1.0])
        D = 1e-10
        L = 0.01
        C_left = 0.0
        C_right = 0.0
        C_init = 1.0
        
        C_ana = finite_domain_solution(x, t, D, L, C_left, C_right, C_init)
        
        # Check shape
        assert C_ana.shape == (len(x), len(t))
        
        # Check boundary conditions
        assert np.isclose(C_ana[0, 1], C_left), "Left BC not satisfied"
        assert np.isclose(C_ana[-1, 1], C_right), "Right BC not satisfied"
        
        # Check decay over time
        center_idx = len(x) // 2
        for n in range(len(t) - 1):
            assert C_ana[center_idx, n+1] <= C_ana[center_idx, n], "Concentration should decay"


class TestErrorMetrics:
    """Test error metrics and validation."""
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        C_num = np.array([1.0, 2.0, 3.0])
        C_ana = np.array([1.1, 2.1, 2.9])
        
        error = rmse(C_num, C_ana)
        expected = np.sqrt((0.1**2 + 0.1**2 + 0.1**2) / 3)
        
        assert np.isclose(error, expected)
    
    def test_relative_error(self):
        """Test relative error calculation."""
        C_num = np.array([1.0, 2.0, 3.0])
        C_ana = np.array([1.0, 2.0, 3.0])
        
        rel_err = relative_error(C_num, C_ana)
        
        # No error case
        assert np.all(rel_err < 1e-10)
    
    def test_maximum_absolute_error(self):
        """Test maximum absolute error."""
        C_num = np.array([1.0, 2.0, 3.0])
        C_ana = np.array([1.1, 2.2, 2.9])
        
        max_err = maximum_absolute_error(C_num, C_ana)
        expected = 0.2
        
        assert np.isclose(max_err, expected)
    
    def test_convergence_order_analysis(self):
        """Test convergence order estimation."""
        # Create synthetic convergence data (2nd order)
        h = np.array([0.1, 0.05, 0.025, 0.0125])
        error = 0.01 * h ** 2  # Second order
        
        order, r_sq = convergence_order_analysis(error, h)
        
        assert np.isclose(order, 2.0, atol=0.1), f"Expected order ~2, got {order}"
        assert r_sq > 0.99, f"Expected good fit, got r²={r_sq}"


class TestSolver:
    """Test explicit FDM solver."""
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        assert solver.nx == 101
        assert np.isclose(solver.r, 0.1)
    
    def test_solver_unstable_initialization(self):
        """Test that unstable solver raises error."""
        with pytest.raises(ValueError):
            solver = ExplicitFDMSolver(
                domain_length=0.01,
                diffusion_coeff=1e-10,
                dx=0.0001,
                dt=10000.0,  # Unstable time step
                boundary_left=1.0,
                boundary_right=0.0
            )
    
    def test_solver_basic_run(self):
        """Test basic solver execution."""
        solver = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0
        )
        
        C_init = np.zeros(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=100.0)
        
        # Check shapes
        assert x.shape == (solver.nx,)
        assert t.shape[0] > 1
        assert C.shape == (solver.nx, len(t))
        
        # Check boundary conditions are enforced
        assert np.isclose(C[0, -1], 1.0), "Left boundary not enforced"
        assert np.isclose(C[-1, -1], 0.0), "Right boundary not enforced"
        
        # Check non-negative concentrations
        assert np.all(C >= 0), "Concentration became negative"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
