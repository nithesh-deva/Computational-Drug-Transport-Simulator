"""
Unit tests for M3 multilayer tissue solver.
Tests multilayer domain, interface conditions, and analytical solutions.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdts.geometry.multilayer import (
    MultilayerDomain, LayerInterface, validate_partition_coefficient,
    calculate_interface_concentration, verify_flux_continuity
)
from cdts.solvers.multilayer_fdm import MultilayerExplicitFDMSolver
from cdts.validation.two_layer import (
    two_layer_uncoupled_solution, calculate_interface_boundary_flux_matching,
    verify_two_layer_mass_conservation
)


class TestMultilayerDomain:
    """Test multilayer domain representation."""
    
    def test_two_layer_domain_creation(self):
        """Test creation of two-layer domain."""
        layers = [
            {'name': 'layer1', 'thickness': 0.005, 'diffusion_coefficient': 1e-10, 'clearance': 0.001},
            {'name': 'layer2', 'thickness': 0.005, 'diffusion_coefficient': 5e-11, 'clearance': 0.002}
        ]
        
        domain = MultilayerDomain(layers)
        
        assert domain.n_layers == 2
        assert np.isclose(domain.total_length, 0.01)
        assert len(domain.interface_positions) == 3
    
    def test_interface_positions(self):
        """Test interface position calculation."""
        layers = [
            {'name': 'L1', 'thickness': 0.002, 'diffusion_coefficient': 1e-10, 'clearance': 0.0},
            {'name': 'L2', 'thickness': 0.003, 'diffusion_coefficient': 1e-10, 'clearance': 0.0},
            {'name': 'L3', 'thickness': 0.005, 'diffusion_coefficient': 1e-10, 'clearance': 0.0}
        ]
        
        domain = MultilayerDomain(layers)
        
        expected_positions = [0.0, 0.002, 0.005, 0.01]
        assert np.allclose(domain.interface_positions, expected_positions)
    
    def test_mesh_generation(self):
        """Test mesh generation with interface resolution."""
        layers = [
            {'name': 'L1', 'thickness': 0.01, 'diffusion_coefficient': 1e-10, 'clearance': 0.0},
            {'name': 'L2', 'thickness': 0.01, 'diffusion_coefficient': 1e-10, 'clearance': 0.0}
        ]
        
        domain = MultilayerDomain(layers)
        dx = 0.001
        x, layer_indices, interface_indices = domain.generate_mesh(dx)
        
        # Check mesh properties
        assert len(x) > 0
        assert len(layer_indices) == len(x)
        assert len(interface_indices) > 0
        assert x[0] >= 0
        assert x[-1] <= domain.total_length
    
    def test_layer_property_retrieval(self):
        """Test retrieval of layer properties."""
        layers = [
            {'name': 'L1', 'thickness': 0.005, 'diffusion_coefficient': 1e-10, 'clearance': 0.001},
            {'name': 'L2', 'thickness': 0.005, 'diffusion_coefficient': 2e-10, 'clearance': 0.002}
        ]
        
        domain = MultilayerDomain(layers)
        
        # Get properties
        D1 = domain.get_layer_property(0, 'diffusion_coefficient')
        D2 = domain.get_layer_property(1, 'diffusion_coefficient')
        k1 = domain.get_layer_property(0, 'clearance')
        
        assert D1 == 1e-10
        assert D2 == 2e-10
        assert k1 == 0.001


class TestInterfaceConditions:
    """Test interface condition handling."""
    
    def test_partition_coefficient_validation(self):
        """Test partition coefficient validation."""
        # Valid
        validate_partition_coefficient(1.0, 'epithelium', 'dermis')
        validate_partition_coefficient(0.5, 'epithelium', 'dermis')
        validate_partition_coefficient(2.0, 'epithelium', 'dermis')
        
        # Invalid
        with pytest.raises(ValueError):
            validate_partition_coefficient(-1.0, 'L1', 'L2')
        
        with pytest.raises(ValueError):
            validate_partition_coefficient(0.0, 'L1', 'L2')
    
    def test_interface_concentration_calculation(self):
        """Test concentration calculation at interface."""
        K = 0.5
        C_left = 1.0
        
        # With partition
        C_right = calculate_interface_concentration(C_left, K, 'partition')
        assert np.isclose(C_right, 0.5)
        
        # With continuity
        C_right = calculate_interface_concentration(C_left, K, 'continuity')
        assert np.isclose(C_right, 1.0)
    
    def test_flux_continuity_verification(self):
        """Test flux continuity verification."""
        D_left = 1e-10
        D_right = 1e-10
        K = 1.0
        dC_dx_left = 100.0
        dC_dx_right = 100.0
        
        is_continuous, error = verify_flux_continuity(
            D_left, D_right, K, dC_dx_left, dC_dx_right, tolerance=0.01
        )
        
        assert is_continuous
        assert error < 0.01
    
    def test_interface_boundary_flux_matching(self):
        """Test interface flux matching calculation."""
        D1 = 1e-10
        D2 = 2e-10  # Different diffusion
        K = 0.5
        dC1_dx = 100.0
        C1 = 1.0
        
        C2, dC2_dx = calculate_interface_boundary_flux_matching(D1, D2, K, dC1_dx, C1)
        
        # Check concentration partition
        assert np.isclose(C2, 0.5)
        
        # Check flux relationship: D1·dC1/dx = D2·dC2/dx
        flux1 = D1 * dC1_dx
        flux2 = D2 * dC2_dx
        assert np.isclose(flux1, flux2, rtol=1e-10)


class TestTwoLayerAnalytical:
    """Test two-layer analytical solutions."""
    
    def test_two_layer_uncoupled_solution(self):
        """Test uncoupled two-layer solution."""
        x = np.linspace(0, 0.02, 201)
        t = np.array([0.0, 0.1, 1.0])
        
        D1 = 1e-10
        D2 = 1e-10
        k1 = 0.0
        k2 = 0.0
        L1 = 0.01
        L2 = 0.01
        K = 1.0
        
        C1, C2 = two_layer_uncoupled_solution(
            x, t, D1, D2, k1, k2, L1, L2, K,
            C_init_1=1.0, C_init_2=0.0
        )
        
        # Check shape
        assert C1.shape == (len(x), len(t))
        assert C2.shape == (len(x), len(t))
        
        # At t=0, check initial conditions are set in first layer
        x1_mask = x <= L1
        assert np.allclose(C1[x1_mask, 0], 1.0, atol=1e-2)
        
        # Layer 2 should start at 0
        x2_mask = x > L1
        assert np.allclose(C2[x2_mask, 0], 0.0, atol=1e-2)
        
        # With K=1, partition should not strongly affect solution structure
        assert C1.shape == C2.shape
    
    def test_two_layer_with_partition(self):
        """Test two-layer solution with partition coefficient."""
        x = np.linspace(0, 0.02, 201)
        t = np.array([0.0, 1.0])
        
        D1 = 1e-10
        D2 = 1e-10
        k1 = 0.0
        k2 = 0.0
        L1 = 0.01
        L2 = 0.01
        K = 0.5  # Layer 2 has half concentration
        
        C1, C2 = two_layer_uncoupled_solution(
            x, t, D1, D2, k1, k2, L1, L2, K,
            C_init_1=1.0, C_init_2=0.0
        )
        
        # Layer 2 should have lower concentrations due to partition
        assert np.mean(C2[:, 1]) < np.mean(C1[:, 1])
    
    def test_two_layer_mass_conservation(self):
        """Test mass conservation in two-layer system."""
        x = np.linspace(0, 0.02, 101)
        t = np.array([0.0, 0.5, 1.0, 2.0])
        
        D1 = 1e-10
        D2 = 1e-10
        k1 = 0.001
        k2 = 0.001
        L1 = 0.01
        L2 = 0.01
        K = 1.0
        
        C1, C2 = two_layer_uncoupled_solution(
            x, t, D1, D2, k1, k2, L1, L2, K,
            C_init_1=1.0, C_init_2=0.0
        )
        
        # Check mass conservation
        total_mass, expected_loss = verify_two_layer_mass_conservation(x, C1, C2, L1, k1, k2)
        
        # With clearance, mass should decrease
        assert total_mass[-1] < total_mass[0]


class TestMultilayerSolver:
    """Test multilayer FDM solver."""
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        layers = [
            {'name': 'L1', 'thickness': 0.01, 'diffusion_coefficient': 1e-10, 'clearance': 0.0},
            {'name': 'L2', 'thickness': 0.01, 'diffusion_coefficient': 1e-10, 'clearance': 0.0}
        ]
        
        domain = MultilayerDomain(layers)
        x, layer_indices, interface_indices = domain.generate_mesh(0.0001)
        
        solver = MultilayerExplicitFDMSolver(
            domain=domain,
            x=x,
            layer_indices=layer_indices,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0,
            interface_partitions={'L1_L2': 1.0}
        )
        
        assert solver.nx == len(x)
        assert solver.domain.n_layers == 2
    
    def test_multilayer_solver_execution(self):
        """Test multilayer solver execution."""
        layers = [
            {'name': 'L1', 'thickness': 0.01, 'diffusion_coefficient': 1e-10, 'clearance': 0.0},
            {'name': 'L2', 'thickness': 0.01, 'diffusion_coefficient': 1e-10, 'clearance': 0.0}
        ]
        
        domain = MultilayerDomain(layers)
        x, layer_indices, interface_indices = domain.generate_mesh(0.0001)
        
        solver = MultilayerExplicitFDMSolver(
            domain=domain,
            x=x,
            layer_indices=layer_indices,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0,
            interface_partitions={'L1_L2': 1.0}
        )
        
        C_init = np.zeros(solver.nx)
        x_out, t_out, C_out, metrics = solver.solve(C_init, final_time=100.0, interface_indices=interface_indices)
        
        # Check output shapes
        assert x_out.shape == x.shape
        assert len(t_out) > 1
        assert C_out.shape == (len(x), len(t_out))
        
        # Check boundary conditions enforced
        assert np.isclose(C_out[0, -1], 1.0)
        assert np.isclose(C_out[-1, -1], 0.0)
    
    def test_multilayer_vs_single_layer(self):
        """Test that two identical layers reduce to single layer."""
        # Two layer system
        layers_two = [
            {'name': 'L1', 'thickness': 0.005, 'diffusion_coefficient': 1e-10, 'clearance': 0.0},
            {'name': 'L2', 'thickness': 0.005, 'diffusion_coefficient': 1e-10, 'clearance': 0.0}
        ]
        
        domain_two = MultilayerDomain(layers_two)
        x_two, layer_idx_two, if_idx_two = domain_two.generate_mesh(0.0001)
        
        solver_two = MultilayerExplicitFDMSolver(
            domain=domain_two,
            x=x_two,
            layer_indices=layer_idx_two,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0,
            interface_partitions={'L1_L2': 1.0}
        )
        
        C_init = np.zeros(solver_two.nx)
        x2, t2, C2, m2 = solver_two.solve(C_init, final_time=100.0, interface_indices=if_idx_two)
        
        # Single layer system
        layers_one = [
            {'name': 'L1', 'thickness': 0.01, 'diffusion_coefficient': 1e-10, 'clearance': 0.0}
        ]
        
        domain_one = MultilayerDomain(layers_one)
        x_one, layer_idx_one, if_idx_one = domain_one.generate_mesh(0.0001)
        
        from cdts.solvers.explicit_fdm import ExplicitFDMSolver
        solver_one = ExplicitFDMSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=10.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0
        )
        
        C_init_one = np.zeros(solver_one.nx)
        x1, t1, C1, m1 = solver_one.solve(C_init_one, final_time=100.0)
        
        # Solutions should be very similar
        # (allowing for minor differences in mesh/grid generation)
        assert np.allclose(C1[:, -1], C2[:, -1], rtol=0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
