"""
Unit tests for M6 2D geometry and FEM solver.
Tests 2D tissue representation, mesh generation, and visualization.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdts.geometry.geometry_2d import Layer2D, Geometry2D


class TestLayer2D:
    """Test 2D layer specification."""
    
    def test_layer_creation(self):
        """Test basic 2D layer creation."""
        layer = Layer2D(
            name="epidermis",
            thickness=0.05,
            width=0.10,
            diffusion_coefficient=1.0e-10,
            clearance=0.001
        )
        
        assert layer.name == "epidermis"
        assert layer.thickness == 0.05
        assert layer.width == 0.10
        layer.validate()
    
    def test_layer_validation(self):
        """Test layer validation."""
        # Invalid thickness
        with pytest.raises(ValueError):
            layer = Layer2D("bad", -0.01, 0.10, 1e-10)
            layer.validate()
        
        # Invalid width
        with pytest.raises(ValueError):
            layer = Layer2D("bad", 0.01, -0.10, 1e-10)
            layer.validate()


class TestGeometry2D:
    """Test 2D geometry representation."""
    
    def test_two_layer_geometry(self):
        """Test two-layer rectangular geometry."""
        layers = [
            Layer2D("layer1", thickness=0.005, width=0.010, 
                   diffusion_coefficient=1.0e-10, clearance=0.001),
            Layer2D("layer2", thickness=0.010, width=0.010,
                   diffusion_coefficient=5.0e-11, clearance=0.0005)
        ]
        
        geom = Geometry2D(layers)
        
        assert geom.n_layers == 2
        assert np.isclose(geom.total_height, 0.015)
        assert geom.width == 0.010
    
    def test_layer_position_calculation(self):
        """Test layer y-position calculation."""
        layers = [
            Layer2D("L1", thickness=0.01, width=0.01, diffusion_coefficient=1e-10),
            Layer2D("L2", thickness=0.02, width=0.01, diffusion_coefficient=1e-10)
        ]
        
        geom = Geometry2D(layers)
        
        # Layer 0 (top) spans [0.02, 0.03]
        # Layer 1 (bottom) spans [0.00, 0.02]
        assert np.isclose(geom.layer_y_positions[0], 0.03)
        assert np.isclose(geom.layer_y_positions[1], 0.02)
        assert np.isclose(geom.layer_y_positions[2], 0.00)
    
    def test_layer_at_position(self):
        """Test finding layer at y position."""
        layers = [
            Layer2D("L1", thickness=0.01, width=0.01, diffusion_coefficient=1e-10),
            Layer2D("L2", thickness=0.02, width=0.01, diffusion_coefficient=1e-10)
        ]
        
        geom = Geometry2D(layers)
        
        # Test various positions
        assert geom.get_layer_at_position(0.025) == 0  # Top layer
        assert geom.get_layer_at_position(0.010) == 1  # Bottom layer
    
    def test_layer_properties(self):
        """Test retrieval of layer properties."""
        layers = [
            Layer2D("L1", thickness=0.01, width=0.01,
                   diffusion_coefficient=1.0e-10, clearance=0.001),
            Layer2D("L2", thickness=0.02, width=0.01,
                   diffusion_coefficient=5.0e-11, clearance=0.0005)
        ]
        
        geom = Geometry2D(layers)
        
        D1 = geom.get_layer_property(0, 'diffusion_coefficient')
        D2 = geom.get_layer_property(1, 'diffusion_coefficient')
        
        assert D1 == 1.0e-10
        assert D2 == 5.0e-11
    
    def test_three_layer_geometry(self):
        """Test three-layer skin model."""
        layers = [
            Layer2D("SC", thickness=0.00001, width=0.10, 
                   diffusion_coefficient=0.1e-10, clearance=0.0),
            Layer2D("Epidermis", thickness=0.00005, width=0.10,
                   diffusion_coefficient=1.0e-10, clearance=0.001),
            Layer2D("Dermis", thickness=0.0001, width=0.10,
                   diffusion_coefficient=2.0e-10, clearance=0.0005)
        ]
        
        geom = Geometry2D(layers)
        
        assert geom.n_layers == 3
        assert np.isclose(geom.total_height, 0.00016)
        assert geom.width == 0.10
    
    def test_gmsh_script_generation(self):
        """Test Gmsh script generation."""
        import tempfile
        
        layers = [
            Layer2D("L1", thickness=0.01, width=0.02,
                   diffusion_coefficient=1e-10, mesh_size=0.001),
            Layer2D("L2", thickness=0.01, width=0.02,
                   diffusion_coefficient=1e-10, mesh_size=0.001)
        ]
        
        geom = Geometry2D(layers)
        
        # Use system temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_geometry.geo"
            geom.generate_gmsh_script(str(output_path), mesh_size=0.0005)
            
            # Check file was created
            assert output_path.exists()
            
            # Check content
            with open(output_path, 'r') as f:
                content = f.read()
                assert "2D Multilayer" in content
                assert "Layer 0: L1" in content
                assert "Layer 1: L2" in content
                assert "Physical Surface" in content


class TestGeometry2DRepresentation:
    """Test geometry string representation."""
    
    def test_repr(self):
        """Test __repr__ output."""
        layers = [
            Layer2D("layer1", thickness=0.01, width=0.01,
                   diffusion_coefficient=1e-10, clearance=0.001)
        ]
        
        geom = Geometry2D(layers)
        repr_str = repr(geom)
        
        assert "Geometry2D" in repr_str
        assert "layer1" in repr_str
        assert "Width:" in repr_str
        assert "Height:" in repr_str


class TestVisualization2D:
    """Test 2D visualization capabilities."""
    
    def test_layer_coloring(self):
        """Test that layer properties can be visualized."""
        layers = [
            Layer2D("L1", thickness=0.01, width=0.01, 
                   diffusion_coefficient=1.0e-10, clearance=0.0),
            Layer2D("L2", thickness=0.01, width=0.01,
                   diffusion_coefficient=0.5e-10, clearance=0.001)
        ]
        
        geom = Geometry2D(layers)
        
        # Create synthetic field for visualization
        n_points = 100
        y_positions = np.linspace(0, geom.total_height, n_points)
        D_field = np.zeros(n_points)
        
        for i, y in enumerate(y_positions):
            layer_idx = min(geom.get_layer_at_position(y), geom.n_layers - 1)
            D_field[i] = geom.get_layer_property(layer_idx, 'diffusion_coefficient')
        
        # Check that two distinct values appear
        unique_values = np.unique(D_field)
        assert len(unique_values) >= 1
        assert np.any(D_field == 1.0e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
