"""
Configuration schema and validation for CDTS experiments.
"""

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BoundaryCondition:
    """Boundary condition specification.
    
    Attributes:
        type: 'dirichlet' or 'neumann'
        value: Boundary value or flux [mol/(m²·s)]
    """
    type: str
    value: float
    
    def validate(self):
        """Validate boundary condition."""
        if self.type not in ['dirichlet', 'neumann']:
            raise ValueError(f"Invalid boundary type: {self.type}. Must be 'dirichlet' or 'neumann'")
        if self.value < 0:
            raise ValueError(f"Boundary value must be non-negative, got {self.value}")


@dataclass
class DrugSpecification:
    """Drug properties specification.
    
    Attributes:
        initial_concentration: Initial drug concentration [mol/m³]
        unit: Unit string for documentation
    """
    initial_concentration: float
    unit: str = "mol/m3"
    
    def validate(self):
        """Validate drug specification."""
        if self.initial_concentration < 0:
            raise ValueError(f"Initial concentration must be non-negative, got {self.initial_concentration}")


@dataclass
class TissueLayer:
    """Single tissue layer specification.
    
    Attributes:
        name: Layer name
        thickness: Layer thickness [m]
        diffusion_coefficient: Diffusion coefficient [m²/s]
        clearance: First-order clearance coefficient [1/s]
    """
    name: str
    thickness: float
    diffusion_coefficient: float
    clearance: float = 0.0
    
    def validate(self):
        """Validate tissue layer."""
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.diffusion_coefficient < 0:
            raise ValueError(f"Diffusion coefficient must be non-negative, got {self.diffusion_coefficient}")
        if self.clearance < 0:
            raise ValueError(f"Clearance coefficient must be non-negative, got {self.clearance}")


@dataclass
class TissueSpecification:
    """Tissue specification.
    
    Attributes:
        layers: List of tissue layers
    """
    layers: list
    
    def validate(self):
        """Validate tissue specification."""
        if not self.layers:
            raise ValueError("At least one tissue layer is required")
        for layer in self.layers:
            layer.validate()


@dataclass
class SimulationParameters:
    """Simulation parameters.
    
    Attributes:
        final_time: Final simulation time [s]
        spatial_step: Spatial discretization step [m]
        time_step: Temporal discretization step [s]
    """
    final_time: float
    spatial_step: float
    time_step: float
    
    def validate(self):
        """Validate simulation parameters."""
        if self.final_time <= 0:
            raise ValueError(f"Final time must be positive, got {self.final_time}")
        if self.spatial_step <= 0:
            raise ValueError(f"Spatial step must be positive, got {self.spatial_step}")
        if self.time_step <= 0:
            raise ValueError(f"Time step must be positive, got {self.time_step}")


@dataclass
class SolverSpecification:
    """Solver specification.
    
    Attributes:
        method: Solver method ('explicit_fdm', 'crank_nicolson', 'fem')
    """
    method: str
    
    def validate(self):
        """Validate solver specification."""
        valid_methods = ['explicit_fdm', 'crank_nicolson', 'fem']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid solver method: {self.method}. Must be one of {valid_methods}")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.
    
    Attributes:
        experiment_id: Unique experiment identifier
        name: Experiment name
        drug: Drug specification
        tissue: Tissue specification
        simulation: Simulation parameters
        solver: Solver specification
        boundary_left: Left boundary condition
        boundary_right: Right boundary condition
        output_directory: Output directory path
    """
    experiment_id: str
    name: str
    drug: DrugSpecification
    tissue: TissueSpecification
    simulation: SimulationParameters
    solver: SolverSpecification
    boundary_left: BoundaryCondition
    boundary_right: BoundaryCondition
    output_directory: str = "results"
    
    def validate(self):
        """Validate complete configuration."""
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.name:
            raise ValueError("name is required")
        
        self.drug.validate()
        self.tissue.validate()
        self.simulation.validate()
        self.solver.validate()
        self.boundary_left.validate()
        self.boundary_right.validate()


def load_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig: Validated experiment configuration
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If YAML file not found
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError("Empty YAML file")
    
    # Parse experiment metadata
    exp_data = data.get('experiment', {})
    exp_id = exp_data.get('id', 'UNKNOWN')
    exp_name = exp_data.get('name', 'Unnamed')
    
    # Parse drug specification
    drug_data = data.get('drug', {})
    drug = DrugSpecification(
        initial_concentration=drug_data.get('initial_concentration', 1.0),
        unit=drug_data.get('unit', 'mol/m3')
    )
    
    # Parse tissue specification
    tissue_data = data.get('tissue', {})
    layers_data = tissue_data.get('layers', [])
    if not layers_data:
        raise ValueError("At least one tissue layer is required in configuration")
    
    layers = []
    for layer_data in layers_data:
        layer = TissueLayer(
            name=layer_data.get('name', 'unnamed'),
            thickness=layer_data.get('thickness'),
            diffusion_coefficient=layer_data.get('diffusion_coefficient'),
            clearance=layer_data.get('clearance', 0.0)
        )
        layers.append(layer)
    
    tissue = TissueSpecification(layers=layers)
    
    # Parse simulation parameters
    sim_data = data.get('simulation', {})
    simulation = SimulationParameters(
        final_time=sim_data.get('final_time'),
        spatial_step=sim_data.get('spatial_step'),
        time_step=sim_data.get('time_step')
    )
    
    # Parse solver specification
    solver_data = data.get('solver', {})
    solver = SolverSpecification(
        method=solver_data.get('method', 'explicit_fdm')
    )
    
    # Parse boundary conditions
    boundary_data = data.get('boundary', {})
    left_bc_data = boundary_data.get('left', {})
    right_bc_data = boundary_data.get('right', {})
    
    boundary_left = BoundaryCondition(
        type=left_bc_data.get('type', 'dirichlet'),
        value=left_bc_data.get('value', 0.0)
    )
    
    boundary_right = BoundaryCondition(
        type=right_bc_data.get('type', 'dirichlet'),
        value=right_bc_data.get('value', 0.0)
    )
    
    output_dir = data.get('output', {}).get('directory', 'results')
    
    # Create and validate configuration
    config = ExperimentConfig(
        experiment_id=exp_id,
        name=exp_name,
        drug=drug,
        tissue=tissue,
        simulation=simulation,
        solver=solver,
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        output_directory=output_dir
    )
    
    config.validate()
    return config
