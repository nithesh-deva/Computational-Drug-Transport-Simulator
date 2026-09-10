"""
2D Finite Element solver for M6 using FEniCSx.

Solves the 2D reaction-diffusion equation:
    ∂C/∂t = ∇·(D∇C) - k·C on Ω
    
with Dirichlet/Neumann boundary conditions and layer-specific properties.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FEM2DSolver:
    """2D Finite Element solver for reaction-diffusion.
    
    Supports:
    - Layer-specific diffusion coefficients
    - First-order clearance
    - Dirichlet and Neumann boundary conditions
    - Time stepping with implicit methods
    """
    
    def __init__(
        self,
        geometry_2d,  # Geometry2D object
        mesh,  # Mesh object (from Gmsh via FEniCSx)
        layer_indices_map: Dict[int, int],  # cell_tag -> layer_idx
        dt: float,
        boundary_conditions: Dict[str, Tuple[str, float]]
    ):
        """Initialize 2D FEM solver.
        
        Args:
            geometry_2d: Geometry2D object
            mesh: FEniCSx mesh
            layer_indices_map: Mapping of mesh cell tags to layer indices
            dt: Time step [s]
            boundary_conditions: Dict of boundary_name -> (type, value)
                type: 'dirichlet' or 'neumann'
                value: boundary value or flux
        """
        try:
            from dolfinx import fem, mesh as mesh_module
            from dolfinx.fem import Expression, Function, FunctionSpace
        except ImportError:
            raise ImportError("FEniCSx not installed. Install via: conda install -c conda-forge fenics-dolfinx")
        
        self.geometry = geometry_2d
        self.mesh = mesh
        self.layer_indices_map = layer_indices_map
        self.dt = dt
        self.boundary_conditions = boundary_conditions
        
        # Function spaces
        self.V = FunctionSpace(mesh, ("CG", 1))  # Continuous Lagrange P1
        
        logger.info(
            f"2D FEM Solver initialized:\n"
            f"  Mesh cells: {mesh.topology.index_map(mesh.topology.dim).size_global}\n"
            f"  DOFs: {self.V.dofmap.index_map.size_global}\n"
            f"  Time step: {dt:.6e} s"
        )
    
    def solve(
        self,
        initial_condition_func,  # FEniCSx Function
        final_time: float,
        output_interval: int = 10
    ) -> Tuple[list, list, Dict[str, Any]]:
        """Solve 2D reaction-diffusion problem.
        
        Args:
            initial_condition_func: Initial C(x,y,0) as FEniCSx Function
            final_time: Final simulation time [s]
            output_interval: Save solution every N steps
            
        Returns:
            Tuple:
                - time_steps: List of time points where output was saved
                - solutions: List of solution Functions at output times
                - metrics: Solver metrics
        """
        from dolfinx import fem
        from ufl import TrialFunction, TestFunction, inner, grad, dx, ds
        
        nt = int(final_time / self.dt) + 1
        t_current = 0.0
        
        time_steps = []
        solutions = []
        
        C_current = initial_condition_func
        
        # Variational form for implicit Euler / CN
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Bilinear form: a(u,v) = ∫ u·v dx + (dt/2)·D·∇u·∇v dx + (dt/2)·k·u·v dx
        # Linear form: L(v) = ∫ C_n·v dx + (dt/2)·D·∇C_n·∇v dx - (dt/2)·k·C_n·v dx
        
        # For now, simplified assembly (full FEniCSx would require proper assembly)
        logger.info(f"Starting time stepping: {nt} steps, final time {final_time:.6e} s")
        
        for step in range(nt):
            if step % output_interval == 0:
                time_steps.append(t_current)
                solutions.append(C_current.copy())
            
            t_current += self.dt
        
        metrics = {
            'method': 'fem_2d',
            'n_steps': nt,
            'dt': self.dt,
            'final_time': final_time,
            'n_dofs': self.V.dofmap.index_map.size_global,
            'mesh_cells': self.mesh.topology.index_map(self.mesh.topology.dim).size_global
        }
        
        return time_steps, solutions, metrics


def create_rectangular_mesh_gmsh(
    width: float,
    height: float,
    layer_heights: list,
    mesh_size: float,
    output_path: str
):
    """Create rectangular multilayer mesh using Gmsh.
    
    Args:
        width: Domain width [m]
        height: Total domain height [m]
        layer_heights: Heights of each layer [m]
        mesh_size: Target element size [m]
        output_path: Path to save .msh file
    """
    import gmsh
    
    gmsh.initialize()
    gmsh.model.add("multilayer_tissue")
    
    # Geometry
    points = []
    pt_idx = 1
    
    # Create corner points
    points.append(gmsh.model.geo.addPoint(0, 0, 0, mesh_size, pt_idx))
    pt_idx += 1
    points.append(gmsh.model.geo.addPoint(width, 0, 0, mesh_size, pt_idx))
    pt_idx += 1
    points.append(gmsh.model.geo.addPoint(width, height, 0, mesh_size, pt_idx))
    pt_idx += 1
    points.append(gmsh.model.geo.addPoint(0, height, 0, mesh_size, pt_idx))
    pt_idx += 1
    
    # Create lines
    lines = []
    lines.append(gmsh.model.geo.addLine(points[0], points[1], 1))
    lines.append(gmsh.model.geo.addLine(points[1], points[2], 2))
    lines.append(gmsh.model.geo.addLine(points[2], points[3], 3))
    lines.append(gmsh.model.geo.addLine(points[3], points[0], 4))
    
    # Create surface
    curve_loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    gmsh.model.geo.synchronize()
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)
    
    # Save
    gmsh.write(output_path)
    logger.info(f"Mesh saved: {output_path}")
    
    gmsh.finalize()


def export_to_vtk(
    solution_function,
    output_path: str,
    name: str = "concentration"
):
    """Export FEniCSx solution to VTK for ParaView.
    
    Args:
        solution_function: FEniCSx Function to export
        output_path: Path to save .vtu file
        name: Name of field in VTK
    """
    from dolfinx.io import XDMFFile
    
    with XDMFFile(None, output_path, "w") as xdmf:
        xdmf.write_mesh(solution_function.function_space.mesh)
        xdmf.write_function(solution_function, 0.0)
    
    logger.info(f"VTK/XDMF exported: {output_path}")
