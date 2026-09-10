"""
Finite Element Method (FEM) solver for 1D reaction-diffusion using FEniCSx (M5).

Weak Formulation:
    Find C ∈ V such that for all v ∈ V:
    
    ∫ (∂C/∂t)·v dx + ∫ D·(∇C)·(∇v) dx + ∫ k·C·v dx = 0
    
    with C(0,t) = C_left, C(L,t) = C_right (Dirichlet BC)
    
Temporal Discretization (Backward Euler for stability):
    Find C^(n+1) ∈ V such that for all v ∈ V:
    
    ∫ ((C^(n+1) - C^n)/Δt)·v dx + ∫ D·(∇C^(n+1))·(∇v) dx + ∫ k·C^(n+1)·v dx = 0

This reformulation leads to a symmetric linear system:
    M·C^(n+1) + Δt·(K + C)·C^(n+1) = M·C^n
    
where:
    M = mass matrix
    K = stiffness (diffusion) matrix
    C = reaction matrix (clearance term)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FEMBasis(ABC):
    """Abstract base class for finite element basis functions."""
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, derivatives: int = 0) -> np.ndarray:
        """Evaluate basis function at points x.
        
        Args:
            x: Evaluation points
            derivatives: Number of derivatives to compute (0=value, 1=gradient, etc.)
            
        Returns:
            Basis function values or derivatives
        """
        pass
    
    @abstractmethod
    def support(self) -> Tuple[float, float]:
        """Return support interval of basis function."""
        pass


class LinearLagrangeBasis1D(FEMBasis):
    """Linear Lagrange basis functions on 1D interval.
    
    For element [x_i, x_{i+1}], basis functions are:
        φ_i(x) = (x_{i+1} - x) / (x_{i+1} - x_i)  (left node)
        φ_{i+1}(x) = (x - x_i) / (x_{i+1} - x_i)  (right node)
    """
    
    def __init__(self, node_index: int, nodes: np.ndarray):
        """Initialize linear Lagrange basis function.
        
        Args:
            node_index: Index of node this basis is associated with
            nodes: Array of all node positions
        """
        self.node_index = node_index
        self.nodes = nodes
        
        if node_index == 0:
            self.x_left = nodes[0]
            self.x_right = nodes[1]
        elif node_index == len(nodes) - 1:
            self.x_left = nodes[-2]
            self.x_right = nodes[-1]
        else:
            self.x_left = nodes[node_index - 1]
            self.x_right = nodes[node_index + 1]
    
    def evaluate(self, x: np.ndarray, derivatives: int = 0) -> np.ndarray:
        """Evaluate basis function or its derivatives.
        
        Args:
            x: Evaluation points
            derivatives: 0 for value, 1 for derivative
            
        Returns:
            Function values or derivatives
        """
        x = np.asarray(x)
        
        if derivatives == 0:
            # Linear basis function value
            result = np.zeros_like(x, dtype=float)
            
            # Handle each element separately
            for i in range(len(self.nodes) - 1):
                x_i, x_ip1 = self.nodes[i], self.nodes[i + 1]
                h = x_ip1 - x_i
                
                mask = (x >= x_i) & (x <= x_ip1)
                
                if self.node_index == i:
                    # Left node basis
                    result[mask] = (x_ip1 - x[mask]) / h
                elif self.node_index == i + 1:
                    # Right node basis
                    result[mask] = (x[mask] - x_i) / h
            
            return result
        
        elif derivatives == 1:
            # Derivative of linear basis
            result = np.zeros_like(x, dtype=float)
            
            for i in range(len(self.nodes) - 1):
                x_i, x_ip1 = self.nodes[i], self.nodes[i + 1]
                h = x_ip1 - x_i
                
                mask = (x >= x_i) & (x < x_ip1)
                
                if self.node_index == i:
                    result[mask] = -1.0 / h
                elif self.node_index == i + 1:
                    result[mask] = 1.0 / h
            
            return result
        
        else:
            raise ValueError(f"Derivative order {derivatives} not supported")
    
    def support(self) -> Tuple[float, float]:
        """Return support of basis function."""
        return (self.x_left, self.x_right)


class FEMAssembler1D:
    """Assemble FEM matrices for 1D problems.
    
    Assembles:
    - Mass matrix M from ∫ φ_i·φ_j dx
    - Stiffness matrix K from ∫ (∇φ_i)·(∇φ_j) dx  
    - Reaction matrix R from ∫ φ_i·φ_j dx
    """
    
    def __init__(self, nodes: np.ndarray):
        """Initialize assembler with node positions.
        
        Args:
            nodes: Array of node positions [x_0, x_1, ..., x_n]
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        
    def assemble_mass_matrix(self) -> np.ndarray:
        """Assemble mass matrix M from ∫ φ_i·φ_j dx.
        
        For linear elements on [x_i, x_{i+1}] with h = x_{i+1} - x_i:
            M_local = (h/6) * [2 1; 1 2]
        
        Returns:
            Global mass matrix (n_nodes × n_nodes)
        """
        M = np.zeros((self.n_nodes, self.n_nodes))
        
        for elem in range(len(self.nodes) - 1):
            x_i, x_ip1 = self.nodes[elem], self.nodes[elem + 1]
            h = x_ip1 - x_i
            
            # Local mass matrix for linear element
            M_local = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
            
            # Add to global matrix
            i, j = elem, elem + 1
            M[i, i] += M_local[0, 0]
            M[i, j] += M_local[0, 1]
            M[j, i] += M_local[1, 0]
            M[j, j] += M_local[1, 1]
        
        return M
    
    def assemble_stiffness_matrix(self, diffusion_coeff: float) -> np.ndarray:
        """Assemble stiffness matrix K from ∫ D·(∇φ_i)·(∇φ_j) dx.
        
        For linear elements:
            K_local = (D/h) * [1 -1; -1 1]
        
        Args:
            diffusion_coeff: D value
            
        Returns:
            Global stiffness matrix
        """
        K = np.zeros((self.n_nodes, self.n_nodes))
        
        for elem in range(len(self.nodes) - 1):
            x_i, x_ip1 = self.nodes[elem], self.nodes[elem + 1]
            h = x_ip1 - x_i
            
            # Local stiffness matrix for linear element
            K_local = (diffusion_coeff / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            
            # Add to global matrix
            i, j = elem, elem + 1
            K[i, i] += K_local[0, 0]
            K[i, j] += K_local[0, 1]
            K[j, i] += K_local[1, 0]
            K[j, j] += K_local[1, 1]
        
        return K
    
    def assemble_reaction_matrix(self, clearance_coeff: float) -> np.ndarray:
        """Assemble reaction matrix R from ∫ k·φ_i·φ_j dx.
        
        For linear elements:
            R_local = (k·h/6) * [2 1; 1 2]
        
        Args:
            clearance_coeff: k value
            
        Returns:
            Global reaction matrix
        """
        R = np.zeros((self.n_nodes, self.n_nodes))
        
        for elem in range(len(self.nodes) - 1):
            x_i, x_ip1 = self.nodes[elem], self.nodes[elem + 1]
            h = x_ip1 - x_i
            
            # Local reaction matrix
            R_local = (clearance_coeff * h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
            
            # Add to global matrix
            i, j = elem, elem + 1
            R[i, i] += R_local[0, 0]
            R[i, j] += R_local[0, 1]
            R[j, i] += R_local[1, 0]
            R[j, j] += R_local[1, 1]
        
        return R
    
    def assemble_load_vector(self, source_func, nodes: Optional[np.ndarray] = None) -> np.ndarray:
        """Assemble load vector from ∫ f·φ_i dx.
        
        Args:
            source_func: Function f(x) for load
            nodes: Override nodes for integration
            
        Returns:
            Global load vector
        """
        if nodes is None:
            nodes = self.nodes
        
        f = np.zeros(self.n_nodes)
        
        for elem in range(len(nodes) - 1):
            x_i, x_ip1 = nodes[elem], nodes[elem + 1]
            h = x_ip1 - x_i
            
            # Midpoint quadrature for linear basis
            x_mid = (x_i + x_ip1) / 2.0
            f_mid = source_func(x_mid)
            
            # Local load vector contributions
            i, j = elem, elem + 1
            f[i] += f_mid * h / 2.0
            f[j] += f_mid * h / 2.0
        
        return f


class FEMSolver1D:
    """1D Finite Element solver for reaction-diffusion with backward Euler."""
    
    def __init__(
        self,
        domain_length: float,
        n_elements: int,
        diffusion_coeff: float,
        dt: float,
        boundary_left: float,
        boundary_right: float,
        clearance: float = 0.0
    ):
        """Initialize FEM solver.
        
        Args:
            domain_length: Domain [0, L]
            n_elements: Number of elements
            diffusion_coeff: D [m²/s]
            dt: Time step [s]
            boundary_left: C(0,t) [mol/m³]
            boundary_right: C(L,t) [mol/m³]
            clearance: k [1/s]
        """
        # Create uniform mesh
        self.x = np.linspace(0, domain_length, n_elements + 1)
        self.n_nodes = len(self.x)
        
        self.domain_length = domain_length
        self.diffusion_coeff = diffusion_coeff
        self.dt = dt
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.clearance = clearance
        
        # Assemble matrices
        assembler = FEMAssembler1D(self.x)
        self.M = assembler.assemble_mass_matrix()
        self.K = assembler.assemble_stiffness_matrix(diffusion_coeff)
        self.R = assembler.assemble_reaction_matrix(clearance)
        
        # System matrix: M + Δt·(K + R)
        self.A = self.M + dt * (self.K + self.R)
        
        logger.info(
            f"FEM solver initialized:\n"
            f"  Elements: {n_elements}\n"
            f"  Nodes: {self.n_nodes}\n"
            f"  Time step: {dt:.6e} s"
        )
    
    def solve(
        self,
        initial_condition: np.ndarray,
        final_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve reaction-diffusion system with backward Euler.
        
        Args:
            initial_condition: C(x,0) at nodes
            final_time: T
            
        Returns:
            Tuple: (x, t, C, metrics)
        """
        nt = int(final_time / self.dt) + 1
        t = np.linspace(0, final_time, nt)
        
        C = np.zeros((self.n_nodes, nt))
        C[:, 0] = initial_condition
        
        # Time stepping with backward Euler
        for n in range(nt - 1):
            C_n = C[:, n]
            
            # RHS: M·C^n
            rhs = self.M @ C_n
            
            # Enforce Dirichlet boundary conditions
            A_bc = self.A.copy()
            rhs_bc = rhs.copy()
            
            # Zero row for left BC
            A_bc[0, :] = 0.0
            A_bc[0, 0] = 1.0
            rhs_bc[0] = self.boundary_left
            
            # Zero row for right BC
            A_bc[-1, :] = 0.0
            A_bc[-1, -1] = 1.0
            rhs_bc[-1] = self.boundary_right
            
            # Solve linear system
            try:
                C_np1 = np.linalg.solve(A_bc, rhs_bc)
            except np.linalg.LinAlgError:
                logger.error(f"Singular matrix at time step {n}")
                raise
            
            # Enforce non-negativity
            C_np1 = np.maximum(C_np1, 0.0)
            C[:, n+1] = C_np1
        
        metrics = {
            'method': 'fem_linear_lagrange',
            'elements': self.n_nodes - 1,
            'nodes': self.n_nodes,
            'dt': self.dt,
            'diffusion_coeff': self.diffusion_coeff,
            'clearance': self.clearance,
            'domain_length': self.domain_length,
            'final_time': final_time,
            'basis': 'linear_lagrange_1d'
        }
        
        return self.x, t, C, metrics
