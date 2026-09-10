"""
Parametric sweep runner for CDTS experiments.

Executes parameter sweeps by:
1. Parsing sweep specification from YAML
2. Generating parameter combinations
3. Running simulations for each combination
4. Collecting metrics
5. Saving aggregated results
"""

import numpy as np
import json
import csv
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import sys

from .results import SweepResults

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for a parametric sweep.

    Attributes:
        experiment_id: Unique sweep identifier
        base_config_path: Path to base YAML configuration
        parameters: Dict of parameter_name -> list of values to sweep
        strategy: 'full_factorial', 'lhs', 'random', 'grid'
        n_samples: Number of samples for lhs/random strategies
        output_directory: Where to save results
        solver_method: Override solver method (optional)
    """
    experiment_id: str
    base_config_path: str
    parameters: Dict[str, List[float]]
    strategy: str = "full_factorial"
    n_samples: int = 50
    output_directory: str = "results"
    solver_method: Optional[str] = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.base_config_path:
            raise ValueError("base_config_path is required")
        if not self.parameters:
            raise ValueError("At least one parameter must be specified")
        if self.strategy not in ["full_factorial", "lhs", "random", "grid"]:
            raise ValueError(f"Unknown sweep strategy: {self.strategy}")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")


class ParametricSweepRunner:
    """Execute parametric sweeps over CDTS experiments."""

    def __init__(
        self,
        config: SweepConfig,
        run_simulation_fn: Optional[Callable] = None,
    ):
        """Initialize sweep runner.

        Args:
            config: Sweep configuration
            run_simulation_fn: Callable(params_dict) -> results_dict.
                               If None, uses default CDTS runner.
        """
        self.config = config
        config.validate()
        self.run_simulation_fn = run_simulation_fn or self._default_run_simulation
        self.results: List[Dict[str, Any]] = []

    def _default_run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default simulation runner using CDTS solvers."""
        from cdts.config import load_config_from_yaml
        from cdts.solvers.explicit_fdm import ExplicitFDMSolver
        from cdts.solvers.crank_nicolson import CrankNicolsonSolver
        from cdts.solvers.fem_1d import FEMSolver1D
        from cdts.validation.metrics import mass_conservation_check, rmse

        # Load base config
        base = load_config_from_yaml(self.config.base_config_path)

        # Override parameters
        layer = base.tissue.layers[0]
        if "diffusion_coefficient" in params:
            layer.diffusion_coefficient = float(params["diffusion_coefficient"])
        if "clearance" in params:
            layer.clearance = float(params["clearance"])
        if "thickness" in params:
            layer.thickness = float(params["thickness"])
        if "initial_concentration" in params:
            base.drug.initial_concentration = float(params["initial_concentration"])
        if "spatial_step" in params:
            base.simulation.spatial_step = float(params["spatial_step"])
        if "time_step" in params:
            base.simulation.time_step = float(params["time_step"])
        if "final_time" in params:
            base.simulation.final_time = float(params["final_time"])
        if "partition_coefficient" in params:
            params["partition_coefficient"] = float(params["partition_coefficient"])

        D = layer.diffusion_coefficient
        k = layer.clearance
        L = layer.thickness
        dx = base.simulation.spatial_step
        dt = base.simulation.time_step
        T = base.simulation.final_time
        C_init_val = base.drug.initial_concentration
        C_left = base.boundary_left.value
        C_right = base.boundary_right.value

        method = self.config.solver_method or base.solver.method

        # Run solver
        t0 = time.time()
        if method == "explicit_fdm":
            solver = ExplicitFDMSolver(
                domain_length=L,
                diffusion_coeff=D,
                dx=dx,
                dt=dt,
                boundary_left=C_left,
                boundary_right=C_right,
                clearance=k,
            )
            C_init = np.full(solver.nx, C_init_val)
            x, t, C, metrics = solver.solve(C_init, T)
        elif method == "crank_nicolson":
            solver = CrankNicolsonSolver(
                domain_length=L,
                diffusion_coeff=D,
                dx=dx,
                dt=dt,
                boundary_left=C_left,
                boundary_right=C_right,
                clearance=k,
            )
            C_init = np.full(solver.nx, C_init_val)
            x, t, C, metrics = solver.solve(C_init, T)
        elif method == "fem":
            n_elem = max(5, int(L / dx))
            solver = FEMSolver1D(
                domain_length=L,
                n_elements=n_elem,
                diffusion_coeff=D,
                dt=dt,
                boundary_left=C_left,
                boundary_right=C_right,
                clearance=k,
            )
            C_init = np.full(solver.n_nodes, C_init_val)
            x, t, C, metrics = solver.solve(C_init, T)
        else:
            raise ValueError(f"Unknown solver method: {method}")

        runtime = time.time() - t0

        # Compute metrics
        C_final = C[:, -1]
        peak_conc = float(np.max(C_final))
        final_mass = float(np.trapz(C_final, x))
        initial_mass = float(np.trapz(C_init[: len(x)], x)) if C_init.shape[0] >= len(x) else float(np.trapz(C_init, x))
        mass_remaining = final_mass / (initial_mass + 1e-14) if initial_mass > 0 else 0.0

        results = {
            "peak_concentration": peak_conc,
            "final_mass": final_mass,
            "initial_mass": initial_mass,
            "mass_remaining_fraction": mass_remaining,
            "runtime_seconds": runtime,
            "method": method,
            "D": D,
            "k": k,
            "L": L,
            "dx": dx,
            "dt": dt,
            "T": T,
            "C_init": C_init_val,
            "nx": C.shape[0],
            "nt": C.shape[1],
        }

        # Add swept params
        for key, val in params.items():
            results[key] = float(val)

        return results

    def generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """Generate parameter combinations based on strategy."""
        strategy = self.config.strategy
        param_dict = self.config.parameters
        param_names = list(param_dict.keys())
        param_values = list(param_dict.values())

        if strategy == "full_factorial":
            return self._full_factorial(param_names, param_values)
        elif strategy == "grid":
            return self._full_factorial(param_names, param_values)
        elif strategy == "lhs":
            return self._lhs_sampling(param_names, param_values, self.config.n_samples)
        elif strategy == "random":
            return self._random_sampling(param_names, param_values, self.config.n_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _full_factorial(self, names, values) -> List[Dict]:
        from itertools import product
        combos = []
        for combo in product(*values):
            combos.append(dict(zip(names, combo)))
        return combos

    def _lhs_sampling(self, names, values, n_samples) -> List[Dict]:
        rng = np.random.default_rng(42)
        n_params = len(names)
        combos = []
        for _ in range(n_samples):
            combo = {}
            for i, name in enumerate(names):
                vmin, vmax = min(values[i]), max(values[i])
                u = rng.uniform(vmin, vmax)
                combo[name] = float(u)
            combos.append(combo)
        return combos

    def _random_sampling(self, names, values, n_samples) -> List[Dict]:
        rng = np.random.default_rng(123)
        combos = []
        for _ in range(n_samples):
            combo = {}
            for i, name in enumerate(names):
                vmin, vmax = min(values[i]), max(values[i])
                u = rng.uniform(vmin, vmax)
                combo[name] = float(u)
            combos.append(combo)
        return combos

    def run(self) -> "SweepResults":
        """Execute the full parametric sweep.

        Returns:
            SweepResults with all simulation results
        """
        combos = self.generate_parameter_combinations()
        logger.info(
            f"Starting sweep: {len(combos)} parameter combinations "
            f"(strategy={self.config.strategy})"
        )

        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        for idx, params in enumerate(combos):
            try:
                result = self.run_simulation_fn(params)
                result["sweep_index"] = idx
                result["status"] = "success"
                self.results.append(result)
                logger.info(
                    f"  [{idx+1}/{len(combos)}] params={params} -> "
                    f"peak={result.get('peak_concentration', 0):.4e}, "
                    f"mass={result.get('mass_remaining_fraction', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"  [{idx+1}/{len(combos)}] FAILED: {e}")
                self.results.append({
                    "sweep_index": idx,
                    "status": "failed",
                    "error": str(e),
                    "params": params,
                })

        # Save results
        self._save_results(output_dir)

        return SweepResults(
            experiment_id=self.config.experiment_id,
            results=self.results,
            output_directory=str(output_dir),
        )

    def _save_results(self, output_dir: Path):
        """Save sweep results to CSV and JSON."""
        # JSON
        json_path = output_dir / f"{self.config.experiment_id}_sweep.json"
        with open(json_path, "w") as f:
            json.dump({
                "experiment_id": self.config.experiment_id,
                "strategy": self.config.strategy,
                "n_samples": self.config.n_samples,
                "results": self.results,
            }, f, indent=2, default=str)
        logger.info(f"Sweep JSON saved: {json_path}")

        # CSV
        csv_path = output_dir / f"{self.config.experiment_id}_sweep.csv"
        if self.results:
            fieldnames = list(self.results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in self.results:
                    writer.writerow(row)
        logger.info(f"Sweep CSV saved: {csv_path}")
