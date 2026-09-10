"""
Sensitivity analysis orchestrator for CDTS.

Combines sweep execution with sensitivity metric computation:
1. Run parametric sweep
2. Compute Sobol indices
3. Compute normalized/local sensitivities
4. Generate tornado data
5. Save full analysis report
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .sweep_runner import ParametricSweepRunner, SweepConfig
from .results import SweepResults
from .metrics import (
    normalized_sensitivity,
    local_sensitivity,
    sobol_indices_estimate,
    sensitivity_tornado_data,
)

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """End-to-end sensitivity analysis for CDTS experiments."""

    def __init__(
        self,
        base_config_path: str,
        sweep_parameters: Dict[str, List[float]],
        output_directory: str = "results",
        strategy: str = "full_factorial",
        n_samples: int = 50,
        solver_method: Optional[str] = None,
    ):
        """Initialize sensitivity analyzer.

        Args:
            base_config_path: Path to base YAML configuration
            sweep_parameters: Dict of parameter_name -> list of values
            output_directory: Results output directory
            strategy: Sweep strategy ('full_factorial', 'lhs', 'random', 'grid')
            n_samples: Number of samples for lhs/random
            solver_method: Optional solver override
        """
        self.base_config_path = base_config_path
        self.sweep_parameters = sweep_parameters
        self.output_directory = output_directory
        self.strategy = strategy
        self.n_samples = n_samples
        self.solver_method = solver_method

        self.sweep_results: Optional[SweepResults] = None
        self.sensitivity_metrics: Dict = {}

    def run(self) -> SweepResults:
        """Execute full sensitivity analysis pipeline."""
        logger.info("=" * 70)
        logger.info("M7 SENSITIVITY ANALYSIS")
        logger.info("=" * 70)

        # Step 1: Build sweep config
        exp_id = Path(self.base_config_path).stem + "_sensitivity"
        sweep_config = SweepConfig(
            experiment_id=exp_id,
            base_config_path=self.base_config_path,
            parameters=self.sweep_parameters,
            strategy=self.strategy,
            n_samples=self.n_samples,
            output_directory=self.output_directory,
            solver_method=self.solver_method,
        )

        # Step 2: Run sweep
        logger.info(f"Running parametric sweep: {self.strategy}")
        runner = ParametricSweepRunner(sweep_config)
        self.sweep_results = runner.run()

        # Step 3: Compute sensitivity metrics
        self._compute_metrics()

        # Step 4: Save analysis report
        self._save_analysis()

        logger.info("=" * 70)
        logger.info("SENSITIVITY ANALYSIS COMPLETE")
        logger.info(f"Successful runs: {self.sweep_results.success_count()}")
        logger.info(f"Output: {self.output_directory}")
        logger.info("=" * 70)

        return self.sweep_results

    def _compute_metrics(self):
        """Compute all sensitivity metrics from sweep results."""
        if self.sweep_results is None:
            return

        successful = [
            r for r in self.sweep_results.results
            if r.get("status") == "success"
        ]
        if len(successful) < 3:
            logger.warning("Not enough successful runs for sensitivity analysis")
            self.sensitivity_metrics = {"warning": "insufficient_data"}
            return

        # Extract parameter matrix and output vectors
        param_names = sorted(self.sweep_parameters.keys())
        param_matrix = np.array([[r.get(p, 0.0) for p in param_names] for r in successful])
        output_peak = np.array([r.get("peak_concentration", 0.0) for r in successful])
        output_mass = np.array([r.get("mass_remaining_fraction", 0.0) for r in successful])

        # Normalized sensitivity
        normalized = {}
        for i, pname in enumerate(param_names):
            pvals = param_matrix[:, i]
            pbaseline = np.mean(pvals)
            if output_peak.max() > 0:
                normalized[pname] = normalized_sensitivity(output_peak, pvals, pbaseline)

        # Sobol indices
        sobol_peak_first, sobol_peak_total = sobol_indices_estimate(param_matrix, output_peak)
        sobol_mass_first, sobol_mass_total = sobol_indices_estimate(param_matrix, output_mass)

        sobol_peak = {}
        sobol_mass = {}
        for i, pname in enumerate(param_names):
            sobol_peak[pname] = {
                "first_order": float(sobol_peak_first[i]),
                "total_order": float(sobol_peak_total[i]),
            }
            sobol_mass[pname] = {
                "first_order": float(sobol_mass_first[i]),
                "total_order": float(sobol_mass_total[i]),
            }

        # Tornado data
        tornado_names, tornado_values = sensitivity_tornado_data(normalized, top_n=10)

        self.sensitivity_metrics = {
            "normalized_sensitivity": normalized,
            "sobol_peak_concentration": sobol_peak,
            "sobol_mass_remaining": sobol_mass,
            "tornado_peak": {"names": tornado_names, "values": tornado_values},
            "summary": self.sweep_results.summary(),
        }

    def _save_analysis(self):
        """Save sensitivity analysis report."""
        if self.sweep_results is None:
            return

        output_dir = Path(self.output_directory)
        report_path = output_dir / f"{self.sweep_results.experiment_id}_sensitivity.json"
        with open(report_path, "w") as f:
            json.dump(self.sensitivity_metrics, f, indent=2, default=str)
        logger.info(f"Sensitivity report saved: {report_path}")

    def get_sweep_results(self) -> Optional[SweepResults]:
        return self.sweep_results

    def get_metrics(self) -> Dict:
        return self.sensitivity_metrics
