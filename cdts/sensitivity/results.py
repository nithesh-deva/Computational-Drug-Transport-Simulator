"""
Sweep result aggregation and summary statistics.
"""

import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path


class SweepResults:
    """Aggregated results from a parametric sweep."""

    def __init__(
        self,
        experiment_id: str,
        results: List[Dict],
        output_directory: str,
    ):
        self.experiment_id = experiment_id
        self.results = results
        self.output_directory = output_directory

    def success_count(self) -> int:
        return sum(1 for r in self.results if r.get("status") == "success")

    def failure_count(self) -> int:
        return sum(1 for r in self.results if r.get("status") == "failed")

    def summary(self) -> Dict:
        """Compute summary statistics over successful results."""
        successful = [r for r in self.results if r.get("status") == "success"]
        if not successful:
            return {"error": "No successful results"}

        numeric_keys = [
            "peak_concentration",
            "final_mass",
            "mass_remaining_fraction",
            "runtime_seconds",
            "D",
            "k",
            "L",
        ]

        summary = {
            "experiment_id": self.experiment_id,
            "total_runs": len(self.results),
            "successful": self.success_count(),
            "failed": self.failure_count(),
        }

        for key in numeric_keys:
            values = [r[key] for r in successful if key in r and isinstance(r[key], (int, float))]
            if values:
                arr = np.array(values)
                summary[f"{key}_mean"] = float(np.mean(arr))
                summary[f"{key}_std"] = float(np.std(arr))
                summary[f"{key}_min"] = float(np.min(arr))
                summary[f"{key}_max"] = float(np.max(arr))
                summary[f"{key}_median"] = float(np.median(arr))

        return summary

    def save_summary(self, path: Optional[str] = None) -> str:
        """Save summary JSON to file."""
        if path is None:
            path = str(Path(self.output_directory) / f"{self.experiment_id}_summary.json")
        summary = self.summary()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def parameter_matrix(self) -> Optional[np.ndarray]:
        """Extract parameter values as matrix for Sobol analysis."""
        successful = [r for r in self.results if r.get("status") == "success"]
        if not successful:
            return None

        param_names = sorted([
            k for k in successful[0].keys()
            if k not in {
                "status", "error", "sweep_index", "method",
                "peak_concentration", "final_mass", "initial_mass",
                "mass_remaining_fraction", "runtime_seconds",
                "nx", "nt",
            }
        ])

        if not param_names:
            return None

        matrix = np.array([[r.get(p, 0.0) for p in param_names] for r in successful])
        return matrix

    def output_vector(self, key: str = "peak_concentration") -> Optional[np.ndarray]:
        """Extract output metric as vector."""
        successful = [r for r in self.results if r.get("status") == "success"]
        if not successful:
            return None
        values = [r.get(key, 0.0) for r in successful]
        return np.array(values)
