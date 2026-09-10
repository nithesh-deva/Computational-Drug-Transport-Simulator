"""
M10 Database & Analysis Suite

Unified data management and analysis framework for CDTS simulations.

Provides:
- Result aggregation from M1-M9 experiments
- Experiment tracking and metadata persistence
- Result comparison and statistical analysis
- Report generation (markdown, JSON, figures)
- Performance analysis and benchmarking aggregation
- Sensitivity analysis tracking
"""

import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for a single experiment run."""
    experiment_id: str
    experiment_name: str
    timestamp: str
    solver_method: str
    nx: int
    nt: int
    dx: float
    dt: float
    final_time: float
    domain_length: float
    diffusion_coeff: float
    clearance: float
    boundary_left: float
    boundary_right: float
    backend: str = "cpu"
    version: str = "1.0"
    python_version: str = ""
    platform: str = ""
    hash_input: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Result of a numerical analysis."""
    metric_name: str
    value: float
    units: str
    description: str
    timestamp: str


class ResultDatabase:
    """Unified result database for all CDTS experiments.
    
    Stores and queries simulation results in HDF5 format.
    """

    def __init__(self, db_path: str):
        """Initialize result database.
        
        Args:
            db_path: Path to HDF5 database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure database schema exists."""
        with h5py.File(self.db_path, "a") as db:
            # Root groups
            db.require_group("experiments")
            db.require_group("metadata")
            db.require_group("analysis")
            db.require_group("performance")
            db.require_group("sensitivity")

            # Root attributes
            if "version" not in db.attrs:
                db.attrs["version"] = "1.0"
            if "created" not in db.attrs:
                db.attrs["created"] = datetime.now().isoformat()
            if "updated" not in db.attrs:
                db.attrs["updated"] = datetime.now().isoformat()

    def store_experiment(
        self,
        metadata: ExperimentMetadata,
        concentration_field: np.ndarray,
        time_points: np.ndarray,
        spatial_points: np.ndarray,
        metrics: Dict[str, Any],
    ):
        """Store complete experiment result.
        
        Args:
            metadata: Experiment metadata
            concentration_field: [nx × nt] concentration array
            time_points: [nt] time array
            spatial_points: [nx] spatial array
            metrics: Dictionary of calculated metrics
        """
        with h5py.File(self.db_path, "a") as db:
            exp_id = metadata.experiment_id
            exp_group = db["experiments"].require_group(exp_id)

            # Store concentration field
            exp_group.create_dataset("concentration", data=concentration_field, compression="gzip")
            exp_group.create_dataset("time", data=time_points, compression="gzip")
            exp_group.create_dataset("spatial", data=spatial_points, compression="gzip")

            # Store metadata
            meta_group = exp_group.require_group("metadata")
            for key, value in metadata.to_dict().items():
                if isinstance(value, str):
                    meta_group.attrs[key] = value
                else:
                    meta_group.attrs[key] = value

            # Store metrics
            metrics_group = exp_group.require_group("metrics")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_group.attrs[key] = value
                elif isinstance(value, str):
                    metrics_group.attrs[key] = value

            # Update database timestamp
            db.attrs["updated"] = datetime.now().isoformat()
            logger.info(f"Stored experiment: {exp_id}")

    def retrieve_experiment(self, experiment_id: str) -> Tuple[ExperimentMetadata, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve stored experiment.
        
        Args:
            experiment_id: Experiment ID to retrieve
            
        Returns:
            Tuple of (metadata, concentration, time, spatial)
        """
        with h5py.File(self.db_path, "r") as db:
            exp_group = db["experiments"][experiment_id]

            # Load arrays
            C = exp_group["concentration"][:]
            t = exp_group["time"][:]
            x = exp_group["spatial"][:]

            # Load metadata
            meta_dict = dict(exp_group["metadata"].attrs)
            metadata = ExperimentMetadata(**meta_dict)

            return metadata, C, t, x

    def list_experiments(self) -> List[str]:
        """List all stored experiment IDs."""
        with h5py.File(self.db_path, "r") as db:
            return list(db["experiments"].keys())

    def store_comparison(self, comparison_name: str, results: Dict[str, Any]):
        """Store experiment comparison result.
        
        Args:
            comparison_name: Name of comparison
            results: Dictionary of comparison results
        """
        with h5py.File(self.db_path, "a") as db:
            cmp_group = db["analysis"].require_group(comparison_name)
            for key, value in results.items():
                if isinstance(value, (int, float, str)):
                    cmp_group.attrs[key] = value
                elif isinstance(value, np.ndarray):
                    cmp_group.create_dataset(key, data=value, compression="gzip")

    def store_benchmark_result(self, backend: str, benchmark_data: Dict[str, Any]):
        """Store performance benchmark result.
        
        Args:
            backend: Backend name (cuda, openmp, numpy, python)
            benchmark_data: Benchmark metrics
        """
        with h5py.File(self.db_path, "a") as db:
            bench_group = db["performance"].require_group(backend)
            for key, value in benchmark_data.items():
                if isinstance(value, (int, float)):
                    bench_group.attrs[key] = value

    def store_sensitivity_analysis(self, analysis_name: str, sensitivity_data: pd.DataFrame):
        """Store sensitivity analysis result.
        
        Args:
            analysis_name: Name of sensitivity analysis
            sensitivity_data: DataFrame with sensitivity metrics
        """
        with h5py.File(self.db_path, "a") as db:
            sens_group = db["sensitivity"].require_group(analysis_name)
            for col in sensitivity_data.columns:
                sens_group.create_dataset(col, data=sensitivity_data[col].values, compression="gzip")


class ExperimentComparison:
    """Compare multiple experiment results."""

    def __init__(self, db: ResultDatabase):
        """Initialize comparison framework.
        
        Args:
            db: ResultDatabase instance
        """
        self.db = db

    def compare_penetration_depth(
        self,
        exp_ids: List[str],
        threshold: float = 0.01,
    ) -> Dict[str, float]:
        """Compare penetration depth across experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            threshold: Concentration threshold for penetration (default 1% of peak)
            
        Returns:
            Dictionary mapping exp_id to penetration depth (m)
        """
        results = {}
        for exp_id in exp_ids:
            metadata, C, t, x = self.db.retrieve_experiment(exp_id)
            # Penetration: rightmost point where C > threshold
            C_final = C[:, -1]
            penetration_idx = np.where(C_final > threshold)[0]
            if len(penetration_idx) > 0:
                penetration = x[penetration_idx[-1]]
            else:
                penetration = 0.0
            results[exp_id] = penetration

        return results

    def compare_peak_concentration(self, exp_ids: List[str]) -> Dict[str, float]:
        """Compare peak concentrations.
        
        Args:
            exp_ids: List of experiment IDs
            
        Returns:
            Dictionary mapping exp_id to peak concentration
        """
        results = {}
        for exp_id in exp_ids:
            _, C, _, _ = self.db.retrieve_experiment(exp_id)
            results[exp_id] = float(np.max(C))
        return results

    def compare_total_mass(self, exp_ids: List[str]) -> Dict[str, Tuple[float, float]]:
        """Compare initial and final total mass.
        
        Args:
            exp_ids: List of experiment IDs
            
        Returns:
            Dictionary mapping exp_id to (initial_mass, final_mass)
        """
        results = {}
        for exp_id in exp_ids:
            metadata, C, t, x = self.db.retrieve_experiment(exp_id)
            dx = metadata.dx
            initial_mass = float(np.sum(C[:, 0]) * dx)
            final_mass = float(np.sum(C[:, -1]) * dx)
            results[exp_id] = (initial_mass, final_mass)
        return results

    def compare_arrival_time(
        self,
        exp_ids: List[str],
        threshold: float = 0.1,
    ) -> Dict[str, float]:
        """Compare arrival time at specific spatial point.
        
        Args:
            exp_ids: List of experiment IDs
            threshold: Concentration threshold for arrival
            
        Returns:
            Dictionary mapping exp_id to arrival time (s)
        """
        results = {}
        for exp_id in exp_ids:
            metadata, C, t, x = self.db.retrieve_experiment(exp_id)
            # Find time when any point reaches threshold
            arrival_times = []
            for i in range(C.shape[0]):
                arrival_idx = np.where(C[i, :] > threshold)[0]
                if len(arrival_idx) > 0:
                    arrival_times.append(t[arrival_idx[0]])

            if arrival_times:
                results[exp_id] = float(np.min(arrival_times))
            else:
                results[exp_id] = float(np.inf)

        return results


class PerformanceAnalyzer:
    """Analyze performance benchmarks across backends."""

    def __init__(self, db: ResultDatabase):
        """Initialize performance analyzer.
        
        Args:
            db: ResultDatabase instance
        """
        self.db = db

    def compute_scaling_efficiency(self, backend_times: Dict[str, float]) -> Dict[str, float]:
        """Compute scaling efficiency for multi-threaded backends.
        
        Args:
            backend_times: Dict mapping backend name to time
            
        Returns:
            Dictionary mapping backend to efficiency (0-1)
        """
        if "python_loop" not in backend_times:
            return {}

        baseline = backend_times["python_loop"]
        efficiency = {}

        for backend, time_val in backend_times.items():
            if backend != "python_loop" and time_val > 0:
                speedup = baseline / time_val
                # Estimate from backend name (rough heuristic)
                if "openmp" in backend:
                    n_threads = 4  # Default assumption
                    efficiency[backend] = speedup / n_threads
                elif "cuda" in backend:
                    efficiency[backend] = speedup / 100  # Rough GPU estimate
                else:
                    efficiency[backend] = speedup

        return efficiency

    def identify_crossover_point(
        self,
        problem_sizes: List[int],
        cpu_times: List[float],
        gpu_times: List[float],
    ) -> Optional[int]:
        """Identify problem size where GPU becomes faster than CPU.
        
        Args:
            problem_sizes: List of nx values
            cpu_times: CPU times for each size
            gpu_times: GPU times for each size
            
        Returns:
            Crossover problem size, or None if GPU always slower
        """
        for size, cpu_t, gpu_t in zip(problem_sizes, cpu_times, gpu_times):
            if gpu_t < cpu_t:
                return size
        return None


class ReportGenerator:
    """Generate analysis reports in multiple formats."""

    def __init__(self, db: ResultDatabase):
        """Initialize report generator.
        
        Args:
            db: ResultDatabase instance
        """
        self.db = db

    def generate_markdown_summary(
        self,
        exp_ids: List[str],
        output_path: str,
    ):
        """Generate markdown summary report.
        
        Args:
            exp_ids: List of experiment IDs to include
            output_path: Output file path
        """
        lines = []
        lines.append("# CDTS Experiment Summary Report\n")
        lines.append(f"Generated: {datetime.now().isoformat()}\n")
        lines.append(f"Total experiments: {len(exp_ids)}\n\n")

        lines.append("## Experiment List\n")
        lines.append("| Exp ID | Solver | Domain (m) | dx (m) | dt (s) | Final Time (s) |\n")
        lines.append("|--------|--------|-----------|--------|--------|----------------|\n")

        for exp_id in exp_ids:
            metadata, _, _, _ = self.db.retrieve_experiment(exp_id)
            lines.append(
                f"| {metadata.experiment_id} | {metadata.solver_method} | "
                f"{metadata.domain_length:.2e} | {metadata.dx:.2e} | "
                f"{metadata.dt:.2e} | {metadata.final_time:.2e} |\n"
            )

        lines.append("\n## Analysis Metrics\n")
        lines.append("Detailed metrics for each experiment stored in database.\n")

        with open(output_path, "w") as f:
            f.writelines(lines)

        logger.info(f"Generated markdown report: {output_path}")

    def generate_json_summary(
        self,
        exp_ids: List[str],
        output_path: str,
    ):
        """Generate JSON summary report.
        
        Args:
            exp_ids: List of experiment IDs
            output_path: Output file path
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(exp_ids),
            "experiments": [],
        }

        for exp_id in exp_ids:
            metadata, C, t, x = self.db.retrieve_experiment(exp_id)
            exp_data = {
                "experiment_id": metadata.experiment_id,
                "experiment_name": metadata.experiment_name,
                "solver": metadata.solver_method,
                "backend": metadata.backend,
                "nx": int(metadata.nx),
                "nt": int(metadata.nt),
                "domain_length": float(metadata.domain_length),
                "final_time": float(metadata.final_time),
                "peak_concentration": float(np.max(C)),
                "final_mass": float(np.sum(C[:, -1]) * float(metadata.dx)),
            }
            report["experiments"].append(exp_data)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Generated JSON report: {output_path}")
