"""
M10 Experiment Tracker

Tracks all experiments with query capabilities and statistical analysis.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """SQLite-based experiment tracker for fast queries."""

    def __init__(self, db_path: str):
        """Initialize experiment tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.isolation_level = None  # Autocommit mode
            cursor = conn.cursor()

            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    solver_method TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    nx INTEGER NOT NULL,
                    nt INTEGER NOT NULL,
                    domain_length REAL NOT NULL,
                    diffusion_coeff REAL NOT NULL,
                    clearance REAL NOT NULL,
                    final_time REAL NOT NULL,
                    hdf5_path TEXT NOT NULL,
                    status TEXT DEFAULT 'completed'
                )
            """)

            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    experiment_id TEXT PRIMARY KEY,
                    peak_concentration REAL,
                    penetration_depth REAL,
                    arrival_time REAL,
                    total_mass_initial REAL,
                    total_mass_final REAL,
                    mass_loss_percent REAL,
                    runtime_seconds REAL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)

            # Performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    total_time_s REAL NOT NULL,
                    throughput_points_per_s REAL,
                    speedup REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)

    def register_experiment(
        self,
        exp_id: str,
        name: str,
        solver_method: str,
        backend: str,
        nx: int,
        nt: int,
        domain_length: float,
        diffusion_coeff: float,
        clearance: float,
        final_time: float,
        hdf5_path: str,
    ):
        """Register new experiment.
        
        Args:
            exp_id: Experiment ID
            name: Experiment name
            solver_method: Numerical solver (explicit_fdm, crank_nicolson, fem)
            backend: Compute backend (python, numpy, openmp, cuda)
            nx, nt: Grid dimensions
            domain_length: Tissue thickness (m)
            diffusion_coeff: Diffusion coefficient (m²/s)
            clearance: Clearance coefficient (1/s)
            final_time: Simulation end time (s)
            hdf5_path: Path to HDF5 result file
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO experiments
                (id, name, timestamp, solver_method, backend, nx, nt,
                 domain_length, diffusion_coeff, clearance, final_time, hdf5_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, name, datetime.now().isoformat(), solver_method, backend,
                nx, nt, domain_length, diffusion_coeff, clearance, final_time, hdf5_path
            ))
            conn.commit()
        logger.info(f"Registered experiment: {exp_id}")

    def store_metrics(
        self,
        exp_id: str,
        peak_concentration: float,
        penetration_depth: float,
        arrival_time: float,
        total_mass_initial: float,
        total_mass_final: float,
        runtime_seconds: float,
    ):
        """Store metrics for experiment.
        
        Args:
            exp_id: Experiment ID
            peak_concentration: Maximum concentration (mol/m³)
            penetration_depth: Penetration depth (m)
            arrival_time: Time to reach threshold (s)
            total_mass_initial: Initial total mass (mol/m)
            total_mass_final: Final total mass (mol/m)
            runtime_seconds: Total runtime (s)
        """
        mass_loss_percent = 0.0
        if total_mass_initial > 0:
            mass_loss_percent = 100.0 * (total_mass_initial - total_mass_final) / total_mass_initial

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metrics
                (experiment_id, peak_concentration, penetration_depth, arrival_time,
                 total_mass_initial, total_mass_final, mass_loss_percent, runtime_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, peak_concentration, penetration_depth, arrival_time,
                total_mass_initial, total_mass_final, mass_loss_percent, runtime_seconds
            ))
            conn.commit()
        logger.info(f"Stored metrics for: {exp_id}")

    def query_experiments(
        self,
        solver_method: Optional[str] = None,
        backend: Optional[str] = None,
        min_nx: Optional[int] = None,
        max_nx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query experiments with filters.
        
        Args:
            solver_method: Filter by solver
            backend: Filter by backend
            min_nx, max_nx: Grid size range
            
        Returns:
            List of matching experiment records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM experiments WHERE 1=1"
            params = []

            if solver_method:
                query += " AND solver_method = ?"
                params.append(solver_method)
            if backend:
                query += " AND backend = ?"
                params.append(backend)
            if min_nx:
                query += " AND nx >= ?"
                params.append(min_nx)
            if max_nx:
                query += " AND nx <= ?"
                params.append(max_nx)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_experiment_metrics(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Dictionary of metrics or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics WHERE experiment_id = ?", (exp_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def store_performance_benchmark(
        self,
        exp_id: str,
        backend: str,
        total_time_s: float,
        throughput_points_per_s: Optional[float] = None,
        speedup: Optional[float] = None,
    ):
        """Store performance benchmark result.
        
        Args:
            exp_id: Experiment ID
            backend: Backend name
            total_time_s: Total runtime
            throughput_points_per_s: Throughput metric
            speedup: Speedup relative to baseline
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance
                (experiment_id, backend, total_time_s, throughput_points_per_s, speedup, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                exp_id, backend, total_time_s, throughput_points_per_s, speedup,
                datetime.now().isoformat()
            ))
            conn.commit()

    def get_performance_comparison(self, exp_id: str) -> List[Dict[str, Any]]:
        """Get performance benchmarks for experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            List of performance records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT backend, total_time_s, throughput_points_per_s, speedup "
                "FROM performance WHERE experiment_id = ? ORDER BY total_time_s",
                (exp_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def list_all_experiments(self) -> List[str]:
        """List all experiment IDs.
        
        Returns:
            List of experiment IDs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM experiments ORDER BY timestamp DESC")
            return [row[0] for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM experiments")
            total_experiments = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT solver_method) FROM experiments")
            unique_solvers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT backend) FROM experiments")
            unique_backends = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(runtime_seconds) FROM metrics")
            avg_runtime = cursor.fetchone()[0] or 0.0

            return {
                "total_experiments": total_experiments,
                "unique_solvers": unique_solvers,
                "unique_backends": unique_backends,
                "average_runtime_seconds": avg_runtime,
            }
