"""
M10 Database and Analysis Unit Tests

Tests for result database, experiment tracking, and analysis framework.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cdts.analysis.database import (
    ResultDatabase,
    ExperimentMetadata,
    ExperimentComparison,
    PerformanceAnalyzer,
    ReportGenerator,
)
from cdts.analysis.tracker import ExperimentTracker
from cdts.analysis.dashboard import PerformanceDashboard


class TestResultDatabase:
    """Test HDF5 result database."""

    def test_database_initialization(self):
        """Database should initialize schema correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.h5"
            db = ResultDatabase(str(db_path))
            assert db_path.exists()

    def test_store_and_retrieve_experiment(self):
        """Should store and retrieve experiment results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.h5"
            db = ResultDatabase(str(db_path))

            # Create test data
            metadata = ExperimentMetadata(
                experiment_id="TEST001",
                experiment_name="Test Experiment",
                timestamp="2026-09-04T00:00:00",
                solver_method="explicit_fdm",
                nx=101,
                nt=51,
                dx=0.0001,
                dt=50.0,
                final_time=200.0,
                domain_length=0.01,
                diffusion_coeff=1e-10,
                clearance=0.001,
                boundary_left=1.0,
                boundary_right=0.0,
            )

            C = np.random.rand(101, 51)
            t = np.linspace(0, 200, 51)
            x = np.linspace(0, 0.01, 101)
            metrics = {"peak": float(np.max(C)), "final_mass": float(np.sum(C[:, -1]))}

            # Store
            db.store_experiment(metadata, C, t, x, metrics)

            # Retrieve
            meta_ret, C_ret, t_ret, x_ret = db.retrieve_experiment("TEST001")

            assert meta_ret.experiment_id == "TEST001"
            assert np.allclose(C, C_ret)
            assert np.allclose(t, t_ret)
            assert np.allclose(x, x_ret)

    def test_list_experiments(self):
        """Should list all stored experiment IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.h5"
            db = ResultDatabase(str(db_path))

            # Store multiple experiments
            for i in range(3):
                metadata = ExperimentMetadata(
                    experiment_id=f"EXP{i:03d}",
                    experiment_name=f"Experiment {i}",
                    timestamp="2026-09-04T00:00:00",
                    solver_method="explicit_fdm",
                    nx=101,
                    nt=51,
                    dx=0.0001,
                    dt=50.0,
                    final_time=200.0,
                    domain_length=0.01,
                    diffusion_coeff=1e-10,
                    clearance=0.0,
                    boundary_left=1.0,
                    boundary_right=0.0,
                )

                C = np.zeros((101, 51))
                t = np.linspace(0, 200, 51)
                x = np.linspace(0, 0.01, 101)
                metrics = {}

                db.store_experiment(metadata, C, t, x, metrics)

            exp_list = db.list_experiments()
            assert len(exp_list) == 3
            assert "EXP000" in exp_list


class TestExperimentTracker:
    """Test SQLite experiment tracker."""

    def test_tracker_initialization(self):
        """Tracker should initialize schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tracker.db"
            tracker = ExperimentTracker(str(db_path))
            assert db_path.exists()

    def test_register_experiment(self):
        """Should register experiment in tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tracker.db"
            tracker = ExperimentTracker(str(db_path))

            tracker.register_experiment(
                exp_id="TEST001",
                name="Test Experiment",
                solver_method="explicit_fdm",
                backend="numpy",
                nx=101,
                nt=51,
                domain_length=0.01,
                diffusion_coeff=1e-10,
                clearance=0.0,
                final_time=200.0,
                hdf5_path="test.h5",
            )

            experiments = tracker.query_experiments()
            assert len(experiments) == 1
            assert experiments[0]['id'] == "TEST001"

    def test_store_and_retrieve_metrics(self):
        """Should store and retrieve metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tracker.db"
            tracker = ExperimentTracker(str(db_path))

            tracker.register_experiment(
                exp_id="TEST001",
                name="Test",
                solver_method="explicit_fdm",
                backend="numpy",
                nx=101,
                nt=51,
                domain_length=0.01,
                diffusion_coeff=1e-10,
                clearance=0.0,
                final_time=200.0,
                hdf5_path="test.h5",
            )

            tracker.store_metrics(
                exp_id="TEST001",
                peak_concentration=1.5,
                penetration_depth=0.008,
                arrival_time=50.0,
                total_mass_initial=0.01,
                total_mass_final=0.009,
                runtime_seconds=2.5,
            )

            metrics = tracker.get_experiment_metrics("TEST001")
            assert metrics is not None
            assert metrics["peak_concentration"] == 1.5
            assert metrics["penetration_depth"] == 0.008

    def test_query_experiments_with_filters(self):
        """Should query experiments with various filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tracker.db"
            tracker = ExperimentTracker(str(db_path))

            # Register multiple experiments
            for i in range(3):
                tracker.register_experiment(
                    exp_id=f"EXP{i:03d}",
                    name=f"Experiment {i}",
                    solver_method="explicit_fdm" if i < 2 else "crank_nicolson",
                    backend="numpy" if i == 0 else "openmp",
                    nx=50 + i * 10,
                    nt=51,
                    domain_length=0.01,
                    diffusion_coeff=1e-10,
                    clearance=0.0,
                    final_time=200.0,
                    hdf5_path="test.h5",
                )

            # Query with filters
            results = tracker.query_experiments(solver_method="explicit_fdm")
            assert len(results) == 2

            results = tracker.query_experiments(backend="openmp")
            assert len(results) == 2

            results = tracker.query_experiments(min_nx=60)
            assert len(results) == 2

    def test_database_statistics(self):
        """Should compute database statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tracker.db"
            tracker = ExperimentTracker(str(db_path))

            for i in range(2):
                tracker.register_experiment(
                    exp_id=f"EXP{i:03d}",
                    name=f"Experiment {i}",
                    solver_method="explicit_fdm",
                    backend="numpy",
                    nx=101,
                    nt=51,
                    domain_length=0.01,
                    diffusion_coeff=1e-10,
                    clearance=0.0,
                    final_time=200.0,
                    hdf5_path="test.h5",
                )

                tracker.store_metrics(
                    exp_id=f"EXP{i:03d}",
                    peak_concentration=1.0,
                    penetration_depth=0.008,
                    arrival_time=50.0,
                    total_mass_initial=0.01,
                    total_mass_final=0.009,
                    runtime_seconds=2.0 + i,
                )

            stats = tracker.get_statistics()
            assert stats["total_experiments"] == 2
            assert stats["unique_solvers"] >= 1
            assert stats["average_runtime_seconds"] > 0


class TestExperimentComparison:
    """Test experiment comparison framework."""

    def _create_test_db(self, tmpdir):
        """Helper to create test database."""
        db_path = Path(tmpdir) / "test.h5"
        db = ResultDatabase(str(db_path))

        for i in range(2):
            metadata = ExperimentMetadata(
                experiment_id=f"EXP{i:03d}",
                experiment_name=f"Experiment {i}",
                timestamp="2026-09-04T00:00:00",
                solver_method="explicit_fdm",
                nx=101,
                nt=51,
                dx=0.0001,
                dt=50.0,
                final_time=200.0,
                domain_length=0.01,
                diffusion_coeff=1e-10,
                clearance=0.0,
                boundary_left=1.0,
                boundary_right=0.0,
            )

            # Create concentration field with varying penetration
            C = np.zeros((101, 51))
            for t_idx in range(51):
                # Simulate diffusion: concentration decays with depth
                C[:, t_idx] = np.exp(-0.5 * (np.linspace(0, 1, 101) ** 2)) * (1 - i * 0.1)

            t = np.linspace(0, 200, 51)
            x = np.linspace(0, 0.01, 101)
            metrics = {}

            db.store_experiment(metadata, C, t, x, metrics)

        return db

    def test_compare_penetration_depth(self):
        """Should compare penetration depths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._create_test_db(tmpdir)
            comparison = ExperimentComparison(db)

            results = comparison.compare_penetration_depth(["EXP000", "EXP001"])
            assert len(results) == 2
            assert all(v >= 0 for v in results.values())

    def test_compare_peak_concentration(self):
        """Should compare peak concentrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._create_test_db(tmpdir)
            comparison = ExperimentComparison(db)

            results = comparison.compare_peak_concentration(["EXP000", "EXP001"])
            assert len(results) == 2
            assert results["EXP000"] > results["EXP001"]

    def test_compare_total_mass(self):
        """Should compare total mass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._create_test_db(tmpdir)
            comparison = ExperimentComparison(db)

            results = comparison.compare_total_mass(["EXP000", "EXP001"])
            assert len(results) == 2
            for exp_id, (initial, final) in results.items():
                assert initial >= 0
                assert final >= 0


class TestReportGenerator:
    """Test report generation."""

    def _create_test_db(self, tmpdir):
        """Helper to create test database."""
        db_path = Path(tmpdir) / "test.h5"
        db = ResultDatabase(str(db_path))

        metadata = ExperimentMetadata(
            experiment_id="TEST001",
            experiment_name="Test Experiment",
            timestamp="2026-09-04T00:00:00",
            solver_method="explicit_fdm",
            nx=101,
            nt=51,
            dx=0.0001,
            dt=50.0,
            final_time=200.0,
            domain_length=0.01,
            diffusion_coeff=1e-10,
            clearance=0.0,
            boundary_left=1.0,
            boundary_right=0.0,
        )

        C = np.random.rand(101, 51)
        t = np.linspace(0, 200, 51)
        x = np.linspace(0, 0.01, 101)
        metrics = {}

        db.store_experiment(metadata, C, t, x, metrics)
        return db

    def test_generate_markdown_summary(self):
        """Should generate markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._create_test_db(tmpdir)
            generator = ReportGenerator(db)

            report_path = Path(tmpdir) / "report.md"
            generator.generate_markdown_summary(["TEST001"], str(report_path))

            assert report_path.exists()
            content = report_path.read_text()
            assert "CDTS" in content
            assert "TEST001" in content

    def test_generate_json_summary(self):
        """Should generate JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._create_test_db(tmpdir)
            generator = ReportGenerator(db)

            report_path = Path(tmpdir) / "report.json"
            generator.generate_json_summary(["TEST001"], str(report_path))

            assert report_path.exists()
            import json
            data = json.loads(report_path.read_text())
            assert data["total_experiments"] == 1
            assert len(data["experiments"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
