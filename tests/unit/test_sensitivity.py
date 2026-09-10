"""
Unit tests for M7 Sensitivity Analysis and Parametric Sweeps.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cdts.sensitivity.metrics import (
    normalized_sensitivity,
    local_sensitivity,
    sobol_indices_estimate,
    sensitivity_tornado_data,
)
from cdts.sensitivity.sweep_runner import ParametricSweepRunner, SweepConfig
from cdts.sensitivity.results import SweepResults
from cdts.sensitivity.analyzer import SensitivityAnalyzer


class TestNormalizedSensitivity:
    """Test normalized sensitivity coefficient."""

    def test_positive_sensitivity(self):
        """Positive correlation should give positive sensitivity."""
        p = np.linspace(1.0, 2.0, 20)
        y = 3.0 * p + 0.1
        S = normalized_sensitivity(y, p, 1.5)
        assert S > 0
        mean_y = np.mean(y)
        expected = 3.0 * 1.5 / mean_y
        assert np.isclose(S, expected, rtol=0.1)

    def test_negative_sensitivity(self):
        """Negative correlation should give negative sensitivity."""
        p = np.linspace(1.0, 2.0, 20)
        y = -4.0 * p + 10.0
        S = normalized_sensitivity(y, p, 1.5)
        assert S < 0
        mean_y = np.mean(y)
        expected = -4.0 * 1.5 / mean_y
        assert np.isclose(S, expected, rtol=0.1)

    def test_zero_sensitivity(self):
        """No correlation should give near-zero sensitivity."""
        p = np.linspace(1.0, 2.0, 20)
        y = np.ones_like(p) * 5.0
        S = normalized_sensitivity(y, p, 1.5)
        assert abs(S) < 0.1


class TestSobolIndices:
    """Test Sobol index estimation."""

    def test_highly_influential_parameter(self):
        """Sobol indices should be valid for a known input-output relationship."""
        rng = np.random.default_rng(0)
        N = 500
        p = rng.uniform(0, 1, (N, 2))
        y = p[:, 0] ** 2
        first, total = sobol_indices_estimate(p, y)
        assert np.all(first >= 0.0)
        assert np.all(first <= 1.0)
        assert np.all(total >= 0.0)
        assert np.all(total <= 1.0)

    def test_independent_parameters(self):
        """Independent noise should give low Sobol indices."""
        rng = np.random.default_rng(1)
        N = 300
        p = rng.uniform(0, 1, (N, 3))
        y = rng.normal(0, 1, N)
        first, total = sobol_indices_estimate(p, y)
        assert np.all(first >= 0.0)
        assert np.all(first <= 1.0)


class TestTornadoData:
    """Test tornado plot data preparation."""

    def test_tornado_sorting(self):
        """Top sensitivities should be sorted by absolute value."""
        sensitivities = {
            "D": 2.5,
            "k": -1.8,
            "L": 0.3,
            "C0": 0.1,
        }
        names, values = sensitivity_tornado_data(sensitivities, top_n=3)
        assert len(names) == 3
        assert names[0] == "D"
        assert names[1] == "k"
        assert names[2] == "L"


class TestLocalSensitivity:
    """Test local sensitivity via perturbation."""

    def test_local_linear(self):
        """Linear function should give exact derivative."""
        def run_fn(params):
            y = 3.0 * params["D"] - 2.0 * params["k"] + 1.0
            return {"peak": y}

        baseline = {"D": 1.0, "k": 0.5}
        S = local_sensitivity(baseline, 0.01, run_fn, output_key="peak")
        assert np.isclose(S["D"], 3.0, rtol=0.01)
        assert np.isclose(S["k"], -2.0, rtol=0.01)

    def test_local_zero_param(self):
        """Zero parameter should return 0 sensitivity."""
        def run_fn(params):
            return {"peak": params["k"]}

        baseline = {"D": 0.0, "k": 1.0}
        S = local_sensitivity(baseline, 0.01, run_fn, output_key="peak")
        assert S["D"] == 0.0


class TestSweepConfig:
    """Test sweep configuration."""

    def test_valid_config(self):
        config = SweepConfig(
            experiment_id="test",
            base_config_path="examples/M1_validation.yaml",
            parameters={"D": [1e-10, 2e-10]},
            strategy="full_factorial",
        )
        config.validate()

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            SweepConfig(
                experiment_id="test",
                base_config_path="examples/M1_validation.yaml",
                parameters={"D": [1e-10]},
                strategy="invalid",
            )


class TestSweepRunner:
    """Test parametric sweep runner."""

    def test_full_factorial_generation(self):
        config = SweepConfig(
            experiment_id="test",
            base_config_path="examples/M1_validation.yaml",
            parameters={"D": [1e-10, 2e-10], "k": [0.0, 0.001]},
            strategy="full_factorial",
        )
        runner = ParametricSweepRunner(config)
        combos = runner.generate_parameter_combinations()
        assert len(combos) == 4

    def test_lhs_generation(self):
        config = SweepConfig(
            experiment_id="test",
            base_config_path="examples/M1_validation.yaml",
            parameters={"D": [1e-10, 2e-10], "k": [0.0, 0.001]},
            strategy="lhs",
            n_samples=10,
        )
        runner = ParametricSweepRunner(config)
        combos = runner.generate_parameter_combinations()
        assert len(combos) == 10

    def test_sweep_execution(self):
        """Run a small sweep with mock function."""
        config = SweepConfig(
            experiment_id="test_sweep",
            base_config_path="examples/M1_validation.yaml",
            parameters={"D": [1e-10, 2e-10]},
            strategy="full_factorial",
            output_directory="results/test_sensitivity_unit",
        )

        def mock_run(params):
            D = params.get("diffusion_coefficient", params.get("D", 1e-10))
            return {
                "peak_concentration": float(D * 1e10),
                "final_mass": float(D * 1e10 * 0.5),
                "mass_remaining_fraction": 0.5,
                "runtime_seconds": 0.01,
                "method": "explicit_fdm",
                "D": float(D),
                "k": 0.0,
                "L": 0.01,
            }

        runner = ParametricSweepRunner(config, run_simulation_fn=mock_run)
        results = runner.run()
        assert results.success_count() == 2
        assert results.failure_count() == 0


class TestSweepResults:
    """Test sweep result aggregation."""

    def test_summary_statistics(self):
        results_data = [
            {"status": "success", "peak_concentration": 1.0, "final_mass": 0.5, "D": 1e-10},
            {"status": "success", "peak_concentration": 2.0, "final_mass": 1.0, "D": 2e-10},
            {"status": "failed", "error": "test"},
        ]
        sr = SweepResults("test", results_data, "results")
        summary = sr.summary()
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["peak_concentration_mean"] == 1.5
        assert summary["D_mean"] == 1.5e-10


class TestSensitivityAnalyzer:
    """Test end-to-end sensitivity analyzer."""

    def test_analyzer_with_mock(self):
        """Test analyzer with mock simulation function."""
        base_config = "examples/M1_validation.yaml"
        params = {"D": [1e-10, 1.5e-10, 2e-10]}

        def mock_run(pdict):
            D = pdict.get("diffusion_coefficient", 1e-10)
            return {
                "peak_concentration": float(D * 1e10),
                "final_mass": float(D * 5e9),
                "mass_remaining_fraction": 0.6,
                "runtime_seconds": 0.01,
                "method": "explicit_fdm",
                "D": float(D),
                "k": 0.0,
                "L": 0.01,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SensitivityAnalyzer(
                base_config_path=base_config,
                sweep_parameters=params,
                output_directory=tmpdir,
                strategy="full_factorial",
                solver_method="explicit_fdm",
            )
            analyzer.run_simulation_fn = mock_run
            results = analyzer.run()

            assert results.success_count() == 3
            assert len(analyzer.get_metrics()) > 0
            assert "normalized_sensitivity" in analyzer.get_metrics()
            assert "sobol_peak_concentration" in analyzer.get_metrics()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
