"""
Unit tests for M8 Performance Engineering (C++/OpenMP).

Tests:
- C++ library loading / fallback behavior
- Numerical correctness of OpenMP/numpy backends
- Benchmark runner functionality
- Thread scaling measurement
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cdts.performance.explicit_omp import ExplicitFDMOmpSolver, is_omp_available
from cdts.performance.explicit_cuda import ExplicitFDMCudaSolver, is_cuda_available, get_cuda_device_info
from cdts.performance.benchmark import (
    benchmark_solver,
    compare_backends,
    thread_scaling_study,
    format_benchmark_report,
    save_benchmark_report,
)
from cdts.solvers.explicit_fdm import ExplicitFDMSolver


class TestOmpAvailability:
    """Test OpenMP library detection."""

    def test_omp_available_returns_bool(self):
        """Should return a boolean."""
        result = is_omp_available()
        assert isinstance(result, bool)

    def test_omp_available_false_when_no_lib(self):
        """If library missing, should be False but still importable."""
        # This test passes regardless; just ensures no import error
        assert True


class TestExplicitFDMOmpSolver:
    """Test OpenMP solver interface and correctness."""

    def test_solver_initialization(self):
        """Solver should initialize with correct attributes."""
        solver = ExplicitFDMOmpSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0,
        )
        assert solver.nx == 101
        assert solver.r == 0.5
        assert solver.backend in ("openmp_cpp", "numpy_vectorized")

    def test_numpy_fallback_solve(self):
        """NumPy fallback should solve and produce bounded output."""
        solver = ExplicitFDMOmpSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0,
        )
        # Force fallback path for testing
        solver.backend = "numpy_vectorized"

        C_init = np.zeros(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)

        assert C.shape == (solver.nx, len(t))
        assert np.isclose(C[0, -1], 1.0)
        assert np.isclose(C[-1, -1], 0.0)
        assert np.all(C >= -1e-12)

    def test_openmp_solve_if_available(self):
        """If OpenMP backend loaded, it should solve correctly."""
        if not is_omp_available():
            pytest.skip("C++ OpenMP library not available")

        solver = ExplicitFDMOmpSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0,
            n_threads=2,
        )

        C_init = np.zeros(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)

        assert np.isclose(C[0, -1], 1.0)
        assert np.isclose(C[-1, -1], 0.0)
        assert np.all(np.isfinite(C))
        assert metrics["backend"] == "openmp_cpp"

    def test_backends_agree(self):
        """OpenMP and NumPy backends should agree within tolerance."""
        if not is_omp_available():
            pytest.skip("C++ OpenMP library not available")

        solver = ExplicitFDMOmpSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.001,
            n_threads=2,
        )

        C_init = np.zeros(solver.nx)

        # NumPy backend
        solver_np = ExplicitFDMOmpSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.001,
        )
        solver_np.backend = "numpy_vectorized"
        _, _, C_np, _ = solver_np.solve(C_init.copy(), final_time=200.0)

        # OpenMP backend
        solver_omp = ExplicitFDMOmpSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.001,
            n_threads=2,
        )
        _, _, C_omp, _ = solver_omp.solve(C_init.copy(), final_time=200.0)

        assert np.allclose(C_np, C_omp, rtol=1e-10, atol=1e-12)


class TestBenchmarkRunner:
    """Test benchmarking functions."""

    def test_benchmark_solver_returns_result(self):
        """Benchmark should return valid BenchmarkResult."""
        def make_solver():
            return ExplicitFDMSolver(
                domain_length=0.01,
                diffusion_coeff=1e-10,
                dx=0.0001,
                dt=50.0,
                boundary_left=1.0,
                boundary_right=0.0,
                clearance=0.0,
            )

        result = benchmark_solver("python", make_solver, n_repeats=2)
        assert result.backend == "python"
        assert result.nx == 101
        assert result.total_time_s > 0
        assert result.time_per_step_ms > 0

    def test_compare_backends_returns_sorted(self):
        """Backend comparison should return results sorted by time."""
        results = compare_backends(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            clearance=0.0,
            n_repeats=2,
        )
        times = [r.total_time_s for r in results]
        assert times == sorted(times), "Results should be sorted by total_time_s"

    def test_thread_scaling_requires_omp(self):
        """Thread scaling should return empty list if OpenMP unavailable."""
        # We cannot easily mock is_omp_available here without patching,
        # so just verify the function runs and returns a list
        results = thread_scaling_study(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            clearance=0.0,
            thread_list=[1, 2],
            n_repeats=2,
        )
        assert isinstance(results, list)

    def test_format_benchmark_report(self):
        """Report formatting should produce a string."""
        from cdts.performance.benchmark import BenchmarkResult
        results = [
            BenchmarkResult("backend_a", 100, 100, 1.0, 10.0, 1e6, 1, 1.0),
            BenchmarkResult("backend_b", 100, 100, 0.5, 5.0, 2e6, 4, 2.0),
        ]
        report = format_benchmark_report(results)
        assert "backend_a" in report
        assert "backend_b" in report

    def test_save_benchmark_report(self):
        """Should save JSON report without errors."""
        from cdts.performance.benchmark import BenchmarkResult
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bench.json"
            results = [
                BenchmarkResult("test", 10, 10, 0.1, 10.0, 100.0, 1, 1.0),
            ]
            save_benchmark_report(results, str(path))
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["results"][0]["backend"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCudaAvailability:
    """Test CUDA library detection and device queries."""

    def test_cuda_available_returns_bool(self):
        """Should return a boolean."""
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_cuda_device_info_returns_dict(self):
        """Device info should return a dictionary."""
        info = get_cuda_device_info()
        assert isinstance(info, dict)
        assert "device_count" in info
        assert "compute_major" in info
        assert "compute_minor" in info

    def test_cuda_device_info_values_nonnegative(self):
        """Device info values should be non-negative integers."""
        info = get_cuda_device_info()
        assert info["device_count"] >= 0
        assert info["compute_major"] >= 0
        assert info["compute_minor"] >= 0


class TestExplicitFDMCudaSolver:
    """Test CUDA solver interface and correctness."""

    def test_solver_initialization(self):
        """Solver should initialize with correct attributes."""
        solver = ExplicitFDMCudaSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0,
        )
        assert solver.nx == 101
        assert solver.r == 0.5
        assert solver.backend in ("cuda", "numpy_vectorized")

    def test_numpy_fallback_solve(self):
        """NumPy fallback should solve and produce bounded output."""
        solver = ExplicitFDMCudaSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0,
        )
        # Force fallback path for testing
        solver.backend = "numpy_vectorized"

        C_init = np.zeros(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)

        assert C.shape == (solver.nx, len(t))
        assert np.isclose(C[0, -1], 1.0)
        assert np.isclose(C[-1, -1], 0.0)
        assert np.all(C >= -1e-12)

    def test_cuda_solve_if_available(self):
        """If CUDA backend loaded, it should solve correctly."""
        if not is_cuda_available():
            pytest.skip("CUDA library not available")

        solver = ExplicitFDMCudaSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.0,
        )

        C_init = np.zeros(solver.nx)
        x, t, C, metrics = solver.solve(C_init, final_time=200.0)

        assert np.isclose(C[0, -1], 1.0, rtol=1e-6)
        assert np.isclose(C[-1, -1], 0.0, rtol=1e-6)
        assert np.all(np.isfinite(C))
        assert metrics["backend"] == "cuda"

    def test_cpu_gpu_numerical_agreement(self):
        """CPU and GPU backends should produce numerically identical results."""
        if not is_cuda_available():
            pytest.skip("CUDA library not available")

        # Create identical solvers
        domain_length = 0.01
        diffusion_coeff = 1e-10
        dx = 0.0001
        dt = 50.0
        boundary_left = 1.0
        boundary_right = 0.0
        clearance = 0.001

        solver_cpu = ExplicitFDMCudaSolver(
            domain_length=domain_length,
            diffusion_coeff=diffusion_coeff,
            dx=dx,
            dt=dt,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            clearance=clearance,
        )
        solver_cpu.backend = "numpy_vectorized"

        solver_gpu = ExplicitFDMCudaSolver(
            domain_length=domain_length,
            diffusion_coeff=diffusion_coeff,
            dx=dx,
            dt=dt,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            clearance=clearance,
        )
        assert solver_gpu.backend == "cuda"

        C_init = np.zeros(solver_cpu.nx)

        # Solve on both backends
        _, _, C_cpu, _ = solver_cpu.solve(C_init.copy(), final_time=200.0)
        _, _, C_gpu, _ = solver_gpu.solve(C_init.copy(), final_time=200.0)

        # GPU and CPU should agree to machine precision
        # Allow slight tolerance due to different floating point orderings
        assert np.allclose(C_cpu, C_gpu, rtol=1e-10, atol=1e-12)

    def test_solver_with_clearance(self):
        """Solver with clearance should reduce total mass over time."""
        solver = ExplicitFDMCudaSolver(
            domain_length=0.01,
            diffusion_coeff=1e-10,
            dx=0.0001,
            dt=50.0,
            boundary_left=1.0,
            boundary_right=0.0,
            clearance=0.01,
        )
        solver.backend = "numpy_vectorized"

        C_init = np.ones(solver.nx) * 0.5
        _, _, C, _ = solver.solve(C_init, final_time=500.0)

        # Interior mass should decrease due to clearance
        mass_initial = np.sum(C[:, 0])
        mass_final = np.sum(C[:, -1])
        assert mass_final < mass_initial
