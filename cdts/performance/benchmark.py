"""
Benchmarking framework for M8 performance studies.

Compares:
- Pure Python explicit FDM
- NumPy-vectorized explicit FDM
- C++/OpenMP explicit FDM

Measures:
- Runtime per time step
- Total runtime
- Speedup relative to baseline
- Thread scaling
"""

import numpy as np
import time
import logging
import json
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict

from ..solvers.explicit_fdm import ExplicitFDMSolver
from .explicit_omp import ExplicitFDMOmpSolver, is_omp_available
from .explicit_cuda import ExplicitFDMCudaSolver, is_cuda_available, get_cuda_device_info

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    backend: str
    nx: int
    nt: int
    total_time_s: float
    time_per_step_ms: float
    throughput_points_per_s: float
    n_threads: int = 1
    speedup: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


def _time_function(fn: Callable, n_repeats: int = 3) -> float:
    """Time a function, returning the best of n_repeats."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


def benchmark_solver(
    solver_name: str,
    solver_factory: Callable,
    n_repeats: int = 3,
) -> BenchmarkResult:
    """Benchmark a single solver backend.

    Args:
        solver_name: Backend name for reporting
        solver_factory: Callable returning a solver with .solve(ic, T)
        n_repeats: Number of timing repeats

    Returns:
        BenchmarkResult with timing metrics
    """
    # Build solver once to inspect dimensions
    temp_solver = solver_factory()
    nx = temp_solver.nx if hasattr(temp_solver, "nx") else temp_solver.n_nodes
    r = temp_solver.r if hasattr(temp_solver, "r") else 0.0
    k = temp_solver.clearance if hasattr(temp_solver, "clearance") else 0.0
    dt = temp_solver.dt if hasattr(temp_solver, "dt") else 0.0
    T = 200.0
    nt = int(T / dt) + 1 if dt > 0 else 100

    # Use a simple uniform IC to avoid extra setup in timing
    ic = np.ones(nx) * 0.5

    def run():
        s = solver_factory()
        s.solve(ic, T)

    total_time = _time_function(run, n_repeats=n_repeats)
    time_per_step_ms = (total_time / max(1, nt - 1)) * 1000.0
    throughput = (nx * nt) / total_time if total_time > 0 else 0.0

    n_threads = 1
    if hasattr(temp_solver, "n_threads"):
        n_threads = temp_solver.n_threads
    elif hasattr(temp_solver, "backend"):
        if temp_solver.backend == "openmp_cpp":
            n_threads = temp_solver.n_threads or 1

    return BenchmarkResult(
        backend=solver_name,
        nx=nx,
        nt=nt,
        total_time_s=total_time,
        time_per_step_ms=time_per_step_ms,
        throughput_points_per_s=throughput,
        n_threads=n_threads,
    )


def compare_backends(
    domain_length: float = 0.01,
    diffusion_coeff: float = 1e-10,
    dx: float = 0.0001,
    dt: float = 50.0,
    clearance: float = 0.001,
    n_repeats: int = 3,
) -> List[BenchmarkResult]:
    """Compare all available explicit FDM backends.

    Returns:
        List of BenchmarkResult, sorted by total_time_s
    """
    b_left = 1.0
    b_right = 0.0
    T = 200.0

    results: List[BenchmarkResult] = []

    # 1. Pure Python explicit FDM
    def make_python():
        return ExplicitFDMSolver(
            domain_length=domain_length,
            diffusion_coeff=diffusion_coeff,
            dx=dx,
            dt=dt,
            boundary_left=b_left,
            boundary_right=b_right,
            clearance=clearance,
        )

    results.append(benchmark_solver("python_loop", make_python, n_repeats=n_repeats))

    # 2. C++/OpenMP explicit FDM
    def make_omp():
        return ExplicitFDMOmpSolver(
            domain_length=domain_length,
            diffusion_coeff=diffusion_coeff,
            dx=dx,
            dt=dt,
            boundary_left=b_left,
            boundary_right=b_right,
            clearance=clearance,
            n_threads=0,
        )

    if is_omp_available():
        results.append(benchmark_solver("openmp_cpp", make_omp, n_repeats=n_repeats))
    else:
        logger.info("C++ OpenMP backend not available; skipping openmp_cpp benchmark")

    # 3. CUDA explicit FDM
    def make_cuda():
        return ExplicitFDMCudaSolver(
            domain_length=domain_length,
            diffusion_coeff=diffusion_coeff,
            dx=dx,
            dt=dt,
            boundary_left=b_left,
            boundary_right=b_right,
            clearance=clearance,
        )

    if is_cuda_available():
        results.append(benchmark_solver("cuda_gpu", make_cuda, n_repeats=n_repeats))
    else:
        logger.info("CUDA backend not available; skipping cuda_gpu benchmark")

    # Normalize speedups relative to slowest backend
    if results:
        baseline = max(r.total_time_s for r in results)
        for r in results:
            r.speedup = baseline / r.total_time_s if r.total_time_s > 0 else 1.0

    results.sort(key=lambda r: r.total_time_s)
    return results


def thread_scaling_study(
    domain_length: float = 0.01,
    diffusion_coeff: float = 1e-10,
    dx: float = 0.0001,
    dt: float = 50.0,
    clearance: float = 0.0,
    thread_list: Optional[List[int]] = None,
    n_repeats: int = 3,
) -> List[BenchmarkResult]:
    """Measure OpenMP thread scaling.

    Args:
        thread_list: List of thread counts to test (default [1,2,4,8])
    """
    if not is_omp_available():
        logger.warning("C++ OpenMP backend not available; thread scaling study skipped")
        return []

    if thread_list is None:
        thread_list = [1, 2, 4, 8]

    results: List[BenchmarkResult] = []
    for n_threads in thread_list:
        def make_omp():
            return ExplicitFDMOmpSolver(
                domain_length=domain_length,
                diffusion_coeff=diffusion_coeff,
                dx=dx,
                dt=dt,
                boundary_left=1.0,
                boundary_right=0.0,
                clearance=clearance,
                n_threads=n_threads,
            )

        res = benchmark_solver(
            f"openmp_cpp_{n_threads}t",
            make_omp,
            n_repeats=n_repeats,
        )
        res.n_threads = n_threads
        results.append(res)

    # Speedup relative to 1-thread run
    if results:
        t1 = next((r.total_time_s for r in results if r.n_threads == 1), None)
        if t1 and t1 > 0:
            for r in results:
                r.speedup = t1 / r.total_time_s

    results.sort(key=lambda r: r.n_threads)
    return results


def format_benchmark_report(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as a human-readable table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Backend':<20} {'Threads':>7} {'Total[s]':>10} {'Step[ms]':>10} {'Speedup':>8} {'Throughput':>15}")
    lines.append("-" * 80)
    for r in results:
        lines.append(
            f"{r.backend:<20} {r.n_threads:>7} {r.total_time_s:>10.4f} "
            f"{r.time_per_step_ms:>10.4f} {r.speedup:>8.2f} "
            f"{r.throughput_points_per_s:>15.2e}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def save_benchmark_report(
    results: List[BenchmarkResult],
    path: str,
    metadata: Optional[Dict] = None,
):
    """Save benchmark results to JSON."""
    report = {
        "results": [r.to_dict() for r in results],
        "metadata": metadata or {},
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Benchmark report saved: {path}")
