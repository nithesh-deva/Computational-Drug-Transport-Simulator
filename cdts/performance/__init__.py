"""
Performance package for CDTS M8.

Provides:
- C++/OpenMP accelerated explicit FDM solver
- Benchmarking against pure Python and NumPy backends
- Thread scaling measurements
"""

from .explicit_omp import ExplicitFDMOmpSolver, is_omp_available
from .benchmark import (
    benchmark_solver,
    compare_backends,
    thread_scaling_study,
    format_benchmark_report,
)

__all__ = [
    "ExplicitFDMOmpSolver",
    "is_omp_available",
    "benchmark_solver",
    "compare_backends",
    "thread_scaling_study",
    "format_benchmark_report",
]
