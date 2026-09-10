"""
C++/OpenMP accelerated explicit FDM solver with ctypes integration.

Falls back to a NumPy-vectorized implementation if the shared library
cannot be loaded.
"""

import numpy as np
import logging
import ctypes
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to locate the compiled shared library
_LIB_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent / "cpp" / "kernels",
    Path(__file__).parent.parent.parent / "build" / "Release",
    Path(__file__).parent.parent.parent / "build",
    Path(".") / "cpp" / "kernels",
    Path(".") / "build" / "Release",
]


def _find_library() -> Optional[ctypes.CDLL]:
    """Attempt to load the C++ OpenMP shared library."""
    lib_names = ["cdts_explicit", "libcdts_explicit"]
    extensions = [".dll", ".so", ".dylib"]

    for path in _LIB_SEARCH_PATHS:
        if not path.exists():
            continue
        for name in lib_names:
            for ext in extensions:
                candidate = path / f"{name}{ext}"
                if candidate.exists():
                    try:
                        lib = ctypes.CDLL(str(candidate))
                        logger.info(f"Loaded C++ OpenMP library: {candidate}")
                        return lib
                    except Exception as e:
                        logger.warning(f"Failed to load {candidate}: {e}")
    return None


_LIB = _find_library()


def is_omp_available() -> bool:
    """Return True if the C++ OpenMP library was successfully loaded."""
    return _LIB is not None


class ExplicitFDMOmpSolver:
    """Explicit FDM solver using C++/OpenMP acceleration.

    If the C++ backend is unavailable, falls back to a NumPy-vectorized
    implementation on the CPU.
    """

    def __init__(
        self,
        domain_length: float,
        diffusion_coeff: float,
        dx: float,
        dt: float,
        boundary_left: float,
        boundary_right: float,
        clearance: float = 0.0,
        n_threads: int = 0,
    ):
        self.domain_length = domain_length
        self.diffusion_coeff = diffusion_coeff
        self.dx = dx
        self.dt = dt
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.clearance = clearance
        self.n_threads = n_threads

        self.x = np.arange(0, domain_length + dx / 2, dx)
        self.nx = len(self.x)
        self.r = diffusion_coeff * dt / (dx ** 2)
        self.k_dt = clearance * dt

        self.backend = "openmp_cpp" if _LIB is not None else "numpy_vectorized"
        logger.info(
            f"ExplicitFDMOmpSolver initialized: backend={self.backend}, "
            f"nx={self.nx}, r={self.r:.6f}, threads={n_threads if n_threads > 0 else 'auto'}"
        )

    def solve(
        self,
        initial_condition: np.ndarray,
        final_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve using available backend."""
        nt = int(final_time / self.dt) + 1
        t = np.linspace(0, final_time, nt)

        C = np.zeros((self.nx, nt))
        C[:, 0] = initial_condition

        if self.backend == "openmp_cpp":
            self._solve_cpp(C, nt)
        else:
            self._solve_numpy(C, nt)

        metrics = {
            "method": "explicit_fdm_omp",
            "backend": self.backend,
            "nx": self.nx,
            "nt": nt,
            "dx": self.dx,
            "dt": self.dt,
            "r": self.r,
            "diffusion_coeff": self.diffusion_coeff,
            "clearance": self.clearance,
            "domain_length": self.domain_length,
            "final_time": final_time,
            "n_threads": self.n_threads if self.n_threads > 0 else _LIB.omp_get_available_threads() if _LIB else 1,
        }
        return self.x, t, C, metrics

    def _solve_cpp(self, C: np.ndarray, nt: int):
        """Call C++ OpenMP kernel."""
        nx = self.nx
        r = self.r
        k_dt = self.k_dt
        b_left = self.boundary_left
        b_right = self.boundary_right
        n_threads = self.n_threads

        C_init = C[:, 0].copy()

        # Prepare ctypes arguments
        _LIB.explicit_fdm_solve_omp.restype = None
        _LIB.explicit_fdm_solve_omp.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # C_out
            ctypes.c_int,                     # nx
            ctypes.c_int,                     # nt
            ctypes.c_double,                  # r
            ctypes.c_double,                  # k_dt
            ctypes.c_double,                  # b_left
            ctypes.c_double,                  # b_right
            ctypes.POINTER(ctypes.c_double),  # C_init
            ctypes.c_int,                     # n_threads
        ]

        # Flatten output array
        C_flat = C.ravel()
        C_init_flat = C_init.ravel()

        _LIB.explicit_fdm_solve_omp(
            C_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(nx),
            ctypes.c_int(nt),
            ctypes.c_double(r),
            ctypes.c_double(k_dt),
            ctypes.c_double(b_left),
            ctypes.c_double(b_right),
            C_init_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_threads),
        )

    def _solve_numpy(self, C: np.ndarray, nt: int):
        """NumPy-vectorized fallback."""
        r = self.r
        k_dt = self.k_dt
        b_left = self.boundary_left
        b_right = self.boundary_right

        for n in range(nt - 1):
            C_current = C[:, n]
            diffusion = r * (C_current[2:] - 2.0 * C_current[1:-1] + C_current[:-2])
            clearance = -k_dt * C_current[1:-1]
            C_next_interior = C_current[1:-1] + diffusion + clearance
            C_next_interior = np.maximum(C_next_interior, 0.0)

            C[1:-1, n + 1] = C_next_interior
            C[0, n + 1] = b_left
            C[-1, n + 1] = b_right
