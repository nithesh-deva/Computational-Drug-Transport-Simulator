"""
C++/CUDA accelerated explicit FDM solver with ctypes integration.

Provides GPU-accelerated computation for explicit finite-difference schemes.
Falls back to NumPy if CUDA library is unavailable.
"""

import numpy as np
import logging
import ctypes
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Search paths for compiled CUDA shared library
_CUDA_LIB_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent / "cpp" / "kernels",
    Path(__file__).parent.parent.parent / "build" / "Release",
    Path(__file__).parent.parent.parent / "build",
    Path(".") / "cpp" / "kernels",
    Path(".") / "build" / "Release",
]


def _find_cuda_library() -> Optional[ctypes.CDLL]:
    """Attempt to load the CUDA-accelerated shared library."""
    lib_names = ["cdts_explicit_cuda", "libcdts_explicit_cuda"]
    extensions = [".dll", ".so", ".dylib"]

    for path in _CUDA_LIB_SEARCH_PATHS:
        if not path.exists():
            continue
        for name in lib_names:
            for ext in extensions:
                candidate = path / f"{name}{ext}"
                if candidate.exists():
                    try:
                        lib = ctypes.CDLL(str(candidate))
                        logger.info(f"Loaded CUDA library: {candidate}")
                        return lib
                    except Exception as e:
                        logger.warning(f"Failed to load {candidate}: {e}")
    return None


_CUDA_LIB = _find_cuda_library()


def is_cuda_available() -> bool:
    """Return True if CUDA library was successfully loaded."""
    if _CUDA_LIB is None:
        return False
    
    # Verify CUDA device is actually available
    try:
        device_count = ctypes.c_int()
        major = ctypes.c_int()
        minor = ctypes.c_int()
        
        _CUDA_LIB.cuda_get_device_info.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        _CUDA_LIB.cuda_get_device_info.restype = ctypes.c_int
        
        result = _CUDA_LIB.cuda_get_device_info(
            ctypes.byref(device_count),
            ctypes.byref(major),
            ctypes.byref(minor)
        )
        
        if result == 0 and device_count.value > 0:
            logger.info(
                f"CUDA device available: {device_count.value} device(s), "
                f"compute capability {major.value}.{minor.value}"
            )
            return True
        else:
            logger.warning("CUDA library loaded but no device available")
            return False
    except Exception as e:
        logger.warning(f"CUDA device query failed: {e}")
        return False


def get_cuda_device_info() -> Dict[str, Any]:
    """Query CUDA device information.
    
    Returns:
        Dictionary with keys: device_count, compute_major, compute_minor
        Returns all zeros if no device available.
    """
    if not is_cuda_available():
        return {"device_count": 0, "compute_major": 0, "compute_minor": 0}
    
    device_count = ctypes.c_int()
    major = ctypes.c_int()
    minor = ctypes.c_int()
    
    _CUDA_LIB.cuda_get_device_info.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    _CUDA_LIB.cuda_get_device_info.restype = ctypes.c_int
    
    _CUDA_LIB.cuda_get_device_info(
        ctypes.byref(device_count),
        ctypes.byref(major),
        ctypes.byref(minor)
    )
    
    return {
        "device_count": device_count.value,
        "compute_major": major.value,
        "compute_minor": minor.value,
    }


class ExplicitFDMCudaSolver:
    """Explicit FDM solver using CUDA GPU acceleration.
    
    If CUDA is unavailable, falls back to NumPy vectorized implementation.
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
    ):
        self.domain_length = domain_length
        self.diffusion_coeff = diffusion_coeff
        self.dx = dx
        self.dt = dt
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.clearance = clearance

        self.x = np.arange(0, domain_length + dx / 2, dx)
        self.nx = len(self.x)
        self.r = diffusion_coeff * dt / (dx ** 2)
        self.k_dt = clearance * dt

        self.backend = "cuda" if _CUDA_LIB is not None and is_cuda_available() else "numpy_vectorized"
        logger.info(
            f"ExplicitFDMCudaSolver initialized: backend={self.backend}, "
            f"nx={self.nx}, r={self.r:.6f}"
        )

    def solve(
        self,
        initial_condition: np.ndarray,
        final_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Solve using available backend."""
        nt = int(final_time / self.dt) + 1
        t = np.linspace(0, final_time, nt)

        C = np.zeros((self.nx, nt), dtype=np.float64, order='C')
        C[:, 0] = initial_condition

        if self.backend == "cuda":
            self._solve_cuda(C, nt)
        else:
            self._solve_numpy(C, nt)

        metrics = {
            "method": "explicit_fdm_cuda",
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
        }
        return self.x, t, C, metrics

    def _solve_cuda(self, C: np.ndarray, nt: int):
        """Call CUDA kernel."""
        nx = self.nx
        r = self.r
        k_dt = self.k_dt
        b_left = self.boundary_left
        b_right = self.boundary_right

        C_init = C[:, 0].copy()

        # Prepare ctypes arguments
        _CUDA_LIB.explicit_fdm_solve_cuda.restype = ctypes.c_int
        _CUDA_LIB.explicit_fdm_solve_cuda.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # C_out
            ctypes.c_int,                     # nx
            ctypes.c_int,                     # nt
            ctypes.c_double,                  # r
            ctypes.c_double,                  # k_dt
            ctypes.c_double,                  # b_left
            ctypes.c_double,                  # b_right
            ctypes.POINTER(ctypes.c_double),  # C_init
        ]

        # Flatten output array (row-major)
        C_flat = C.ravel(order='C')
        C_init_flat = C_init.ravel(order='C')

        result = _CUDA_LIB.explicit_fdm_solve_cuda(
            C_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(nx),
            ctypes.c_int(nt),
            ctypes.c_double(r),
            ctypes.c_double(k_dt),
            ctypes.c_double(b_left),
            ctypes.c_double(b_right),
            C_init_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        if result != 0:
            raise RuntimeError(f"CUDA solver failed with code {result}")

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
