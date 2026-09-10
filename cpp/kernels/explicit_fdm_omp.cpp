/**
 * M8 Performance Kernel: Explicit FDM with OpenMP
 *
 * Governing equation:
 *     dC/dt = D * d2C/dx2 - k * C
 *
 * Spatial discretization (centered differences):
 *     d2C/dx2[i] = (C[i+1] - 2*C[i] + C[i-1]) / dx^2
 *
 * Temporal discretization (forward Euler):
 *     C_next[i] = C[i] + dt * (D * d2C/dx2[i] - k * C[i])
 *               = C[i] + r * (C[i+1] - 2*C[i] + C[i-1]) - k*dt * C[i]
 *
 * where r = D * dt / dx^2  (Fourier number)
 *
 * Boundary conditions: Dirichlet at i=0 and i=nx-1
 *
 * Compile with OpenMP:
 *   g++ -O3 -fopenmp -shared -o cdts_explicit.dll explicit_fdm_omp.cpp
 *   cl /O2 /openmp /LD explicit_fdm_omp.cpp
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * Perform one explicit FDM time step with OpenMP parallelization.
 *
 * @param C          Input concentration array [nx]
 * @param C_next     Output concentration array [nx]
 * @param nx         Number of spatial grid points
 * @param r          Fourier number = D * dt / dx^2
 * @param k_dt       Clearance * dt [1/s * s = dimensionless]
 * @param b_left     Left boundary value (Dirichlet)
 * @param b_right    Right boundary value (Dirichlet)
 */
EXPORT void explicit_fdm_step_omp(
    const double* C,
    double* C_next,
    int nx,
    double r,
    double k_dt,
    double b_left,
    double b_right
) {
    // Apply boundary conditions
    C_next[0] = b_left;
    C_next[nx - 1] = b_right;

    // Parallelize interior points
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < nx - 1; i++) {
        double diffusion = r * (C[i + 1] - 2.0 * C[i] + C[i - 1]);
        double clearance = -k_dt * C[i];
        double val = C[i] + diffusion + clearance;

        // Enforce non-negativity
        if (val < 0.0) val = 0.0;

        C_next[i] = val;
    }
}

/**
 * Run full explicit FDM simulation with OpenMP time stepping.
 *
 * @param C_out      Output concentration field [nx * nt], row-major
 * @param nx         Number of spatial grid points
 * @param nt         Number of time steps
 * @param r          Fourier number
 * @param k_dt       Clearance * dt
 * @param b_left     Left boundary value
 * @param b_right    Right boundary value
 * @param C_init     Initial concentration [nx]
 * @param n_threads  Number of OpenMP threads (0 = auto)
 */
EXPORT void explicit_fdm_solve_omp(
    double* C_out,
    int nx,
    int nt,
    double r,
    double k_dt,
    double b_left,
    double b_right,
    const double* C_init,
    int n_threads
) {
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }

    // Copy initial condition
    for (int i = 0; i < nx; i++) {
        C_out[i] = C_init[i];
    }

    // Temporary buffer for next time step
    double* C_next = (double*)malloc(nx * sizeof(double));
    if (!C_next) {
        fprintf(stderr, "Memory allocation failed in explicit_fdm_solve_omp\n");
        return;
    }

    for (int n = 0; n < nt - 1; n++) {
        const double* C_current = &C_out[n * nx];
        double* C_write = &C_out[(n + 1) * nx];

        // Apply BCs
        C_write[0] = b_left;
        C_write[nx - 1] = b_right;

        // Parallel interior update
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < nx - 1; i++) {
            double diffusion = r * (C_current[i + 1] - 2.0 * C_current[i] + C_current[i - 1]);
            double clearance = -k_dt * C_current[i];
            double val = C_current[i] + diffusion + clearance;

            if (val < 0.0) val = 0.0;
            C_write[i] = val;
        }
    }

    free(C_next);
}

/**
 * Get available OpenMP thread count.
 */
EXPORT int omp_get_available_threads(void) {
    return omp_get_max_threads();
}

} // extern "C"
