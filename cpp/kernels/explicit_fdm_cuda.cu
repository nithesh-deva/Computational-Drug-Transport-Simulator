/**
 * M9 Performance Kernel: Explicit FDM with CUDA
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
 * Compile with CUDA:
 *   nvcc -O3 -arch=sm_70 --shared -o cdts_explicit_cuda.dll explicit_fdm_cuda.cu
 *   nvcc -O3 -arch=sm_70 --shared -o libcdts_explicit_cuda.so explicit_fdm_cuda.cu (Linux)
 *
 * Note: Adjust -arch based on GPU capability (sm_60, sm_70, sm_80, sm_90, etc.)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * CUDA kernel: single time step update for interior points.
 *
 * Grid-stride loop: each thread processes multiple points to handle various grid sizes.
 * Boundary conditions are applied in host code.
 *
 * @param C_current    Input concentration array [nx]
 * @param C_next       Output concentration array [nx]
 * @param nx           Number of spatial grid points
 * @param r            Fourier number = D * dt / dx^2
 * @param k_dt         Clearance * dt [dimensionless]
 * @param b_left       Left boundary value (Dirichlet)
 * @param b_right      Right boundary value (Dirichlet)
 */
__global__ void explicit_fdm_step_kernel(
    const double* C_current,
    double* C_next,
    int nx,
    double r,
    double k_dt,
    double b_left,
    double b_right
) {
    // Grid-stride loop: handle all interior points [1, nx-2]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Boundary handling on device
    if (idx == 0) {
        C_next[0] = b_left;
    }
    if (idx == 1) {
        C_next[nx - 1] = b_right;
    }

    // Synchronize to ensure boundaries are set before interior update
    __syncthreads();

    // Interior points: [1, nx-2]
    for (int i = idx + 1; i < nx - 1; i += stride) {
        double diffusion = r * (C_current[i + 1] - 2.0 * C_current[i] + C_current[i - 1]);
        double clearance = -k_dt * C_current[i];
        double val = C_current[i] + diffusion + clearance;

        // Enforce non-negativity
        if (val < 0.0) val = 0.0;

        C_next[i] = val;
    }
}

/**
 * Host wrapper: perform one CUDA time step.
 *
 * Allocates device memory, copies data, launches kernel, and returns result to host.
 *
 * @param C_current    Input concentration array [nx] on host
 * @param C_next       Output concentration array [nx] on host
 * @param nx           Number of spatial grid points
 * @param r            Fourier number
 * @param k_dt         Clearance * dt
 * @param b_left       Left boundary value
 * @param b_right      Right boundary value
 *
 * @return 0 on success, -1 on CUDA error
 */
int cuda_explicit_fdm_step(
    const double* C_current,
    double* C_next,
    int nx,
    double r,
    double k_dt,
    double b_left,
    double b_right
) {
    double *d_C_current = nullptr, *d_C_next = nullptr;
    size_t bytes = nx * sizeof(double);

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_C_current, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for C_current: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&d_C_next, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for C_next: %s\n", cudaGetErrorString(err));
        cudaFree(d_C_current);
        return -1;
    }

    // Copy input to device
    err = cudaMemcpy(d_C_current, C_current, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_C_current);
        cudaFree(d_C_next);
        return -1;
    }

    // Configure grid and block
    int block_size = 256;  // Typical warp-multiple block size
    int num_blocks = (nx + block_size - 1) / block_size;

    // Launch kernel
    explicit_fdm_step_kernel<<<num_blocks, block_size>>>(
        d_C_current, d_C_next, nx, r, k_dt, b_left, b_right
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_C_current);
        cudaFree(d_C_next);
        return -1;
    }

    // Copy output to host
    err = cudaMemcpy(C_next, d_C_next, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_C_current);
        cudaFree(d_C_next);
        return -1;
    }

    // Cleanup
    cudaFree(d_C_current);
    cudaFree(d_C_next);

    return 0;
}

/**
 * Run full explicit FDM simulation with CUDA.
 *
 * Persistent memory approach: allocate device arrays once, perform time stepping.
 * This minimizes H2D/D2H transfers for bulk data.
 *
 * @param C_out        Output concentration field [nx * nt], row-major on host
 * @param nx           Number of spatial grid points
 * @param nt           Number of time steps
 * @param r            Fourier number
 * @param k_dt         Clearance * dt
 * @param b_left       Left boundary value
 * @param b_right      Right boundary value
 * @param C_init       Initial concentration [nx] on host
 *
 * @return 0 on success, -1 on CUDA error
 */
EXPORT int explicit_fdm_solve_cuda(
    double* C_out,
    int nx,
    int nt,
    double r,
    double k_dt,
    double b_left,
    double b_right,
    const double* C_init
) {
    size_t bytes_grid = nx * nt * sizeof(double);
    size_t bytes_slice = nx * sizeof(double);

    double *d_C_out = nullptr, *d_C_next = nullptr;

    // Allocate device arrays
    cudaError_t err = cudaMalloc(&d_C_out, bytes_grid);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C_out: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&d_C_next, bytes_slice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C_next: %s\n", cudaGetErrorString(err));
        cudaFree(d_C_out);
        return -1;
    }

    // Copy initial condition to first row of device array
    err = cudaMemcpy(d_C_out, C_init, bytes_slice, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy initial condition failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_C_out);
        cudaFree(d_C_next);
        return -1;
    }

    // Configure kernel execution
    int block_size = 256;
    int num_blocks = (nx + block_size - 1) / block_size;

    // Time stepping loop
    for (int n = 0; n < nt - 1; n++) {
        const double* d_C_current = &d_C_out[n * nx];
        double* d_C_write = &d_C_out[(n + 1) * nx];

        // Launch kernel
        explicit_fdm_step_kernel<<<num_blocks, block_size>>>(
            d_C_current, d_C_write, nx, r, k_dt, b_left, b_right
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed at step %d: %s\n", n, cudaGetErrorString(err));
            cudaFree(d_C_out);
            cudaFree(d_C_next);
            return -1;
        }
    }

    // Copy result back to host
    err = cudaMemcpy(C_out, d_C_out, bytes_grid, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy result back to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_C_out);
        cudaFree(d_C_next);
        return -1;
    }

    // Cleanup
    cudaFree(d_C_out);
    cudaFree(d_C_next);

    return 0;
}

/**
 * Query CUDA device information and availability.
 *
 * @param device_count  Pointer to store number of available CUDA devices
 * @param compute_cap_major  Pointer to store compute capability major version
 * @param compute_cap_minor  Pointer to store compute capability minor version
 *
 * @return 0 if CUDA device available, -1 otherwise
 */
EXPORT int cuda_get_device_info(
    int* device_count,
    int* compute_cap_major,
    int* compute_cap_minor
) {
    cudaError_t err = cudaGetDeviceCount(device_count);
    if (err != cudaSuccess || *device_count <= 0) {
        *device_count = 0;
        *compute_cap_major = 0;
        *compute_cap_minor = 0;
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return -1;
    }

    *compute_cap_major = prop.major;
    *compute_cap_minor = prop.minor;
    return 0;
}

/**
 * Reset CUDA device to ensure clean state.
 */
EXPORT void cuda_reset_device(void) {
    cudaDeviceReset();
}

} // extern "C"
