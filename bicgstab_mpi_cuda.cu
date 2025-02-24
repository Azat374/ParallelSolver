#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "preconditioner_cuda.h"
#include "spmv_kernel.h"
#include "device_launch_parameters.h"

// Kernel: dense matrix-vector multiplication for local block: y = A*x
__global__ void denseMatVecLocal(const double* A, const double* x, double* y, int N, int local_N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < local_N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

// Kernel: vector subtraction for local block: r = b - v
__global__ void vecSubLocal(double* r, const double* b, const double* v, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        r[i] = b[i] - v[i];
}

// Kernel: update p: p = r + beta*(p - omega*v)
__global__ void updatePLocal(double* p, const double* r, const double* v, double beta, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
}

// Kernel: compute v = A * global_p for local block
__global__ void computeVLocal(const double* A, const double* global_p, double* v, int N, int local_N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < local_N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * global_p[j];
        }
        v[row] = sum;
    }
}

// Kernel: compute s = r - alpha*v
__global__ void computeSLocal(double* s, const double* r, const double* v, double alpha, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        s[i] = r[i] - alpha * v[i];
}

// Kernel: update x: x = x + alpha*p
__global__ void updateXLocal(double* x, const double* p, double alpha, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        x[i] += alpha * p[i];
}

// Kernel: update x fully: x = x + alpha*p + omega*s
__global__ void updateXFullLocal(double* x, const double* p, const double* s, double alpha, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        x[i] += alpha * p[i] + omega * s[i];
}

// Kernel: update r: r = s - omega*t
__global__ void updateRLocal(double* r, const double* s, const double* t, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        r[i] = s[i] - omega * t[i];
}

// Kernel: compute t = A * s for local block
__global__ void computeTLocal(const double* A, const double* s, double* t, int N, int local_N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < local_N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * s[j];
        }
        t[row] = sum;
    }
}

extern "C" void BiCGStab2_MPI_CUDA(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_N = N / size;
    double* local_A = new double[local_N * N];
    double* local_b = new double[local_N];
    double* local_x = new double[local_N];
    memset(local_x, 0, local_N * sizeof(double));

    MPI_Scatter(A, local_N * N, MPI_DOUBLE, local_A, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_N, MPI_DOUBLE, local_b, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* d_A, * d_x, * d_b, * d_r, * d_p, * d_v, * d_s, * d_t;
    cudaMalloc((void**)&d_A, local_N * N * sizeof(double));
    cudaMalloc((void**)&d_x, local_N * sizeof(double));
    cudaMalloc((void**)&d_b, local_N * sizeof(double));
    cudaMalloc((void**)&d_r, local_N * sizeof(double));
    cudaMalloc((void**)&d_p, local_N * sizeof(double));
    cudaMalloc((void**)&d_v, local_N * sizeof(double));
    cudaMalloc((void**)&d_s, local_N * sizeof(double));
    cudaMalloc((void**)&d_t, local_N * sizeof(double));

    cudaMemcpy(d_A, local_A, local_N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, local_x, local_N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, local_b, local_N * sizeof(double), cudaMemcpyHostToDevice);

    // Применяем ILU0 на GPU для локального блока
    ILU0_GPU(local_N, d_A);

    int blockSize = 256;
    int gridSize = (local_N + blockSize - 1) / blockSize;

    denseMatVecLocal << <gridSize, blockSize >> > (d_A, d_x, d_v, N, local_N);
    cudaDeviceSynchronize();
    vecSubLocal << <gridSize, blockSize >> > (d_r, d_b, d_v, local_N);
    cudaDeviceSynchronize();
    cudaMemcpy(d_p, d_r, local_N * sizeof(double), cudaMemcpyDeviceToDevice);

    double alpha = 1.0, omega = 1.0, rho_old = 1.0;
    int iter = 0;
    double local_normb = 0.0;
    double* h_b_local = new double[local_N];
    cudaMemcpy(h_b_local, d_b, local_N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < local_N; i++) {
        local_normb += h_b_local[i] * h_b_local[i];
    }
    double global_normb;
    MPI_Allreduce(&local_normb, &global_normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_normb = sqrt(global_normb);
    if (global_normb < 1e-10) global_normb = 1.0;
    delete[] h_b_local;

    while (iter < maxIter) {
        double local_rho = 0.0;
        double* h_r_local = new double[local_N];
        cudaMemcpy(h_r_local, d_r, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < local_N; i++) {
            local_rho += h_r_local[i] * h_r_local[i];
        }
        delete[] h_r_local;
        double rho;
        MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rho) < tol) break;
        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);
        updatePLocal << <gridSize, blockSize >> > (d_p, d_r, d_v, beta, omega, local_N);
        cudaDeviceSynchronize();

        double* global_p = new double[N];
        double* h_p_local = new double[local_N];
        cudaMemcpy(h_p_local, d_p, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        MPI_Allgather(h_p_local, local_N, MPI_DOUBLE, global_p, local_N, MPI_DOUBLE, MPI_COMM_WORLD);
        delete[] h_p_local;

        computeVLocal << <gridSize, blockSize >> > (d_A, global_p, d_v, N, local_N);
        cudaDeviceSynchronize();
        delete[] global_p;

        double local_rhat_dot_v = 0.0;
        double* h_rhat_local = new double[local_N];
        double* h_v_local = new double[local_N];
        cudaMemcpy(h_rhat_local, d_r, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v_local, d_v, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < local_N; i++) {
            local_rhat_dot_v += h_rhat_local[i] * h_v_local[i];
        }
        delete[] h_rhat_local; delete[] h_v_local;
        double rhat_dot_v;
        MPI_Allreduce(&local_rhat_dot_v, &rhat_dot_v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rhat_dot_v) < tol) break;
        alpha = rho / rhat_dot_v;
        computeSLocal << <gridSize, blockSize >> > (d_s, d_r, d_v, alpha, local_N);
        cudaDeviceSynchronize();

        double local_norm_s = 0.0;
        double* h_s_local = new double[local_N];
        cudaMemcpy(h_s_local, d_s, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < local_N; i++) {
            local_norm_s += h_s_local[i] * h_s_local[i];
        }
        double norm_s = sqrt(local_norm_s);
        delete[] h_s_local;
        if (norm_s / global_normb < tol) {
            updateXLocal << <gridSize, blockSize >> > (d_x, d_p, alpha, local_N);
            cudaDeviceSynchronize();
            iter++;
            break;
        }
        computeTLocal << <gridSize, blockSize >> > (d_A, d_s, d_t, N, local_N);
        cudaDeviceSynchronize();

        double local_t_dot_s = 0.0, local_t_dot_t = 0.0;
        double* h_t_local = new double[local_N];
        double* h_s2_local = new double[local_N];
        cudaMemcpy(h_t_local, d_t, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s2_local, d_s, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < local_N; i++) {
            local_t_dot_s += h_t_local[i] * h_s2_local[i];
            local_t_dot_t += h_t_local[i] * h_t_local[i];
        }
        delete[] h_t_local; delete[] h_s2_local;
        double t_dot_s, t_dot_t;
        MPI_Allreduce(&local_t_dot_s, &t_dot_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_t_dot_t, &t_dot_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(t_dot_t) < tol) break;
        omega = t_dot_s / t_dot_t;
        updateXFullLocal << <gridSize, blockSize >> > (d_x, d_p, d_s, alpha, omega, local_N);
        cudaDeviceSynchronize();

        updateRLocal << <gridSize, blockSize >> > (d_r, d_s, d_t, omega, local_N);
        cudaDeviceSynchronize();

        double local_norm_r = 0.0;
        double* h_r_new_local = new double[local_N];
        cudaMemcpy(h_r_new_local, d_r, local_N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < local_N; i++) {
            local_norm_r += h_r_new_local[i] * h_r_new_local[i];
        }
        double norm_r = sqrt(local_norm_r);
        delete[] h_r_new_local;
        if (norm_r / global_normb < tol) break;
        rho_old = rho;
        iter++;
    }

    *iterCount = iter;
    cudaMemcpy(local_x, d_x, local_N * sizeof(double), cudaMemcpyDeviceToHost);
    MPI_Gather(local_x, local_N, MPI_DOUBLE, x, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_t);

    delete[] local_A;
    delete[] local_b;
    delete[] local_x;
}
