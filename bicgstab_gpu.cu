#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include "preconditioner_cuda.h"
#include "spmv_kernel.h"
#include "device_launch_parameters.h"
#include "utils.h"  // для convertDenseToCSR

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if(err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__\
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

const double eps = 1e-12;

// --- Таймер на GPU ---
void gpuTimerStart(cudaEvent_t* start) {
    cudaEventCreate(start);
    cudaEventRecord(*start, 0);
}

float gpuTimerStop(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed;
}

// --- Определения необходимых кернелов ---
// Обновление вектора: p = r + beta*(p - omega*v)
__global__ void updateP(double* p, const double* r, const double* v, double beta, double omega, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
}

// Вычисление s: s = r - alpha*v
__global__ void computeS(double* s, const double* r, const double* v, double alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        s[i] = r[i] - alpha * v[i];
}

// Обновление x: x = x + alpha*p
__global__ void updateX(double* x, const double* p, double alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        x[i] += alpha * p[i];
}

// Обновление x полностью: x = x + alpha*p + omega*s
__global__ void updateXFull(double* x, const double* p, const double* s, double alpha, double omega, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        x[i] += alpha * p[i] + omega * s[i];
}

// Обновление r: r = s - omega*t
__global__ void updateR(double* r, const double* s, const double* t, double omega, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        r[i] = s[i] - omega * t[i];
}

extern "C" void BiCGStab2_GPU(const double* A, double* x, const double* b,
    int N, double tol, int maxIter, int* iterCount)
{
    // Переводим плотную матрицу в CSR
    double* h_values = nullptr;
    int* h_rowPtr = nullptr;
    int* h_colIdx = nullptr;
    int nnz = 0;
    convertDenseToCSR(A, N, &h_values, &nnz, &h_rowPtr, &h_colIdx);

    // Аллокация CSR на GPU
    double* d_values; int* d_rowPtr; int* d_colIdx;
    CUDA_CHECK(cudaMalloc((void**)&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_rowPtr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_colIdx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    delete[] h_values; delete[] h_rowPtr; delete[] h_colIdx;

    // Аллокация векторов на GPU
    double* d_x, * d_b, * d_r, * d_rhat, * d_p, * d_v, * d_s, * d_t, * d_z;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_r, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_rhat, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_p, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_v, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_s, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_t, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_z, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Замер времени на GPU
    cudaEvent_t startEvent;
    gpuTimerStart(&startEvent);

    // r = b - A*x. Для SpMV используем наш CSR кернел.
    double* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(double)));
    SpMVKernelCSR << <gridSize, blockSize >> > (N, d_values, d_rowPtr, d_colIdx, d_x, d_temp);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Вычисляем r = b - y (в данном варианте выполняем передачу на хост, но для производительности нужно убрать лишние cudaMemcpy)
    {
        double* h_temp = new double[N];
        double* h_b = new double[N];
        CUDA_CHECK(cudaMemcpy(h_temp, d_temp, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b, d_b, N * sizeof(double), cudaMemcpyDeviceToHost));
        double* h_r = new double[N];
        for (int i = 0; i < N; i++)
            h_r[i] = h_b[i] - h_temp[i];
        CUDA_CHECK(cudaMemcpy(d_r, h_r, N * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_temp; delete[] h_b; delete[] h_r;
    }
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaMemcpy(d_rhat, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0, omega = 1.0, rho_old = 1.0;
    int iter = 0;
    double normb = 0.0;
    cublasDnrm2(handle, N, d_b, 1, &normb);
    if (normb < eps) normb = 1.0;

    while (iter < maxIter) {
        double rho = 0.0;
        cublasDdot(handle, N, d_rhat, 1, d_r, 1, &rho);
        if (fabs(rho) < tol) break;
        double beta = (iter == 0) ? 0.0 : (rho / (rho_old + eps)) * (alpha / (omega + eps));

        updateP << <gridSize, blockSize >> > (d_p, d_r, d_v, beta, omega, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        SpMVKernelCSR << <gridSize, blockSize >> > (N, d_values, d_rowPtr, d_colIdx, d_p, d_v);
        CUDA_CHECK(cudaDeviceSynchronize());

        double rhat_dot_v = 0.0;
        cublasDdot(handle, N, d_rhat, 1, d_v, 1, &rhat_dot_v);
        if (fabs(rhat_dot_v) < tol) break;
        alpha = rho / (rhat_dot_v + eps);

        computeS << <gridSize, blockSize >> > (d_s, d_r, d_v, alpha, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        double norm_s = 0.0;
        cublasDnrm2(handle, N, d_s, 1, &norm_s);
        if (norm_s / normb < tol) {
            updateX << <gridSize, blockSize >> > (d_x, d_p, alpha, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            iter++;
            break;
        }

        SpMVKernelCSR << <gridSize, blockSize >> > (N, d_values, d_rowPtr, d_colIdx, d_s, d_t);
        CUDA_CHECK(cudaDeviceSynchronize());

        double t_dot_s = 0.0, t_dot_t = 0.0;
        cublasDdot(handle, N, d_t, 1, d_s, 1, &t_dot_s);
        double temp;
        cublasDnrm2(handle, N, d_t, 1, &temp);
        t_dot_t = temp * temp;
        if (fabs(t_dot_t) < tol) break;
        omega = t_dot_s / (t_dot_t + eps);

        updateXFull << <gridSize, blockSize >> > (d_x, d_p, d_s, alpha, omega, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        updateR << <gridSize, blockSize >> > (d_r, d_s, d_t, omega, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        double norm_r = 0.0;
        cublasDnrm2(handle, N, d_r, 1, &norm_r);
        if (norm_r / normb < tol)
            break;

        rho_old = rho;
        iter++;
    }
    float elapsed_ms = gpuTimerStop(startEvent, 0);
    std::cout << "GPU: Total time = " << elapsed_ms / 1000.0 << " s" << std::endl;
    if (iterCount)
        *iterCount = iter;

    CUDA_CHECK(cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_values); cudaFree(d_rowPtr); cudaFree(d_colIdx);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_rhat);
    cudaFree(d_p); cudaFree(d_v); cudaFree(d_s); cudaFree(d_t); cudaFree(d_z);

    cublasDestroy(handle);
}
