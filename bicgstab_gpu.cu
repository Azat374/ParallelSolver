#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "preconditioner_cuda.h"
#include "spmv_kernel.h"
#include <mpi.h>
#include "device_launch_parameters.h"

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if(err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__\
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

const double eps = 1e-12; // защита от деления на ноль

// Кернел для плотного матрично-векторного умножения: y = A * x
__global__ void denseMatVec(const double* A, const double* x, double* y, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

// Кернел для векторного вычитания: r = b - v
__global__ void vecSub(double* r, const double* b, const double* v, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        r[i] = b[i] - v[i];
    }
}

// Кернел для обновления вектора: p = r + beta*(p - omega*v)
__global__ void updateP(double* p, const double* r, const double* v, double beta, double omega, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
    }
}

// Кернел для вычисления s: s = r - alpha*v
__global__ void computeS(double* s, const double* r, const double* v, double alpha, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        s[i] = r[i] - alpha * v[i];
    }
}

// Кернел для обновления x: x = x + alpha*p
__global__ void updateX(double* x, const double* p, double alpha, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] += alpha * p[i];
    }
}

// Кернел для полного обновления x: x = x + alpha*p + omega*s
__global__ void updateXFull(double* x, const double* p, const double* s, double alpha, double omega, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] += alpha * p[i] + omega * s[i];
    }
}

// Кернел для обновления r: r = s - omega*t
__global__ void updateR(double* r, const double* s, const double* t, double omega, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        r[i] = s[i] - omega * t[i];
    }
}

// Функция для вычисления скалярного произведения на хосте (для диагностики)
double dotProduct(const double* h_a, const double* h_b, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += h_a[i] * h_b[i];
    return sum;
}

// Реализация метода BiCGStab(2) на GPU с защитными проверками.
// ILU предобусловливание отключено для диагностики (раскомментируйте ILU0_GPU, если потребуется).
extern "C" void BiCGStab2_GPU(const double* A, double* x, const double* b, int N, double tol, int maxIter, int* iterCount)
{
    // Выбор GPU на основе MPI-ранга
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    double* d_A, * d_x, * d_b, * d_r, * d_rhat, * d_p, * d_v, * d_s, * d_t;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_r, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_rhat, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_p, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_v, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_s, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_t, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

    // Если ILU предобусловливание нужно, раскомментируйте следующую строку:
    // ILU0_GPU(N, d_A);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Вычисляем начальный остаток: r = b - A*x
    denseMatVec << <gridSize, blockSize >> > (d_A, d_x, d_v, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    vecSub << <gridSize, blockSize >> > (d_r, d_b, d_v, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Фиксируем начальный остаток: rhat = r
    CUDA_CHECK(cudaMemcpy(d_rhat, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));

    // Инициализируем p как копию r
    CUDA_CHECK(cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));

    double alpha = 1.0, omega = 1.0, rho_old = 1.0;
    int iter = 0;
    double normb = 0.0;
    double* h_b_host = new double[N];
    CUDA_CHECK(cudaMemcpy(h_b_host, d_b, N * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
        normb += h_b_host[i] * h_b_host[i];
    normb = sqrt(normb);
    if (normb < eps) normb = 1.0;
    delete[] h_b_host;

    while (iter < maxIter) {
        // Вычисляем ρ = <rhat, r>
        double rho = 0.0;
        double* h_r = new double[N];
        double* h_rhat = new double[N];
        CUDA_CHECK(cudaMemcpy(h_r, d_r, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rhat, d_rhat, N * sizeof(double), cudaMemcpyDeviceToHost));
        rho = dotProduct(h_rhat, h_r, N);
        delete[] h_r;
        delete[] h_rhat;
        if (fabs(rho) < tol) break;
        double beta = (iter == 0) ? 0.0 : (rho / (rho_old + eps)) * (alpha / (omega + eps));

        updateP << <gridSize, blockSize >> > (d_p, d_r, d_v, beta, omega, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // v = A * p
        denseMatVec << <gridSize, blockSize >> > (d_A, d_p, d_v, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Вычисляем <rhat, v>
        double rhat_dot_v = 0.0;
        double* h_rhat2 = new double[N];
        double* h_v = new double[N];
        CUDA_CHECK(cudaMemcpy(h_rhat2, d_rhat, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v, d_v, N * sizeof(double), cudaMemcpyDeviceToHost));
        rhat_dot_v = dotProduct(h_rhat2, h_v, N);
        delete[] h_rhat2;
        delete[] h_v;
        if (fabs(rhat_dot_v) < tol) break;
        alpha = rho / (rhat_dot_v + eps);

        computeS << <gridSize, blockSize >> > (d_s, d_r, d_v, alpha, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        double norm_s = 0.0;
        double* h_s = new double[N];
        CUDA_CHECK(cudaMemcpy(h_s, d_s, N * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++)
            norm_s += h_s[i] * h_s[i];
        norm_s = sqrt(norm_s);
        delete[] h_s;
        if (norm_s / normb < tol) {
            updateX << <gridSize, blockSize >> > (d_x, d_p, alpha, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            iter++;
            break;
        }

        // t = A * s
        denseMatVec << <gridSize, blockSize >> > (d_A, d_s, d_t, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        double t_dot_s = 0.0, t_dot_t = 0.0;
        double* h_t = new double[N];
        double* h_s2 = new double[N];
        CUDA_CHECK(cudaMemcpy(h_t, d_t, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_s2, d_s, N * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) {
            t_dot_s += h_t[i] * h_s2[i];
            t_dot_t += h_t[i] * h_t[i];
        }
        delete[] h_t;
        delete[] h_s2;
        if (fabs(t_dot_t) < tol) break;
        omega = t_dot_s / (t_dot_t + eps);

        updateXFull << <gridSize, blockSize >> > (d_x, d_p, d_s, alpha, omega, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        updateR << <gridSize, blockSize >> > (d_r, d_s, d_t, omega, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        double norm_r = 0.0;
        double* h_r_new = new double[N];
        CUDA_CHECK(cudaMemcpy(h_r_new, d_r, N * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++)
            norm_r += h_r_new[i] * h_r_new[i];
        norm_r = sqrt(norm_r);
        delete[] h_r_new;
        if (norm_r / normb < tol) break;

        rho_old = rho;
        iter++;
    }

    if (iterCount) {
        *iterCount = iter;
    }

    CUDA_CHECK(cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_rhat));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_t));
}
