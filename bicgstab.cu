#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <vector>

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

//----------------------------------------
// CUDA-ядра для операций над векторами
//----------------------------------------
__global__ void csrMatVecMulKernel(int N, const double* values, const int* rowPtr,
    const int* colIdx, const double* x, double* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        double sum = 0.0;
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];
        for (int j = rowStart; j < rowEnd; j++) {
            sum += values[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}

__global__ void axpyKernel(int N, double a, const double* x, double* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] += a * x[idx];
    }
}

__global__ void copyKernel(int N, const double* src, double* dest) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dest[idx] = src[idx];
    }
}

__global__ void scaleKernel(int N, double a, double* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] *= a;
    }
}

__global__ void subtractKernel(int N, const double* x, const double* y, double* z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        z[idx] = x[idx] - y[idx];
    }
}

__global__ void initZeroKernel(int N, double* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = 0.0;
    }
}

// Редукция для dot product через атомарное сложение
__global__ void dotProductKernel(int N, const double* x, const double* y, double* result) {
    __shared__ double shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double temp = 0.0;
    while (idx < N) {
        temp += x[idx] * y[idx];
        idx += gridDim.x * blockDim.x;
    }
    shared[tid] = temp;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    double oldVal;
    do {
        assumed = old;
        oldVal = __longlong_as_double(assumed);
        double newVal = oldVal + val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(newVal));
    } while (assumed != old);
    return oldVal;
}


// Вспомогательная функция для вычисления dot product на GPU
double gpuDotProduct(int N, const double* d_x, const double* d_y) {
    double* d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dotProductKernel << <gridSize, blockSize >> > (N, d_x, d_y, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    double h_result = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    return h_result;
}

//----------------------------------------
// Реализация BiCGStab(2) для разреженной матрицы в формате CSR на GPU
//----------------------------------------
extern "C" void BiCGStab2_GPU(int N, const std::vector<double>&csr_values,
    const std::vector<int>&csr_rowPtr, const std::vector<int>&csr_colIdx,
    double* x, const double* b, double tol, int maxIter, int* iterCount) {
    int nnz = csr_values.size();
    // Выделяем память на устройстве
    double* d_values, * d_x, * d_r, * d_r_hat, * d_p, * d_v, * d_s, * d_t, * d_tmp;
    int* d_rowPtr, * d_colIdx;
    CUDA_CHECK(cudaMalloc((void**)&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_rowPtr, csr_rowPtr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_colIdx, csr_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_r, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_r_hat, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_p, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_v, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_s, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_t, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_tmp, N * sizeof(double)));

    // Копируем данные с хоста на устройство
    CUDA_CHECK(cudaMemcpy(d_values, csr_values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, csr_rowPtr.data(), csr_rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, csr_colIdx.data(), csr_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
    // Инициализируем r = b
    CUDA_CHECK(cudaMemcpy(d_r, b, N * sizeof(double), cudaMemcpyHostToDevice));

    // r_hat = r
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    copyKernel << <gridSize, blockSize >> > (N, d_r, d_r_hat);
    CUDA_CHECK(cudaDeviceSynchronize());

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    int iter = 0;
    double norm_r = sqrt(gpuDotProduct(N, d_r, d_r));

    while (iter < maxIter && norm_r > tol) {
        for (int j = 1; j <= 2; j++) {
            double rho = gpuDotProduct(N, d_r_hat, d_r);
            if (fabs(rho) < tol) {
                std::cerr << "GPU Breakdown: rho near zero at iteration " << iter << std::endl;
                *iterCount = iter;
                goto cleanup;
            }
            double beta = (j == 1) ? 0.0 : (rho / rho_old) * (alpha / omega);
            if (j == 1) {
                // p = r
                copyKernel << <gridSize, blockSize >> > (N, d_r, d_p);
            }
            else {
                // p = r + beta*(p - omega*v)
                // p = p - omega*v
                axpyKernel << <gridSize, blockSize >> > (N, -omega, d_v, d_p);
                // p = r + beta*p
                axpyKernel << <gridSize, blockSize >> > (N, beta, d_p, d_r); // reuse d_r as temporary
                copyKernel << <gridSize, blockSize >> > (N, d_r, d_p);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // v = A * p (CSR матрично-векторное умножение)
            csrMatVecMulKernel << <gridSize, blockSize >> > (N, d_values, d_rowPtr, d_colIdx, d_p, d_v);
            CUDA_CHECK(cudaDeviceSynchronize());

            double rhat_dot_v = gpuDotProduct(N, d_r_hat, d_v);
            if (fabs(rhat_dot_v) < tol) {
                std::cerr << "GPU Breakdown: rhat_dot_v near zero at iteration " << iter << std::endl;
                *iterCount = iter;
                goto cleanup;
            }
            alpha = rho / rhat_dot_v;

            // s = r - alpha*v
            copyKernel << <gridSize, blockSize >> > (N, d_v, d_tmp);
            scaleKernel << <gridSize, blockSize >> > (N, alpha, d_tmp);
            subtractKernel << <gridSize, blockSize >> > (N, d_r, d_tmp, d_s);
            CUDA_CHECK(cudaDeviceSynchronize());

            double norm_s = sqrt(gpuDotProduct(N, d_s, d_s));
            if (norm_s < tol) {
                axpyKernel << <gridSize, blockSize >> > (N, alpha, d_p, d_x);
                CUDA_CHECK(cudaDeviceSynchronize());
                *iterCount = iter;
                goto cleanup;
            }

            // t = A * s
            csrMatVecMulKernel << <gridSize, blockSize >> > (N, d_values, d_rowPtr, d_colIdx, d_s, d_t);
            CUDA_CHECK(cudaDeviceSynchronize());

            double t_dot_t = gpuDotProduct(N, d_t, d_t);
            if (fabs(t_dot_t) < tol) {
                std::cerr << "GPU Breakdown: t_dot_t near zero at iteration " << iter << std::endl;
                *iterCount = iter;
                goto cleanup;
            }
            double t_dot_s = gpuDotProduct(N, d_t, d_s);
            omega = t_dot_s / t_dot_t;

            // Обновление решения: x = x + alpha*p + omega*s
            axpyKernel << <gridSize, blockSize >> > (N, alpha, d_p, d_x);
            axpyKernel << <gridSize, blockSize >> > (N, omega, d_s, d_x);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Обновление остатка: r = s - omega*t
            copyKernel << <gridSize, blockSize >> > (N, d_t, d_tmp);
            scaleKernel << <gridSize, blockSize >> > (N, omega, d_tmp);
            subtractKernel << <gridSize, blockSize >> > (N, d_s, d_tmp, d_r);
            CUDA_CHECK(cudaDeviceSynchronize());

            norm_r = sqrt(gpuDotProduct(N, d_r, d_r));
            if (norm_r < tol) break;
            rho_old = rho;
        }
        iter++;
    }
cleanup:
    *iterCount = iter;
    CUDA_CHECK(cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << "GPU BiCGStab(2) iterations: " << iter << std::endl;

    cudaFree(d_values);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_r_hat);
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_t);
    cudaFree(d_tmp);
}
