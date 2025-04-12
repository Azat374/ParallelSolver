#include "preconditioner_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include "device_launch_parameters.h"

//  аждый блок обрабатывает одну строку.
__global__ void ilu0_kernel(double* A, int N) {
    int k = blockIdx.x;
    if (k < N) {
        double diag = A[k * N + k];
        if (fabs(diag) < 1e-12)
            diag = 1e-12;
        for (int i = k + 1; i < N; i++) {
            A[i * N + k] /= diag;
        }
        for (int i = k + 1; i < N; i++) {
            double lik = A[i * N + k];
            for (int j = k + 1; j < N; j++) {
                A[i * N + j] -= lik * A[k * N + j];
            }
        }
    }
}

extern "C" void ILU0_GPU(int N, double* d_A) {
    dim3 grid(N), block(1);
    ilu0_kernel << <grid, block >> > (d_A, N);
    cudaDeviceSynchronize();
}

__global__ void forwardSolveKernel(const double* A, const double* b, double* y, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < N; i++) {
            double sum = b[i];
            for (int j = 0; j < i; j++) {
                sum -= A[i * N + j] * y[j];
            }
            y[i] = sum;
        }
    }
}

__global__ void backwardSolveKernel(const double* A, const double* y, double* x, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = N - 1; i >= 0; i--) {
            double sum = y[i];
            double diag = A[i * N + i];
            if (fabs(diag) < 1e-12)
                diag = 1e-12;
            for (int j = i + 1; j < N; j++) {
                sum -= A[i * N + j] * x[j];
            }
            x[i] = sum / diag;
        }
    }
}
