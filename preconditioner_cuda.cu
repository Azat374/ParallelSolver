
#include "device_launch_parameters.h"
#include "preconditioner_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Кернел для ILU(0) факторизации по строкам.
// (Упрощённая реализация – для полноценной параллельной ILU требуются более сложные алгоритмы)
__global__ void ilu0_kernel(double* A, int N) {
    int k = blockIdx.x; // каждый блок обрабатывает одну строку
    if (k < N) {
        double diag = A[k * N + k];
        if (fabs(diag) < 1e-12) {
            diag = 1e-12;
        }
        for (int i = k + 1; i < N; i++) {
            A[i * N + k] /= diag;
        }
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i * N + j] -= A[i * N + k] * A[k * N + j];
            }
        }
    }
}

extern "C" void ILU0_GPU(int N, double* d_A)
{
    // Запуск N блоков по 1 потоку
    ilu0_kernel << <N, 1 >> > (d_A, N);
    cudaDeviceSynchronize();
}
