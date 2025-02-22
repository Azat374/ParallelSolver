#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

// Наивная реализация ILU(0) на GPU для плотной матрицы A (row-major).
// Для демонстрационных целей реализовано последовательное выполнение на GPU (один поток).
__global__ void ILU0Kernel(int N, double* A) {
    for (int k = 0; k < N; k++) {
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

// Обёртка для запуска ILU0Kernel на GPU
extern "C" void ILU0_GPU(int N, double* d_A) {
    // Запускаем ядро с одним блоком и одним потоком (последовательное выполнение)
    ILU0Kernel << <1, 1 >> > (N, d_A);
    cudaDeviceSynchronize();
}
