#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

// ������� ���������� ILU(0) �� GPU ��� ������� ������� A (row-major).
// ��� ���������������� ����� ����������� ���������������� ���������� �� GPU (���� �����).
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

// ������ ��� ������� ILU0Kernel �� GPU
extern "C" void ILU0_GPU(int N, double* d_A) {
    // ��������� ���� � ����� ������ � ����� ������� (���������������� ����������)
    ILU0Kernel << <1, 1 >> > (N, d_A);
    cudaDeviceSynchronize();
}
