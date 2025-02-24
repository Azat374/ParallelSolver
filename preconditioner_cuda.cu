
#include "device_launch_parameters.h"
#include "preconditioner_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// ������ ��� ILU(0) ������������ �� �������.
// (���������� ���������� � ��� ����������� ������������ ILU ��������� ����� ������� ���������)
__global__ void ilu0_kernel(double* A, int N) {
    int k = blockIdx.x; // ������ ���� ������������ ���� ������
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
    // ������ N ������ �� 1 ������
    ilu0_kernel << <N, 1 >> > (d_A, N);
    cudaDeviceSynchronize();
}
