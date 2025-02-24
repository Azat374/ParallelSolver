#ifndef SPMV_KERNEL_H
#define SPMV_KERNEL_H

#include <cuda_runtime.h>

// ������ ��� ������������ ��������-���������� ��������� � ������� CSR.
// y = A*x, ��� A ������ ���������: values, rowPtr, colIdx.
__global__ void SpMVKernel(int N, const double* values, const int* rowPtr, const int* colIdx, const double* x, double* y);

#endif // SPMV_KERNEL_H
