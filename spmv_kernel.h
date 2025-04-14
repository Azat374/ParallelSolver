#ifndef SPMV_KERNEL_H
#define SPMV_KERNEL_H

#include <cuda_runtime.h>

//  ернел дл€ умножени€ матрицы в формате CSR на вектор: y = A*x.
__global__ void SpMVKernelCSR(int N, const double* values, const int* rowPtr, const int* colIdx, const double* x, double* y);


#endif // SPMV_KERNEL_H
