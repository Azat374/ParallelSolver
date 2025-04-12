#include "spmv_kernel.h"
#include "device_launch_parameters.h"

__global__ void SpMVKernelCSR(int N, const double* values, const int* rowPtr, const int* colIdx, const double* x, double* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        double sum = 0.0;
        int start = rowPtr[row];
        int end = rowPtr[row + 1];
        for (int k = start; k < end; k++) {
            sum += values[k] * x[colIdx[k]];
        }
        y[row] = sum;
    }
}
