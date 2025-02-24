#include "spmv_kernel.h"
#include "device_launch_parameters.h"

__global__ void SpMVKernel(int N, const double* values, const int* rowPtr, const int* colIdx, const double* x, double* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        double sum = 0.0;
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}
