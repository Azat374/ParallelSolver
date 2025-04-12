#ifndef PRECONDITIONER_CUDA_H
#define PRECONDITIONER_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

	// ILU(0) предобусловливание на GPU для плотной матрицы d_A (NxN) в памяти устройства, in-place.
	void ILU0_GPU(int N, double* d_A);

	// Последовательное forward/backward решение на GPU (для демонстрации; в production рекомендуется cuSPARSE)
	__global__ void forwardSolveKernel(const double* A, const double* b, double* y, int N);
	__global__ void backwardSolveKernel(const double* A, const double* y, double* x, int N);

#ifdef __cplusplus
}
#endif

#endif // PRECONDITIONER_CUDA_H
