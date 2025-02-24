#ifndef PRECONDITIONER_CUDA_H
#define PRECONDITIONER_CUDA_H

// Применение ILU(0) предобусловливания на GPU.
// d_A – указатель на матрицу на устройстве (размер N x N).
extern "C" void ILU0_GPU(int N, double* d_A);

#endif // PRECONDITIONER_CUDA_H
