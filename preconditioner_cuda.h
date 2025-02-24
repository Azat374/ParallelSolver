#ifndef PRECONDITIONER_CUDA_H
#define PRECONDITIONER_CUDA_H

// ���������� ILU(0) ������������������ �� GPU.
// d_A � ��������� �� ������� �� ���������� (������ N x N).
extern "C" void ILU0_GPU(int N, double* d_A);

#endif // PRECONDITIONER_CUDA_H
