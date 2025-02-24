#ifndef PRECONDITIONER_CPU_H
#define PRECONDITIONER_CPU_H

// Применение ILU(0) предобусловливания на CPU.
// A – матрица системы (размер N x N), которая модифицируется in-place.
extern "C" void ILU0_CPU(int N, double* A);

#endif // PRECONDITIONER_CPU_H
