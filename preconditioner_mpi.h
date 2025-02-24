#ifndef PRECONDITIONER_MPI_H
#define PRECONDITIONER_MPI_H

// Применение ILU(0) предобусловливания с использованием MPI.
// A – матрица системы (размер N x N), распределённая между процессами.
extern "C" void ILU0_MPI(int N, double* A);

#endif // PRECONDITIONER_MPI_H
