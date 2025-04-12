#ifndef PRECONDITIONER_MPI_H
#define PRECONDITIONER_MPI_H

#ifdef __cplusplus
extern "C" {
#endif

	// ILU(0) предобусловливание с использованием MPI для распределённой матрицы A (NxN).
	// На 0-м процессе A – полная матрица; затем производится разбивка по строкам.
	void ILU0_MPI(int N, double* A);

#ifdef __cplusplus
}
#endif

#endif // PRECONDITIONER_MPI_H
