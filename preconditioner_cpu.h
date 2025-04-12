#ifndef PRECONDITIONER_CPU_H
#define PRECONDITIONER_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

	// ¬ыполнение ILU(0)-факторизации дл€ плотной матрицы A (NxN), in-place.
	void ILU0_CPU(int N, double* A);

	// –ешение системы L * y = b (forward solve). L хранитс€ в A (нижн€€ часть, с единичной диагональю).
	void forwardSolve(int N, const double* A, const double* b, double* y);

	// –ешение системы U * x = y (backward solve). U хранитс€ в A (диагональ и выше).
	void backwardSolve(int N, const double* A, const double* y, double* x);

#ifdef __cplusplus
}
#endif

#endif // PRECONDITIONER_CPU_H
