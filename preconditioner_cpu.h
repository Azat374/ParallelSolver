#ifndef PRECONDITIONER_CPU_H
#define PRECONDITIONER_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

	// ���������� ILU(0)-������������ ��� ������� ������� A (NxN), in-place.
	void ILU0_CPU(int N, double* A);

	// ������� ������� L * y = b (forward solve). L �������� � A (������ �����, � ��������� ����������).
	void forwardSolve(int N, const double* A, const double* b, double* y);

	// ������� ������� U * x = y (backward solve). U �������� � A (��������� � ����).
	void backwardSolve(int N, const double* A, const double* y, double* x);

#ifdef __cplusplus
}
#endif

#endif // PRECONDITIONER_CPU_H
