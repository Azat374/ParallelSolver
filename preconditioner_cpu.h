#ifndef PRECONDITIONER_CPU_H
#define PRECONDITIONER_CPU_H

// ���������� ILU(0) ������������������ �� CPU.
// A � ������� ������� (������ N x N), ������� �������������� in-place.
extern "C" void ILU0_CPU(int N, double* A);

#endif // PRECONDITIONER_CPU_H
