#ifndef PRECONDITIONER_MPI_H
#define PRECONDITIONER_MPI_H

// ���������� ILU(0) ������������������ � �������������� MPI.
// A � ������� ������� (������ N x N), ������������� ����� ����������.
extern "C" void ILU0_MPI(int N, double* A);

#endif // PRECONDITIONER_MPI_H
