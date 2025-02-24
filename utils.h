#ifndef UTILS_H
#define UTILS_H

// ��������� ������� ���������� ������� ������� N x N.
// ������������ �������� ��������������� ������� 10.0, ��������� � ��������� ������� �����.
void generateMatrix(double* A, int N);

// ��������� ������� ������ ����� ����� N �� ���������� ������ ����������.
void generateVector(double* b, int N);

// ����� ������� �� �����.
void printVector(const double* vec, int N);

// ����� ������� �� �����.
void printMatrix(const double* A, int N);

#endif // UTILS_H
