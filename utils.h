#ifndef UTILS_H
#define UTILS_H

#include <vector>

// ��������� ������� ������� �������� N x N (row-major)
void generateDenseMatrix(std::vector<double>& A, int N);

// �������������� ������� ������� � ������ CSR
void convertDenseToCSR(const std::vector<double>& A, int N,
    std::vector<double>& values,
    std::vector<int>& rowPtr,
    std::vector<int>& colIdx);

// ��������� ������� ������ ����� ������� b ������� N
void generateVector(double* b, int N);

// ������� ��� ��������� �������� ������� (� ��������)
double get_time();

// ������� ��� ������ ������� (��� �������)
void printVector(const double* vec, int N);

#endif // UTILS_H
