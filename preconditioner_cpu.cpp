#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

// ������� ���������� ILU(0) ��� ������� ������� A (������ N x N) � ������� row-major.
// ������������ A ���, ��� ������ ����������� ����� �������� L-������.
extern "C" void ILU0_CPU(int N, double* A) {
    for (int k = 0; k < N; k++) {
        double diag = A[k * N + k];
        if (fabs(diag) < 1e-12) {
            std::cerr << "Warning: near zero diagonal at row " << k << std::endl;
            diag = 1e-12;
        }
#pragma omp parallel for
        for (int i = k + 1; i < N; i++) {
            A[i * N + k] /= diag;
        }
#pragma omp parallel for collapse(2)
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i * N + j] -= A[i * N + k] * A[k * N + j];
            }
        }
    }
}
