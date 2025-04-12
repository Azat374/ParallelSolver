#include "preconditioner_cpu.h"
#include <cmath>
#include <iostream>

extern "C" void ILU0_CPU(int N, double* A) {
    for (int k = 0; k < N; k++) {
        double diag = A[k * N + k];
        if (fabs(diag) < 1e-12) {
            std::cerr << "Warning: near zero diagonal at row " << k << std::endl;
            diag = 1e-12;
        }
        for (int i = k + 1; i < N; i++) {
            A[i * N + k] /= diag;
        }
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i * N + j] -= A[i * N + k] * A[k * N + j];
            }
        }
    }
}

extern "C" void forwardSolve(int N, const double* A, const double* b, double* y) {
    for (int i = 0; i < N; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= A[i * N + j] * y[j];
        }
        y[i] = sum;  // единичная диагональ
    }
}

extern "C" void backwardSolve(int N, const double* A, const double* y, double* x) {
    for (int i = N - 1; i >= 0; i--) {
        double sum = y[i];
        double diag = A[i * N + i];
        if (fabs(diag) < 1e-12) diag = 1e-12;
        for (int j = i + 1; j < N; j++) {
            sum -= A[i * N + j] * x[j];
        }
        x[i] = sum / diag;
    }
}
