#include "utils.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>

// Генерация плотной матрицы A (NxN) с диагональными элементами равными 10.0, а вне диагонали случайными числами
void generateDenseMatrix(std::vector<double>& A, int N) {
    A.resize(N * N);
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 10.0 : (rand() % 10) / 10.0;
        }
    }
}

// Преобразование плотной матрицы в формат CSR
void convertDenseToCSR(const std::vector<double>& A, int N,
    std::vector<double>& values,
    std::vector<int>& rowPtr,
    std::vector<int>& colIdx) {
    values.clear();
    colIdx.clear();
    rowPtr.resize(N + 1, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double elem = A[i * N + j];
            if (fabs(elem) > 1e-12) {
                values.push_back(elem);
                colIdx.push_back(j);
            }
        }
        rowPtr[i + 1] = values.size();
    }
}

// Генерация вектора b размера N
void generateVector(double* b, int N) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++) {
        b[i] = rand() % 10;
    }
}

// Функция для получения текущего времени (в секундах)
double get_time() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
}

// Функция для печати вектора (для отладки)
void printVector(const double* vec, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}
