#ifndef UTILS_H
#define UTILS_H

#include <vector>

// Генерация плотной матрицы размером N x N (row-major)
void generateDenseMatrix(std::vector<double>& A, int N);

// Преобразование плотной матрицы в формат CSR
void convertDenseToCSR(const std::vector<double>& A, int N,
    std::vector<double>& values,
    std::vector<int>& rowPtr,
    std::vector<int>& colIdx);

// Генерация вектора правой части системы b размера N
void generateVector(double* b, int N);

// Функция для получения текущего времени (в секундах)
double get_time();

// Функция для печати вектора (для отладки)
void printVector(const double* vec, int N);

#endif // UTILS_H
