#ifndef UTILS_H
#define UTILS_H

// Генерация плотной квадратной матрицы (N×N).
// Диагональные элементы устанавливаются равными 10, остальные – случайные дробные числа.
void generateMatrix(double* A, int N);

// Генерация вектора правой части длины N (случайные целые от 0 до 9).
void generateVector(double* b, int N);

// Печать вектора (для отладки).
void printVector(const double* vec, int N);

// Печать матрицы (для отладки, при малых N).
void printMatrix(const double* A, int N);

// Преобразование плотной матрицы в формат CSR.
// На выходе: values – ненулевые элементы, nnz – число ненулевых, rowPtr (размер N+1) и colIdx.
void convertDenseToCSR(const double* A, int N, double** values, int* nnz, int** rowPtr, int** colIdx);

#endif // UTILS_H
