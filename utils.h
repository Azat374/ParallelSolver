#ifndef UTILS_H
#define UTILS_H

// Генерация плотной квадратной матрицы размера N x N.
// Диагональные элементы устанавливаются равными 10.0, остальные – случайные дробные числа.
void generateMatrix(double* A, int N);

// Генерация вектора правой части длины N со случайными целыми значениями.
void generateVector(double* b, int N);

// Вывод вектора на экран.
void printVector(const double* vec, int N);

// Вывод матрицы на экран.
void printMatrix(const double* A, int N);

#endif // UTILS_H
