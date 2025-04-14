#include "utils.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>

void generateMatrix(double* A, int N) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i * N + j] = (i == j) ? 10.0 : (rand() % 10) / 10.0;
}

void generateVector(double* b, int N) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++)
        b[i] = rand() % 10;
}

void printVector(const double* vec, int N) {
    for (int i = 0; i < N; i++)
        std::cout << vec[i] << " ";
    std::cout << std::endl;
}

void printMatrix(const double* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << A[i * N + j] << " ";
        std::cout << std::endl;
    }
}

void convertDenseToCSR(const double* A, int N, double** values, int* nnz, int** rowPtr, int** colIdx) {
    int count = 0;
    for (int i = 0; i < N * N; i++) {
        if (fabs(A[i]) > 1e-12)
            count++;
    }
    *nnz = count;
    *values = new double[count];
    *colIdx = new int[count];
    *rowPtr = new int[N + 1];
    int pos = 0;
    (*rowPtr)[0] = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double val = A[i * N + j];
            if (fabs(val) > 1e-12) {
                (*values)[pos] = val;
                (*colIdx)[pos] = j;
                pos++;
            }
        }
        (*rowPtr)[i + 1] = pos;
    }
}


