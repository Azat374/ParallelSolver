#include <iostream>
#include <cstdlib>
#include <ctime>
#include "utils.h"

void generateMatrix(double* A, int N) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 10.0 : (rand() % 10) / 10.0;
        }
    }
}

void generateVector(double* b, int N) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++) {
        b[i] = rand() % 10;
    }
}

void printVector(const double* vec, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void printMatrix(const double* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}
