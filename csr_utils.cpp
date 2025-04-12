#include "csr_utils.h"
#include <iostream>

void denseToCSR(const std::vector<double>& A, int N, CSRMatrix& csr) {
    csr.N = N;
    csr.rowPtr.resize(N + 1);
    csr.values.clear();
    csr.colIdx.clear();

    int nnz = 0;
    for (int i = 0; i < N; ++i) {
        csr.rowPtr[i] = nnz;
        for (int j = 0; j < N; ++j) {
            double val = A[i * N + j];
            if (val != 0.0) {
                csr.values.push_back(val);
                csr.colIdx.push_back(j);
                ++nnz;
            }
        }
    }
    csr.rowPtr[N] = nnz;
}

void printCSRMatrix(const CSRMatrix& csr) {
    std::cout << "Values: ";
    for (double v : csr.values)
        std::cout << v << " ";
    std::cout << "\nColumn Indices: ";
    for (int idx : csr.colIdx)
        std::cout << idx << " ";
    std::cout << "\nRow Pointers: ";
    for (int ptr : csr.rowPtr)
        std::cout << ptr << " ";
    std::cout << std::endl;
}
