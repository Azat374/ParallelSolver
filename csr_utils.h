#ifndef CSR_UTILS_H
#define CSR_UTILS_H

#include <vector>

struct CSRMatrix {
    std::vector<double> values;
    std::vector<int> colIdx;
    std::vector<int> rowPtr;
    int N;
};

void denseToCSR(const std::vector<double>& A, int N, CSRMatrix& csr);
void printCSRMatrix(const CSRMatrix& csr);

#endif // CSR_UTILS_H
