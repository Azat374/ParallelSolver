#include "preconditioner_mpi.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <cstdlib>

extern "C" void ILU0_MPI(int N, double* A) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base = N / size;
    int rem = N % size;
    int local_N = base + (rank < rem ? 1 : 0);

    int* counts = new int[size];
    int* displs = new int[size];
    for (int r = 0; r < size; r++) {
        counts[r] = (base + (r < rem ? 1 : 0)) * N;
        displs[r] = (r * base + (r < rem ? r : rem)) * N;
    }

    double* local_A = new double[local_N * N];
    MPI_Scatterv(A, counts, displs, MPI_DOUBLE, local_A, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Выполняем ILU0 для локальной части
    for (int k = 0; k < local_N; k++) {
        double diag = local_A[k * N + k];
        if (fabs(diag) < 1e-12) diag = 1e-12;
        for (int i = k + 1; i < local_N; i++) {
            local_A[i * N + k] /= diag;
        }
        for (int i = k + 1; i < local_N; i++) {
            for (int j = k + 1; j < N; j++) {
                local_A[i * N + j] -= local_A[i * N + k] * local_A[k * N + j];
            }
        }
    }

    MPI_Gatherv(local_A, counts[rank], MPI_DOUBLE, A, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete[] local_A;
    delete[] counts;
    delete[] displs;
}
