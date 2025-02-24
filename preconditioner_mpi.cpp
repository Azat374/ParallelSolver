#include "preconditioner_mpi.h"
#include <mpi.h>
#include <cmath>
#include <iostream>

extern "C" void ILU0_MPI(int N, double* A)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_N = N / size;
    double* local_A = new double[local_N * N];

    MPI_Scatter(A, local_N * N, MPI_DOUBLE, local_A, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Применяем ILU0 для локальной части
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

    MPI_Gather(local_A, local_N * N, MPI_DOUBLE, A, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete[] local_A;
}
