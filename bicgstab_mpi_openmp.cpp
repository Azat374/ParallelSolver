#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include "preconditioner_mpi.h"

extern "C" void BiCGStab2_MPI_OpenMP(int N, const double* A, double* x,
    const double* b, double tol, int maxIter, int* iterCount)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base = N / size;
    int rem = N % size;
    int local_N = base + (rank < rem ? 1 : 0);
    int offset = rank * base + (rank < rem ? rank : rem);

    int* counts = new int[size];
    int* displs = new int[size];
    int* counts_b = new int[size];
    int* displs_b = new int[size];
    for (int r = 0; r < size; r++) {
        counts[r] = (base + (r < rem ? 1 : 0)) * N;
        displs[r] = (r * base + (r < rem ? r : rem)) * N;
        counts_b[r] = base + (r < rem ? 1 : 0);
        displs_b[r] = r * base + (r < rem ? r : rem);
    }
    double* local_A = new double[counts[rank]];
    double* local_b = new double[counts_b[rank]];
    double* local_x = new double[counts_b[rank]];
    memset(local_x, 0, counts_b[rank] * sizeof(double));

    MPI_Scatterv(A, counts, displs, MPI_DOUBLE, local_A, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts_b, displs_b, MPI_DOUBLE, local_b, counts_b[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    ILU0_MPI(N, (double*)A); // Факторизация выполняется для полной матрицы на CPU (если требуется)

    double* r = new double[counts_b[rank]];
    double* r_hat = new double[counts_b[rank]];
    double* p = new double[counts_b[rank]];
    double* v = new double[counts_b[rank]];
    double* s = new double[counts_b[rank]];
    double* t = new double[counts_b[rank]];

#pragma omp parallel for
    for (int i = 0; i < counts_b[rank]; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += local_A[i * N + j] * local_x[j % local_N];
        }
        r[i] = local_b[i] - sum;
        r_hat[i] = r[i];
        p[i] = 0.0;
        v[i] = 0.0;
    }

    double local_normb = 0.0;
#pragma omp parallel for reduction(+:local_normb)
    for (int i = 0; i < counts_b[rank]; i++)
        local_normb += local_b[i] * local_b[i];
    double global_normb = 0.0;
    MPI_Allreduce(&local_normb, &global_normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_normb = sqrt(global_normb);
    if (global_normb < 1e-10) global_normb = 1.0;

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    int iter = 0;
    while (iter < maxIter) {
        double local_rho = 0.0;
#pragma omp parallel for reduction(+:local_rho)
        for (int i = 0; i < counts_b[rank]; i++)
            local_rho += r_hat[i] * r[i];
        double rho;
        MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rho) < tol) break;
        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);
#pragma omp parallel for
        for (int i = 0; i < counts_b[rank]; i++)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        double* global_p = new double[N];
        MPI_Allgather(p, counts_b[rank], MPI_DOUBLE, global_p, counts_b[0], MPI_DOUBLE, MPI_COMM_WORLD);
#pragma omp parallel for
        for (int i = 0; i < counts_b[rank]; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += local_A[i * N + j] * global_p[j];
            }
            v[i] = sum;
        }
        delete[] global_p;
        double local_rhat_v = 0.0;
#pragma omp parallel for reduction(+:local_rhat_v)
        for (int i = 0; i < counts_b[rank]; i++)
            local_rhat_v += r_hat[i] * v[i];
        double rhat_v;
        MPI_Allreduce(&local_rhat_v, &rhat_v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rhat_v) < tol) break;
        alpha = rho / rhat_v;
#pragma omp parallel for
        for (int i = 0; i < counts_b[rank]; i++)
            s[i] = r[i] - alpha * v[i];
        double local_norm_s = 0.0;
#pragma omp parallel for reduction(+:local_norm_s)
        for (int i = 0; i < counts_b[rank]; i++)
            local_norm_s += s[i] * s[i];
        double norm_s = sqrt(local_norm_s);
        if (norm_s / global_normb < tol) {
#pragma omp parallel for
            for (int i = 0; i < counts_b[rank]; i++)
                local_x[i] += alpha * p[i];
            iter++;
            break;
        }
        double* global_s = new double[N];
        MPI_Allgather(s, counts_b[rank], MPI_DOUBLE, global_s, counts_b[0], MPI_DOUBLE, MPI_COMM_WORLD);
#pragma omp parallel for
        for (int i = 0; i < counts_b[rank]; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += local_A[i * N + j] * global_s[j];
            }
            t[i] = sum;
        }
        delete[] global_s;
        double local_ts = 0.0, local_tt = 0.0;
        for (int i = 0; i < counts_b[rank]; i++) {
            local_ts += t[i] * s[i];
            local_tt += t[i] * t[i];
        }
        double ts, tt;
        MPI_Allreduce(&local_ts, &ts, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_tt, &tt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(tt) < tol) break;
        omega = ts / tt;
#pragma omp parallel for
        for (int i = 0; i < counts_b[rank]; i++) {
            local_x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }
        double local_nr = 0.0;
        for (int i = 0; i < counts_b[rank]; i++)
            local_nr += r[i] * r[i];
        double norm_r = sqrt(local_nr);
        if (norm_r / global_normb < tol) break;
        rho_old = rho;
        iter++;
    }
    MPI_Gatherv(local_x, counts_b[rank], MPI_DOUBLE, x, counts_b, displs_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (iterCount)
        *iterCount = iter;

    delete[] local_A; delete[] local_b; delete[] local_x;
}
