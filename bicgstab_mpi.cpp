#include <mpi.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include "preconditioner_mpi.h"

extern "C" void BiCGStab2_MPI(int N, const double* A, double* x,
    const double* b, double tol, int maxIter, int* iterCount)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base = N / size;
    int rem = N % size;
    int local_N = base + (rank < rem ? 1 : 0);

    // Формируем массивы для Scatterv/Gatherv
    int* counts = new int[size];
    int* displs = new int[size];
    int* counts_b = new int[size];
    int* displs_b = new int[size];
    int offset = 0, offset_b = 0;

    for (int r = 0; r < size; r++) {
        int rows = base + (r < rem ? 1 : 0);
        counts[r] = rows * N;
        displs[r] = offset;
        counts_b[r] = rows;
        displs_b[r] = offset_b;
        offset += counts[r];
        offset_b += rows;
    }

    double* local_A = new double[counts[rank]];
    double* local_b = new double[counts_b[rank]];
    double* local_x = new double[counts_b[rank]]();
    MPI_Scatterv(A, counts, displs, MPI_DOUBLE, local_A, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts_b, displs_b, MPI_DOUBLE, local_b, counts_b[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    ILU0_MPI(counts_b[rank], local_A);  // ILU0 применяем только к локальной части!

    int local_size = counts_b[rank];
    double* r = new double[local_size];
    double* r_hat = new double[local_size];
    double* p = new double[local_size];
    double* v = new double[local_size];
    double* s = new double[local_size];
    double* t = new double[local_size];

    for (int i = 0; i < local_size; ++i) {
        r[i] = local_b[i];
        r_hat[i] = r[i];
        p[i] = v[i] = 0.0;
    }

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    int iter = 0;

    // Вычисляем норму правой части
    double local_normb = 0.0;
    for (int i = 0; i < local_size; ++i)
        local_normb += local_b[i] * local_b[i];
    double normb = 0.0;
    MPI_Allreduce(&local_normb, &normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    normb = std::sqrt(normb);
    if (normb < 1e-10) normb = 1.0;

    // Буферы для Allgatherv
    int global_N = N;
    double* global_p = new double[N];
    double* global_s = new double[N];

    while (iter < maxIter) {
        double local_rho = 0.0;
        for (int i = 0; i < local_size; ++i)
            local_rho += r_hat[i] * r[i];
        double rho = 0.0;
        MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (std::fabs(rho) < tol) break;

        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);
        for (int i = 0; i < local_size; ++i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        MPI_Allgatherv(p, local_size, MPI_DOUBLE, global_p, counts_b, displs_b, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int i = 0; i < local_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j)
                sum += local_A[i * N + j] * global_p[j];
            v[i] = sum;
        }

        double local_rhv = 0.0;
        for (int i = 0; i < local_size; ++i)
            local_rhv += r_hat[i] * v[i];
        double rhv = 0.0;
        MPI_Allreduce(&local_rhv, &rhv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (std::fabs(rhv) < tol) break;
        alpha = rho / rhv;

        for (int i = 0; i < local_size; ++i)
            s[i] = r[i] - alpha * v[i];

        double local_norm_s = 0.0;
        for (int i = 0; i < local_size; ++i)
            local_norm_s += s[i] * s[i];
        double norm_s = std::sqrt(local_norm_s);
        if (norm_s / normb < tol) {
            for (int i = 0; i < local_size; ++i)
                local_x[i] += alpha * p[i];
            iter++;
            break;
        }

        MPI_Allgatherv(s, local_size, MPI_DOUBLE, global_s, counts_b, displs_b, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int i = 0; i < local_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j)
                sum += local_A[i * N + j] * global_s[j];
            t[i] = sum;
        }

        double local_ts = 0.0, local_tt = 0.0;
        for (int i = 0; i < local_size; ++i) {
            local_ts += t[i] * s[i];
            local_tt += t[i] * t[i];
        }
        double ts = 0.0, tt = 0.0;
        MPI_Allreduce(&local_ts, &ts, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_tt, &tt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (std::fabs(tt) < tol) break;

        omega = ts / tt;

        for (int i = 0; i < local_size; ++i) {
            local_x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        double local_nr = 0.0;
        for (int i = 0; i < local_size; ++i)
            local_nr += r[i] * r[i];
        double norm_r = 0.0;
        MPI_Allreduce(&local_nr, &norm_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        norm_r = std::sqrt(norm_r);
        if (norm_r / normb < tol) break;

        rho_old = rho;
        iter++;
    }

    MPI_Gatherv(local_x, local_size, MPI_DOUBLE, x, counts_b, displs_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (iterCount) *iterCount = iter;

    // Освобождение ресурсов
    delete[] local_A;
    delete[] local_b;
    delete[] local_x;
    delete[] r; delete[] r_hat; delete[] p; delete[] v; delete[] s; delete[] t;
    delete[] counts; delete[] displs; delete[] counts_b; delete[] displs_b;
    delete[] global_p;
    delete[] global_s;
}
