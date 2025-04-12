#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <iostream>
#include "preconditioner_cpu.h"

extern "C" void BiCGStab2_CPU(int N, const double* A, double* x, const double* b,
    double tol, int maxIter, int* iterCount)
{
    double* A_fact = new double[N * N];
    for (int i = 0; i < N * N; i++)
        A_fact[i] = A[i];
    double t_ilu_start = omp_get_wtime();
    ILU0_CPU(N, A_fact);
    double t_ilu_end = omp_get_wtime();

    double* r = new double[N];
    double* r_hat = new double[N];
    double* p = new double[N];
    double* v = new double[N];
    double* s = new double[N];
    double* t = new double[N];
    double* z = new double[N];

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
            sum += A[i * N + j] * x[j];
        r[i] = b[i] - sum;
        r_hat[i] = r[i];
        p[i] = 0.0;
        v[i] = 0.0;
    }

    double normb = 0.0;
#pragma omp parallel for reduction(+:normb)
    for (int i = 0; i < N; i++)
        normb += b[i] * b[i];
    normb = sqrt(normb);
    if (normb < 1e-10) normb = 1.0;

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    int iter = 0;
    double t0 = omp_get_wtime();
    while (iter < maxIter) {
        double rho = 0.0;
#pragma omp parallel for reduction(+:rho)
        for (int i = 0; i < N; i++)
            rho += r_hat[i] * r[i];
        if (fabs(rho) < tol) break;
        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);
#pragma omp parallel for
        for (int i = 0; i < N; i++)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        // Предобусловливание: решаем M*z = p
        double* y = new double[N];
        forwardSolve(N, A_fact, p, y);
        backwardSolve(N, A_fact, y, z);
        delete[] y;

        double t_spmv_start = omp_get_wtime();
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++)
                sum += A[i * N + j] * z[j];
            v[i] = sum;
        }
        double t_spmv = omp_get_wtime() - t_spmv_start;

        double t_dot_start = omp_get_wtime();
        double rhat_dot_v = 0.0;
#pragma omp parallel for reduction(+:rhat_dot_v)
        for (int i = 0; i < N; i++)
            rhat_dot_v += r_hat[i] * v[i];
        double t_dot = omp_get_wtime() - t_dot_start;
        if (fabs(rhat_dot_v) < tol) break;
        alpha = rho / rhat_dot_v;

#pragma omp parallel for
        for (int i = 0; i < N; i++)
            s[i] = r[i] - alpha * v[i];

        double norm_s = 0.0;
#pragma omp parallel for reduction(+:norm_s)
        for (int i = 0; i < N; i++)
            norm_s += s[i] * s[i];
        norm_s = sqrt(norm_s);
        if (norm_s / normb < tol) {
#pragma omp parallel for
            for (int i = 0; i < N; i++)
                x[i] += alpha * z[i];
            iter++;
            break;
        }

#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++)
                sum += A[i * N + j] * s[j];
            t[i] = sum;
        }
        double t_dot_s = 0.0, t_dot_t = 0.0;
        for (int i = 0; i < N; i++) {
            t_dot_s += t[i] * s[i];
            t_dot_t += t[i] * t[i];
        }
        if (fabs(t_dot_t) < tol) break;
        omega = t_dot_s / t_dot_t;

#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x[i] += alpha * z[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }
        double norm_r = 0.0;
        for (int i = 0; i < N; i++) {
            norm_r += r[i] * r[i];
        }
        norm_r = sqrt(norm_r);
        if (norm_r / normb < tol)
            break;

        rho_old = rho;
        iter++;
    }
    double t1 = omp_get_wtime();
    std::cout << "CPU: ILU time = " << (t_ilu_end - t_ilu_start) << " s, Total time = "
        << (t1 - t0) << " s" << std::endl;

    if (iterCount)
        *iterCount = iter;

    delete[] A_fact;
    delete[] r; delete[] r_hat; delete[] p; delete[] v; delete[] s; delete[] t; delete[] z;
}
