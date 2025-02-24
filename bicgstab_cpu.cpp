#include <cmath>
#include <cstdlib>
#include <omp.h>
#include "preconditioner_cpu.h"

// Улучшенная реализация BiCGStab(2) на CPU с ILU(0) предобусловливанием.
// A – входная матрица (N x N) в формате row-major (неизменяемая)
// x – начальное приближение и итоговое решение (массив длины N)
// b – вектор правой части (массив длины N)
// tol – требуемая относительная точность
// maxIter – максимальное число итераций
// iterCount – выходное число итераций
extern "C" void BiCGStab2_CPU(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount)
{
    // Создаем копию матрицы A для предобусловливания
    double* A_ilu = new double[N * N];
    for (int i = 0; i < N * N; i++) {
        A_ilu[i] = A[i];
    }
    // Применяем ILU0 на CPU (in-place)
    ILU0_CPU(N, A_ilu);

    double* r = new double[N];
    double* r_hat = new double[N];
    double* p = new double[N];
    double* v = new double[N];
    double* s = new double[N];
    double* t = new double[N];

    // Вычисляем начальный остаток: r = b - A*x
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        r[i] = b[i] - sum;
        r_hat[i] = r[i];
        p[i] = 0.0;
        v[i] = 0.0;
    }

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    int iter = 0;
    double normb = 0.0;
    for (int i = 0; i < N; i++) normb += b[i] * b[i];
    normb = sqrt(normb);
    if (normb < 1e-10) normb = 1.0;

    while (iter < maxIter) {
        double rho = 0.0;
#pragma omp parallel for reduction(+:rho)
        for (int i = 0; i < N; i++) {
            rho += r_hat[i] * r[i];
        }
        if (fabs(rho) < tol) break;
        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * p[j];
            }
            v[i] = sum;
        }

        double rhat_dot_v = 0.0;
#pragma omp parallel for reduction(+:rhat_dot_v)
        for (int i = 0; i < N; i++) {
            rhat_dot_v += r_hat[i] * v[i];
        }
        if (fabs(rhat_dot_v) < tol) break;
        alpha = rho / rhat_dot_v;

#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            s[i] = r[i] - alpha * v[i];
        }
        double norm_s = 0.0;
#pragma omp parallel for reduction(+:norm_s)
        for (int i = 0; i < N; i++) {
            norm_s += s[i] * s[i];
        }
        norm_s = sqrt(norm_s);
        if (norm_s / normb < tol) {
#pragma omp parallel for
            for (int i = 0; i < N; i++) {
                x[i] += alpha * p[i];
            }
            iter++;
            break;
        }
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * s[j];
            }
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
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        double norm_r = 0.0;
        for (int i = 0; i < N; i++) {
            norm_r += r[i] * r[i];
        }
        norm_r = sqrt(norm_r);
        if (norm_r / normb < tol) break;

        rho_old = rho;
        iter++;
    }

    *iterCount = iter;

    delete[] r;
    delete[] r_hat;
    delete[] p;
    delete[] v;
    delete[] s;
    delete[] t;
    delete[] A_ilu;
}
