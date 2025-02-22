#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// Функция матрично-векторного умножения для плотной матрицы (row-major)
void denseMatVec(int N, const std::vector<double>& A, const double* x, double* y) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// Функция вычисления скалярного произведения двух векторов
double denseDotProduct(int N, const double* x, const double* y) {
    double dot = 0.0;
#pragma omp parallel for reduction(+:dot)
    for (int i = 0; i < N; i++) {
        dot += x[i] * y[i];
    }
    return dot;
}

// Копирование вектора: dest = src
void denseCopy(int N, const double* src, double* dest) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        dest[i] = src[i];
    }
}

// Реализация BiCGStab(2) для плотной матрицы A (row-major) на CPU с использованием OpenMP.
// При завершении работы в iterCount возвращается число итераций.
extern "C" void BiCGStab2_CPU(int N, const std::vector<double>&A, double* x, const double* b,
    double tol, int maxIter, int* iterCount) {
    std::vector<double> r(N), r_hat(N), p(N, 0.0), v(N, 0.0), s(N), t(N);

    // r = b - A*x
    denseMatVec(N, A, x, r.data());
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - r[i];
    }
    // r_hat = r (фиксируем начальный вектор)
    r_hat = r;

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    int iter = 0;
    double norm_r = sqrt(denseDotProduct(N, r.data(), r.data()));

    while (iter < maxIter && norm_r > tol) {
        for (int j = 1; j <= 2; j++) { // два шага стабилизации BiCGStab(2)
            double rho = denseDotProduct(N, r_hat.data(), r.data());
            if (fabs(rho) < tol) {
                std::cerr << "CPU Breakdown: rho near zero at iteration " << iter << std::endl;
                *iterCount = iter;
                return;
            }
            double beta = (j == 1) ? 0.0 : (rho / rho_old) * (alpha / omega);
            if (j == 1) {
                denseCopy(N, r.data(), p.data());
            }
            else {
                for (int i = 0; i < N; i++) {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                }
            }

            denseMatVec(N, A, p.data(), v.data());
            double rhat_dot_v = denseDotProduct(N, r_hat.data(), v.data());
            if (fabs(rhat_dot_v) < tol) {
                std::cerr << "CPU Breakdown: rhat_dot_v near zero at iteration " << iter << std::endl;
                *iterCount = iter;
                return;
            }
            alpha = rho / rhat_dot_v;

            // s = r - alpha*v
            for (int i = 0; i < N; i++) {
                s[i] = r[i] - alpha * v[i];
            }
            double norm_s = sqrt(denseDotProduct(N, s.data(), s.data()));
            if (norm_s < tol) {
                for (int i = 0; i < N; i++) {
                    x[i] += alpha * p[i];
                }
                *iterCount = iter;
                return;
            }

            denseMatVec(N, A, s.data(), t.data());
            double t_dot_t = denseDotProduct(N, t.data(), t.data());
            if (fabs(t_dot_t) < tol) {
                std::cerr << "CPU Breakdown: t_dot_t near zero at iteration " << iter << std::endl;
                *iterCount = iter;
                return;
            }
            double t_dot_s = denseDotProduct(N, t.data(), s.data());
            omega = t_dot_s / t_dot_t;

            // Обновление решения: x = x + alpha*p + omega*s
            for (int i = 0; i < N; i++) {
                x[i] += alpha * p[i] + omega * s[i];
            }
            // Обновление остатка: r = s - omega*t
            for (int i = 0; i < N; i++) {
                r[i] = s[i] - omega * t[i];
            }

            norm_r = sqrt(denseDotProduct(N, r.data(), r.data()));
            if (norm_r < tol) break;
            rho_old = rho;
        }
        iter++;
    }
    *iterCount = iter;
    std::cout << "CPU BiCGStab(2) iterations: " << iter << std::endl;
}
