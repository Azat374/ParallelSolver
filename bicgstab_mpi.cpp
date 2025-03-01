#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "preconditioner_mpi.h"

// Улучшенная реализация BiCGStab2 для распределённых вычислений с MPI (CPU версия).
extern "C" void BiCGStab2_MPI(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Предполагаем, что N делится на size
    int local_N = N / size;

    // Распределяем матрицу A (по строкам) и вектор b
    double* local_A = new double[local_N * N];
    double* local_b = new double[local_N];
    // Локальное решение x для каждой группы строк (начальное приближение = 0)
    double* local_x = new double[local_N];
    memset(local_x, 0, local_N * sizeof(double));

    MPI_Scatter(A, local_N * N, MPI_DOUBLE, local_A, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_N, MPI_DOUBLE, local_b, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Применяем ILU(0) предобусловливание к локальной части матрицы
    ILU0_MPI(N, local_A);

    // Выделяем рабочие массивы для итерационного метода
    double* r = new double[local_N];
    double* r_hat = new double[local_N];
    double* p = new double[local_N];
    double* v = new double[local_N];
    double* s = new double[local_N];
    double* t = new double[local_N];

    // Инициализация:
    // При x = 0, A*x = 0, следовательно, начальный остаток r = b.
    for (int i = 0; i < local_N; i++) {
        r[i] = local_b[i];
        r_hat[i] = r[i]; // Выбираем r_hat = r (фиксированный вектор для вычисления ρ)
        p[i] = 0.0;
        v[i] = 0.0;
    }

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    int iter = 0;
    double global_normb = 0.0;
    double local_normb = 0.0;
    for (int i = 0; i < local_N; i++) {
        local_normb += local_b[i] * local_b[i];
    }
    MPI_Allreduce(&local_normb, &global_normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_normb = sqrt(global_normb);
    if (global_normb < 1e-10) global_normb = 1.0;

    while (iter < maxIter) {
        // Вычисляем ρ = <r_hat, r> локально и затем по всем процессам
        double local_rho = 0.0;
        for (int i = 0; i < local_N; i++) {
            local_rho += r_hat[i] * r[i];
        }
        double rho;
        MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rho) < tol) break;
        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);

        // Обновляем p: p = r + beta*(p - omega*v)
        for (int i = 0; i < local_N; i++) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Собираем глобальный вектор p от всех процессов
        double* global_p = new double[N];
        MPI_Allgather(p, local_N, MPI_DOUBLE, global_p, local_N, MPI_DOUBLE, MPI_COMM_WORLD);

        // Вычисляем v = A * global_p для локальных строк
        for (int i = 0; i < local_N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += local_A[i * N + j] * global_p[j];
            }
            v[i] = sum;
        }
        delete[] global_p;

        // Вычисляем <r_hat, v>
        double local_rhat_dot_v = 0.0;
        for (int i = 0; i < local_N; i++) {
            local_rhat_dot_v += r_hat[i] * v[i];
        }
        double rhat_dot_v;
        MPI_Allreduce(&local_rhat_dot_v, &rhat_dot_v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rhat_dot_v) < tol) break;
        alpha = rho / rhat_dot_v;

        // Вычисляем s = r - alpha*v
        for (int i = 0; i < local_N; i++) {
            s[i] = r[i] - alpha * v[i];
        }

        // Вычисляем норму s для проверки условия останова
        double local_norm_s = 0.0;
        for (int i = 0; i < local_N; i++) {
            local_norm_s += s[i] * s[i];
        }
        double norm_s = sqrt(local_norm_s);
        if (norm_s / global_normb < tol) {
            for (int i = 0; i < local_N; i++) {
                local_x[i] += alpha * p[i];
            }
            iter++;
            break;
        }

        // Вычисляем t = A * s для локальных строк
        for (int i = 0; i < local_N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += local_A[i * N + j] * s[j];
            }
            t[i] = sum;
        }

        double local_t_dot_s = 0.0, local_t_dot_t = 0.0;
        for (int i = 0; i < local_N; i++) {
            local_t_dot_s += t[i] * s[i];
            local_t_dot_t += t[i] * t[i];
        }
        double t_dot_s, t_dot_t;
        MPI_Allreduce(&local_t_dot_s, &t_dot_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_t_dot_t, &t_dot_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(t_dot_t) < tol) break;
        omega = t_dot_s / t_dot_t;

        // Обновляем локальное решение и остаток:
        for (int i = 0; i < local_N; i++) {
            local_x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        double local_norm_r = 0.0;
        for (int i = 0; i < local_N; i++) {
            local_norm_r += r[i] * r[i];
        }
        double norm_r = sqrt(local_norm_r);
        if (norm_r / global_normb < tol) break;

        rho_old = rho;
        iter++;
    }

    MPI_Gather(local_x, local_N, MPI_DOUBLE, x, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (iterCount) {
        *iterCount = iter;
    }


    delete[] local_A;
    delete[] local_b;
    delete[] local_x;
    delete[] r;
    delete[] r_hat;
    delete[] p;
    delete[] v;
    delete[] s;
    delete[] t;
}
