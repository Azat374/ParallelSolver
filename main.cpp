#include <iostream>
#include <vector>
#include <mpi.h>
#include <numeric>
#include <cmath>
#include "utils.h"

// Объявления внешних функций (GPU-решатель, CPU-решатель)
extern "C" void BiCGStab2_GPU(int N, const std::vector<double>&csr_values, const std::vector<int>&csr_rowPtr,
    const std::vector<int>&csr_colIdx, double* x, const double* b,
    double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_CPU(int N, const std::vector<double>&A, double* x, const double* b,
    double tol, int maxIter, int* iterCount);

// Реализация метода Якоби (для плотной матрицы)
void jacobiMethod(int N, const std::vector<double>& A, double* x, const double* b, int maxIter, double tol) {
    std::vector<double> x_new(N, 0.0);
    for (int iter = 0; iter < maxIter; iter++) {
        for (int i = 0; i < N; i++) {
            double sigma = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sigma += A[i * N + j] * x[j];
                }
            }
            x_new[i] = (b[i] - sigma) / A[i * N + i];
        }
        double err = 0.0;
        for (int i = 0; i < N; i++) {
            err += fabs(x_new[i] - x[i]);
            x[i] = x_new[i];
        }
        if (err < tol) break;
    }
}

// Вычисление нормы остатка: ||b - A*x||
double computeResidualNorm(int N, const std::vector<double>& A, const double* x, const double* b) {
    std::vector<double> r(N, 0.0);
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        r[i] = b[i] - sum;
    }
    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        norm += r[i] * r[i];
    }
    return sqrt(norm);
}

// Вычисление стандартного отклонения
double computeStdDev(const std::vector<double>& times, double mean) {
    double sumSq = 0.0;
    for (double t : times) {
        sumSq += (t - mean) * (t - mean);
    }
    return sqrt(sumSq / times.size());
}

#define N 1024
#define TRIALS 5

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Генерация входных данных на процессе 0
    std::vector<double> dense_A;
    double* b = new double[N];
    if (rank == 0) {
        generateDenseMatrix(dense_A, N);
        generateVector(b, N);
    }
    if (rank != 0) {
        dense_A.resize(N * N);
    }
    MPI_Bcast(dense_A.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Преобразование плотной матрицы в CSR на процессе 0 и рассылка всем процессам
    std::vector<double> csr_values;
    std::vector<int> csr_rowPtr, csr_colIdx;
    if (rank == 0) {
        convertDenseToCSR(dense_A, N, csr_values, csr_rowPtr, csr_colIdx);
    }
    int csr_values_size, csr_rowPtr_size, csr_colIdx_size;
    if (rank == 0) {
        csr_values_size = csr_values.size();
        csr_rowPtr_size = csr_rowPtr.size();
        csr_colIdx_size = csr_colIdx.size();
    }
    MPI_Bcast(&csr_values_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&csr_rowPtr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&csr_colIdx_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        csr_values.resize(csr_values_size);
        csr_rowPtr.resize(csr_rowPtr_size);
        csr_colIdx.resize(csr_colIdx_size);
    }
    MPI_Bcast(csr_values.data(), csr_values_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(csr_rowPtr.data(), csr_rowPtr_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(csr_colIdx.data(), csr_colIdx_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Выделение памяти для решений
    double* x_gpu = new double[N]();
    double* x_cpu = new double[N]();
    double* x_jacobi = new double[N]();

    std::vector<double> times_gpu, times_cpu, times_jacobi;
    std::vector<int> iterCounts_gpu, iterCounts_cpu;

    std::cout << "Process " << rank << " out of " << size << " is computing..." << std::endl;

    for (int trial = 0; trial < TRIALS; trial++) {
        std::fill(x_gpu, x_gpu + N, 0.0);
        std::fill(x_cpu, x_cpu + N, 0.0);
        std::fill(x_jacobi, x_jacobi + N, 0.0);
        int iter_gpu = 0, iter_cpu = 0;

        double start_gpu = MPI_Wtime();
        BiCGStab2_GPU(N, csr_values, csr_rowPtr, csr_colIdx, x_gpu, b, 1e-6, 1000, &iter_gpu);
        double end_gpu = MPI_Wtime();
        times_gpu.push_back(end_gpu - start_gpu);
        iterCounts_gpu.push_back(iter_gpu);

        double start_cpu = MPI_Wtime();
        BiCGStab2_CPU(N, dense_A, x_cpu, b, 1e-6, 1000, &iter_cpu);
        double end_cpu = MPI_Wtime();
        times_cpu.push_back(end_cpu - start_cpu);
        iterCounts_cpu.push_back(iter_cpu);

        double start_jacobi = MPI_Wtime();
        jacobiMethod(N, dense_A, x_jacobi, b, 10000, 1e-6);
        double end_jacobi = MPI_Wtime();
        times_jacobi.push_back(end_jacobi - start_jacobi);
    }

    auto avgTime = [](const std::vector<double>& times) {
        return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    };
    double avg_gpu = avgTime(times_gpu);
    double avg_cpu = avgTime(times_cpu);
    double avg_jacobi = avgTime(times_jacobi);

    double std_gpu = computeStdDev(times_gpu, avg_gpu);
    double std_cpu = computeStdDev(times_cpu, avg_cpu);
    double std_jacobi = computeStdDev(times_jacobi, avg_jacobi);

    double avgIter_gpu = std::accumulate(iterCounts_gpu.begin(), iterCounts_gpu.end(), 0.0) / iterCounts_gpu.size();
    double avgIter_cpu = std::accumulate(iterCounts_cpu.begin(), iterCounts_cpu.end(), 0.0) / iterCounts_cpu.size();

    double residual_gpu = computeResidualNorm(N, dense_A, x_gpu, b);
    double residual_cpu = computeResidualNorm(N, dense_A, x_cpu, b);
    double residual_jacobi = computeResidualNorm(N, dense_A, x_jacobi, b);

    // Примерная оценка GFLOPS (учитываем количество операций в BiCGStab(2))
    double flops_cpu = 8.0 * N * N * avgIter_cpu;
    double gflops_cpu = (avg_cpu > 0) ? flops_cpu / (avg_cpu * 1e9) : 0;
    double nnz = csr_values.size();
    double flops_gpu = 4.0 * nnz * avgIter_gpu;
    double gflops_gpu = (avg_gpu > 0) ? flops_gpu / (avg_gpu * 1e9) : 0;

    double speedup = (avg_gpu > 0) ? avg_cpu / avg_gpu : 0;
    double efficiency = (size > 0) ? speedup / size : 0;

    if (rank == 0) {
        std::cout << "\n--- Experimental Results (averaged over " << TRIALS << " trials) ---" << std::endl;
        std::cout << "GPU BiCGStab(2):" << std::endl;
        std::cout << "  Avg time: " << avg_gpu * 1000 << " ms, StdDev: " << std_gpu * 1000 << " ms" << std::endl;
        std::cout << "  Avg iterations: " << avgIter_gpu << std::endl;
        std::cout << "  Residual norm: " << residual_gpu << std::endl;
        std::cout << "  GFLOPS: " << gflops_gpu << std::endl;

        std::cout << "\nCPU BiCGStab(2):" << std::endl;
        std::cout << "  Avg time: " << avg_cpu * 1000 << " ms, StdDev: " << std_cpu * 1000 << " ms" << std::endl;
        std::cout << "  Avg iterations: " << avgIter_cpu << std::endl;
        std::cout << "  Residual norm: " << residual_cpu << std::endl;
        std::cout << "  GFLOPS: " << gflops_cpu << std::endl;

        std::cout << "\nJacobi Method:" << std::endl;
        std::cout << "  Avg time: " << avg_jacobi * 1000 << " ms, StdDev: " << std_jacobi * 1000 << " ms" << std::endl;
        std::cout << "  Residual norm: " << residual_jacobi << std::endl;

        std::cout << "\nSpeedup (CPU BiCGStab(2) / GPU BiCGStab(2)): " << speedup << std::endl;
        std::cout << "Efficiency (Speedup / " << size << " processes): " << efficiency << std::endl;
        std::cout << "\nNote: Energy consumption and memory usage metrics are not measured in this experiment." << std::endl;
    }

    delete[] b;
    delete[] x_gpu;
    delete[] x_cpu;
    delete[] x_jacobi;

    MPI_Finalize();
    return 0;
}
