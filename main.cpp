#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>
#include <cmath>
#include <cuda_runtime.h>
#include "utils.h"

// Внешние объявления функций BiCGStab(2)
extern "C" void BiCGStab2_CPU(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_GPU(const double* A, double* x, const double* b, int N, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI_CUDA(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI_OpenMP(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);

#define TRIALS 5

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000;
    double* A = new (std::nothrow) double[N * N];
    double* b = new (std::nothrow) double[N];
    double* x_cpu = new (std::nothrow) double[N]();
    double* x_gpu = new (std::nothrow) double[N]();
    double* x_mpi_4 = new (std::nothrow) double[N]();
    double* x_mpi_8 = new (std::nothrow) double[N]();
    double* x_mpi_12 = new (std::nothrow) double[N]();
    double* x_mpi_cuda = new (std::nothrow) double[N]();

    if (!A || !b || !x_cpu || !x_gpu || !x_mpi_4 || !x_mpi_8 || !x_mpi_12 || !x_mpi_cuda) {
        std::cerr << "Ошибка выделения памяти!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        generateMatrix(A, N);
        generateVector(b, N);
    }

    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> times_cpu, times_gpu, times_mpi_4, times_mpi_8, times_mpi_12, times_mpi_cuda;

    for (int trial = 0; trial < TRIALS; trial++) {
        double start, end;

        start = MPI_Wtime();
        BiCGStab2_CPU(N, A, x_cpu, b, 1e-6, 1000, nullptr);
        end = MPI_Wtime();
        times_cpu.push_back(end - start);

        start = MPI_Wtime();
        BiCGStab2_GPU(A, x_gpu, b, N, 1e-6, 1000, nullptr);
        end = MPI_Wtime();
        times_gpu.push_back(end - start);

        start = MPI_Wtime();
        BiCGStab2_MPI(N, A, x_mpi_4, b, 1e-6, 1000, nullptr);
        end = MPI_Wtime();
        times_mpi_4.push_back(end - start);

        start = MPI_Wtime();
        BiCGStab2_MPI(N, A, x_mpi_8, b, 1e-6, 1000, nullptr);
        end = MPI_Wtime();
        times_mpi_8.push_back(end - start);

        start = MPI_Wtime();
        BiCGStab2_MPI(N, A, x_mpi_12, b, 1e-6, 1000, nullptr);
        end = MPI_Wtime();
        times_mpi_12.push_back(end - start);

        start = MPI_Wtime();
        BiCGStab2_MPI_CUDA(N, A, x_mpi_cuda, b, 1e-6, 1000, nullptr);
        end = MPI_Wtime();
        times_mpi_cuda.push_back(end - start);
    }

    if (rank == 0) {
        auto avg = [](const std::vector<double>& times) {
            return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        };

        double gflops_cpu = (2.0 * N * N) / (avg(times_cpu) * 1e9);
        double gflops_gpu = (2.0 * N * N) / (avg(times_gpu) * 1e9);
        double speedup_gpu = avg(times_cpu) / avg(times_gpu);

        std::cout << "\n--- Experimental Results ---" << std::endl;
        std::cout << "CPU: " << avg(times_cpu) * 1000 << " ms, GFLOPS: " << gflops_cpu << std::endl;
        std::cout << "GPU: " << avg(times_gpu) * 1000 << " ms, GFLOPS: " << gflops_gpu << ", Speedup: " << speedup_gpu << std::endl;
        std::cout << "MPI (4 cores): " << avg(times_mpi_4) * 1000 << " ms, GFLOPS: " << (2.0 * N * N) / (avg(times_mpi_4) * 1e9) << std::endl;
        std::cout << "MPI (8 cores): " << avg(times_mpi_8) * 1000 << " ms, GFLOPS: " << (2.0 * N * N) / (avg(times_mpi_8) * 1e9) << std::endl;
        std::cout << "MPI (12 cores): " << avg(times_mpi_12) * 1000 << " ms, GFLOPS: " << (2.0 * N * N) / (avg(times_mpi_12) * 1e9) << std::endl;
        std::cout << "MPI+CUDA: " << avg(times_mpi_cuda) * 1000 << " ms, GFLOPS: " << (2.0 * N * N) / (avg(times_mpi_cuda) * 1e9) << std::endl;
    }

    delete[] A;
    delete[] b;
    delete[] x_cpu;
    delete[] x_gpu;
    delete[] x_mpi_4;
    delete[] x_mpi_8;
    delete[] x_mpi_12;
    delete[] x_mpi_cuda;

    MPI_Finalize();
    return 0;
}
