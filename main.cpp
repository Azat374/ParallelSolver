#include <iostream>
#include <mpi.h>
#include "utils.h"

// ќбъ€влени€ внешних функций
extern "C" void BiCGStab2_CPU(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_GPU(const double* A, double* x, const double* b, int N, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI_CUDA(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI_OpenMP(int N, const double* A, double* x, const double* b, double tol, int maxIter, int* iterCount);

int main(int argc, char** argv)
{
    setlocale(LC_ALL, "Russian");
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // –азмер системы (можно измен€ть дл€ тестировани€)
    const int N = 512;
    double* A = new double[N * N];
    double* b = new double[N];
    double* x_cpu = new double[N];
    double* x_gpu = new double[N];
    double* x_mpi = new double[N];
    double* x_mpi_cuda = new double[N];
    double* x_mpi_openmp = new double[N];
    int iter_cpu = 0, iter_gpu = 0, iter_mpi = 0, iter_mpi_cuda = 0, iter_mpi_openmp = 0;

    // “олько процесс 0 генерирует данные
    if (rank == 0) {
        generateMatrix(A, N);
        generateVector(b, N);
        for (int i = 0; i < N; i++) {
            x_cpu[i] = 0.0;
            x_gpu[i] = 0.0;
            x_mpi[i] = 0.0;
            x_mpi_cuda[i] = 0.0;
            x_mpi_openmp[i] = 0.0;
        }
    }

    // –аспространение данных на все процессы
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "«апуск CPU-решател€ BiCGStab2 с предобусловливанием..." << std::endl;
    }
    BiCGStab2_CPU(N, A, x_cpu, b, 1e-6, 1000, &iter_cpu);
    if (rank == 0) {
        std::cout << "CPU: найдено решение за " << iter_cpu << " итераций:" << std::endl;
        printVector(x_cpu, N);
    }

    if (rank == 0) {
        std::cout << "«апуск GPU-решател€ BiCGStab2 с предобусловливанием..." << std::endl;
    }
    BiCGStab2_GPU(A, x_gpu, b, N, 1e-6, 1000, &iter_gpu);
    if (rank == 0) {
        std::cout << "GPU: найдено решение за " << iter_gpu << " итераций:" << std::endl;
        printVector(x_gpu, N);
    }

    if (rank == 0) {
        std::cout << "«апуск MPI-решател€ BiCGStab2 с предобусловливанием..." << std::endl;
    }
    BiCGStab2_MPI(N, A, x_mpi, b, 1e-6, 1000, &iter_mpi);
    if (rank == 0) {
        std::cout << "MPI: найдено решение за " << iter_mpi << " итераций:" << std::endl;
        printVector(x_mpi, N);
    }

    if (rank == 0) {
        std::cout << "44«апуск MPI+CUDA-решател€ BiCGStab2 с предобусловливанием..." << std::endl;
    }
    BiCGStab2_MPI_CUDA(N, A, x_mpi_cuda, b, 1e-6, 1000, &iter_mpi_cuda);
    if (rank == 0) {
        std::cout << "44MPI+CUDA: найдено решение за " << iter_mpi_cuda << " итераций:" << std::endl;
        printVector(x_mpi_cuda, N);
    }

    if (rank == 0) {
        std::cout << "55«апуск MPI+OpenMP-решател€ BiCGStab2 с предобусловливанием..." << std::endl;
    }
    BiCGStab2_MPI_OpenMP(N, A, x_mpi_openmp, b, 1e-6, 1000, &iter_mpi_openmp);
    if (rank == 0) {
        std::cout << "55MPI+OpenMP: найдено решение за " << iter_mpi_openmp << " итераций:" << std::endl;
        printVector(x_mpi_openmp, N);
    }

    delete[] A;
    delete[] b;
    delete[] x_cpu;
    delete[] x_gpu;
    delete[] x_mpi;
    delete[] x_mpi_cuda;
    delete[] x_mpi_openmp;

    MPI_Finalize();
    return 0;
}
