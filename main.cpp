#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cstring>
#include <numeric>
#include <vector>
#include "utils.h"

// Внешние объявления solver-функций:
extern "C" void BiCGStab2_CPU(int N, const double* A, double* x, const double* b,
    double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_GPU(const double* A, double* x, const double* b,
    int N, double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI(int N, const double* A, double* x, const double* b,
    double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI_CUDA(int N, const double* A, double* x, const double* b,
    double tol, int maxIter, int* iterCount);
extern "C" void BiCGStab2_MPI_OpenMP(int N, const double* A, double* x, const double* b,
    double tol, int maxIter, int* iterCount);

#define TRIALS 2

// Функция для вычисления среднего значения вектора
double avg(const std::vector<double>& times) {
    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Параметры по умолчанию:
    int N = 1000;
    double tol = 1e-6;
    int maxIter = 1000;
    std::string mode = "cpu";

    // Простой разбор аргументов:
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            tol = std::atof(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--maxiter") == 0 && i + 1 < argc) {
            maxIter = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        }
    }

    double* A = new double[N * N];
    double* b = new double[N];
    double* x = new double[N];
    std::memset(x, 0, N * sizeof(double));

    if (rank == 0) {
        generateMatrix(A, N);
        generateVector(b, N);

        // Вывод первых 10 строк и столбцов матрицы (для отладки)
        std::cout << "Matrix A (" << N << "x" << N << "):\n";
        for (int i = 0; i < std::min(N, 10); ++i) {
            for (int j = 0; j < std::min(N, 10); ++j) {
                std::cout << A[i * N + j] << " ";
            }
            std::cout << "\n";
        }

        // Вывод первых 10 элементов вектора b
        std::cout << "\nVector b (" << N << "):\n";
        for (int i = 0; i < std::min(N, 10); ++i) {
            std::cout << b[i] << " ";
        }
        std::cout << "\n\n";
    }

    // Передаём данные всем процессам
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> times_cpu, times_gpu, times_mpi, times_mpi_cuda;
    int iterCount = 0;
    for (int trial = 0; trial < TRIALS; trial++) {
        double start = MPI_Wtime();
        if (mode == "cpu") {
            if (rank == 0)
                BiCGStab2_CPU(N, A, x, b, tol, maxIter, &iterCount);
        }
        else if (mode == "gpu") {
            if (rank == 0)
                BiCGStab2_GPU(A, x, b, N, tol, maxIter, &iterCount);
        }
        else if (mode == "mpi") {
            BiCGStab2_MPI(N, A, x, b, tol, maxIter, &iterCount);
        }
        else if (mode == "mpi+cuda") {
            BiCGStab2_MPI_CUDA(N, A, x, b, tol, maxIter, &iterCount);
        }
        else if (mode == "mpi+openmp") {
            BiCGStab2_MPI_OpenMP(N, A, x, b, tol, maxIter, &iterCount);
        }
        else {
            if (rank == 0)
                std::cerr << "Неизвестный режим: " << mode << std::endl;
        }
        double end = MPI_Wtime();
        if (rank == 0) {
            if (mode == "cpu")
                times_cpu.push_back(end - start);
            else if (mode == "gpu")
                times_gpu.push_back(end - start);
            else if (mode == "mpi")
                times_mpi.push_back(end - start);
            else if (mode == "mpi+cuda")
                times_mpi_cuda.push_back(end - start);
        }
    }

    // На основном процессе (ранг 0) вычисляем результаты и записываем их в CSV файл
    if (rank == 0) {
        double gflops_cpu = (2.0 * N * N) / (avg(times_cpu) * 1e9);
        double gflops_gpu = (2.0 * N * N) / (avg(times_gpu) * 1e9);
        double gflops_mpi = (2.0 * N * N) / (avg(times_mpi) * 1e9);
        double gflops_mpi_cuda = (2.0 * N * N) / (avg(times_mpi_cuda) * 1e9);

        double speedup_gpu = (mode == "cpu") ? 0.0 : avg(times_cpu) / avg(times_gpu);
        double speedup_mpi = (mode == "cpu") ? 0.0 : avg(times_cpu) / avg(times_mpi);
        double speedup_mpi_cuda = (mode == "cpu") ? 0.0 : avg(times_cpu) / avg(times_mpi_cuda);

        std::cout << "\n--- Experimental Results (averaged over " << TRIALS << " trials) ---" << std::endl;
        if (mode == "cpu")
            std::cout << "CPU: " << avg(times_cpu) * 1000 << " ms, GFLOPS: " << gflops_cpu << std::endl;
        else if (mode == "gpu")
            std::cout << "GPU: " << avg(times_gpu) * 1000 << " ms, GFLOPS: " << gflops_gpu << ", Speedup: " << speedup_gpu << std::endl;
        else if (mode == "mpi")
            std::cout << "MPI (" << size << " procs): " << avg(times_mpi) * 1000 << " ms, GFLOPS: " << gflops_mpi << ", Speedup: " << speedup_mpi << std::endl;
        else if (mode == "mpi+cuda")
            std::cout << "MPI+CUDA: " << avg(times_mpi_cuda) * 1000 << " ms, GFLOPS: " << gflops_mpi_cuda << ", Speedup: " << speedup_mpi_cuda << std::endl;
        else if (mode == "mpi+openmp")
            std::cout << "MPI+OpenMP: " << avg(times_mpi) * 1000 << " ms, GFLOPS: " << gflops_mpi << ", Speedup: " << speedup_mpi << std::endl;

        // Формируем имя CSV файла в зависимости от режима
        std::string filename;
        if (mode == "cpu")
            filename = "results_cpu.csv";
        else if (mode == "gpu")
            filename = "results_gpu.csv";
        else if (mode == "mpi")
            filename = "results_mpi.csv";
        else if (mode == "mpi+cuda")
            filename = "results_mpi_cuda.csv";
        else if (mode == "mpi+openmp")
            filename = "results_mpi_openmp.csv";
        else
            filename = "results.csv";

        // Открываем файл для дозаписи (append)
        std::ofstream outFile;
        outFile.open(filename, std::ios::app);
        if (!outFile) {
            std::cerr << "Ошибка открытия файла " << filename << " для записи!" << std::endl;
        }
        else {
            // Если файл пустой – добавляем заголовок
            std::ifstream inFile(filename);
            bool fileEmpty = inFile.peek() == std::ifstream::traits_type::eof();
            inFile.close();
            if (fileEmpty) {
                outFile << "Matrix size (N),Max iterations,Process,Average time (ms),GFLOPS,Speedup,Mode\n";
            }

            if (mode == "cpu")
                outFile << N << "," << maxIter << "," << size << "," << avg(times_cpu) * 1000 << "," << gflops_cpu << "," << "NA" << "," << mode << "\n";
            else if (mode == "gpu")
                outFile << N << "," << maxIter << "," << size << "," << avg(times_gpu) * 1000 << "," << gflops_gpu << "," << speedup_gpu << "," << mode << "\n";
            else if (mode == "mpi")
                outFile << N << "," << maxIter << "," << size << "," << avg(times_mpi) * 1000 << "," << gflops_mpi << "," << speedup_mpi << "," << mode << "\n";
            else if (mode == "mpi+cuda")
                outFile << N << "," << maxIter << "," << size << "," << avg(times_mpi_cuda) * 1000 << "," << gflops_mpi_cuda << "," << speedup_mpi_cuda << "," << mode << "\n";
            else if (mode == "mpi+openmp")
                outFile << N << "," << maxIter << "," << size << "," << avg(times_mpi) * 1000 << "," << gflops_mpi << "," << speedup_mpi << "," << mode << "\n";
            outFile.close();
            std::cout << "The results are written to the file: " << filename << std::endl;
        }
    }

    delete[] A;
    delete[] b;
    delete[] x;

    MPI_Finalize();
    return 0;
}
