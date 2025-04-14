#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include "preconditioner_cuda.h"
#include "device_launch_parameters.h"
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA ERROR in " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Кернелы для локальных операций
__global__ void updatePLocal(double* p, const double* r, const double* v, double beta, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
}

__global__ void computeVLocal(const double* A, const double* global_p, double* v, int N, int local_N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < local_N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
            sum += A[row * N + j] * global_p[j];
        v[row] = sum;
    }
}

__global__ void computeSLocal(double* s, const double* r, const double* v, double alpha, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        s[i] = r[i] - alpha * v[i];
}

__global__ void updateXFullLocal(double* x, const double* p, const double* s, double alpha, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        x[i] += alpha * p[i] + omega * s[i];
}

__global__ void updateRLocal(double* r, const double* s, const double* t, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        r[i] = s[i] - omega * t[i];
}

extern "C" void BiCGStab2_MPI_CUDA(int N, const double* A, double* x,
    const double* b, double tol, int maxIter, int* iterCount)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Разделение данных между процессами
    int base = N / size;
    int rem = N % size;
    int local_N = base + (rank < rem ? 1 : 0);


    if (rank == 0) {
        std::cout << "DEBUG: Total N = " << N << ", size = " << size << std::endl;
    }
    std::cout << "Rank " << rank << ": local_N = " << local_N << std::endl;


    // Массивы для Scatterv/Gatherv (выделяются один раз)
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

    // Выделение памяти на CPU
    double* local_A = new double[counts[rank]];
    double* local_b = new double[counts_b[rank]];
    double* local_x = new double[counts_b[rank]];
    memset(local_x, 0, counts_b[rank] * sizeof(double));

    
    // Распределение данных
    MPI_Scatterv(A, counts, displs, MPI_DOUBLE, local_A, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts_b, displs_b, MPI_DOUBLE, local_b, counts_b[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Выбор GPU
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    CUDA_CHECK(cudaSetDevice(rank % devCount));

    cudaEvent_t startEvent, stopEvent;
    float elapsed_ms = 0.0f;
    if (rank == 0) {

        std::cout << "Available GPUs: " << devCount << std::endl;

        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);
    }

    // Создание CUDA потока и cuBLAS хэндла
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // Выделение памяти на GPU (один раз перед циклом)
    double* d_A, * d_x, * d_b, * d_r, * d_p, * d_v, * d_s, * d_t;
    double* d_global_p; // Для коллективных операций
    double* h_p_buffer = new double[counts_b[rank]]; // Буфер для передачи p
    double* h_global_p = new double[N]; // Буфер для собранного p

    CUDA_CHECK(cudaMalloc(&d_A, counts[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_s, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_global_p, N * sizeof(double)));

    // Копирование данных на GPU (асинхронно)
    CUDA_CHECK(cudaMemcpyAsync(d_A, local_A, counts[rank] * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, local_b, counts_b[rank] * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_x, 0, counts_b[rank] * sizeof(double), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Ждем завершения копирования

    // Факторизация локальной части A на GPU
    ILU0_GPU(local_N, d_A);

    // Настройка размеров для CUDA кернелов
    int blockSize = 256;
    int gridSize = (local_N + blockSize - 1) / blockSize;

    // Инициализация: r = b (так как x = 0)
    CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_p, d_r, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    // Начальные параметры для BiCGStab
    double alpha = 1.0, omega = 1.0, rho_old = 1.0;
    int iter = 0;

    // Вычисление нормы b с использованием cuBLAS
    double local_normb = 0.0;
    cublasDnrm2(handle, local_N, d_b, 1, &local_normb);
    local_normb = local_normb * local_normb; // Квадрат нормы

    double global_normb = 0.0;
    MPI_Allreduce(&local_normb, &global_normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_normb = sqrt(global_normb);

    if (global_normb < 1e-10) global_normb = 1.0;

    // Итерационный процесс BiCGStab
    while (iter < maxIter) {
        // Вычисление rho = (r_hat, r) с использованием cuBLAS
        double local_rho = 0.0;
        cublasDdot(handle, local_N, d_r, 1, d_r, 1, &local_rho);

        double rho = 0.0;
        MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (fabs(rho) < tol) break;

        // Вычисление beta и обновление p
        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);
        updatePLocal << <gridSize, blockSize, 0, stream >> > (d_p, d_r, d_v, beta, omega, local_N);

        // Сбор всех p от всех процессов
        CUDA_CHECK(cudaMemcpyAsync(h_p_buffer, d_p, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Allgatherv(h_p_buffer, counts_b[rank], MPI_DOUBLE,
            h_global_p, counts_b, displs_b, MPI_DOUBLE, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(d_global_p, h_global_p, N * sizeof(double), cudaMemcpyHostToDevice, stream));

        // Вычисление v = A * p
        computeVLocal << <gridSize, blockSize, 0, stream >> > (d_A, d_global_p, d_v, N, local_N);

        // Вычисление dot product (r_hat, v)
        double local_rhat_dot_v = 0.0;
        cublasDdot(handle, local_N, d_r, 1, d_v, 1, &local_rhat_dot_v);

        double rhat_dot_v = 0.0;
        MPI_Allreduce(&local_rhat_dot_v, &rhat_dot_v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (fabs(rhat_dot_v) < tol) break;

        // Вычисление alpha и s
        alpha = rho / rhat_dot_v;
        computeSLocal << <gridSize, blockSize, 0, stream >> > (d_s, d_r, d_v, alpha, local_N);

        // Проверка нормы s
        double local_norm_s_sq = 0.0;
        cublasDnrm2(handle, local_N, d_s, 1, &local_norm_s_sq);
        local_norm_s_sq = local_norm_s_sq * local_norm_s_sq;

        double global_norm_s_sq = 0.0;
        MPI_Allreduce(&local_norm_s_sq, &global_norm_s_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double norm_s = sqrt(global_norm_s_sq);

        if (norm_s / global_normb < tol) {
            // x = x + alpha * p
            double alpha_val = alpha;
            cublasDaxpy(handle, local_N, &alpha_val, d_p, 1, d_x, 1);
            iter++;
            break;
        }

        // Вычисление t = A * s
        // Сначала соберем все s
        CUDA_CHECK(cudaMemcpyAsync(h_p_buffer, d_s, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Allgatherv(h_p_buffer, counts_b[rank], MPI_DOUBLE,
            h_global_p, counts_b, displs_b, MPI_DOUBLE, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(d_global_p, h_global_p, N * sizeof(double), cudaMemcpyHostToDevice, stream));

        computeVLocal << <gridSize, blockSize, 0, stream >> > (d_A, d_global_p, d_t, N, local_N);

        // Вычисление t_dot_s и t_dot_t
        double local_t_dot_s = 0.0, local_t_dot_t = 0.0;
        cublasDdot(handle, local_N, d_t, 1, d_s, 1, &local_t_dot_s);
        cublasDdot(handle, local_N, d_t, 1, d_t, 1, &local_t_dot_t);

        double t_dot_s = 0.0, t_dot_t = 0.0;
        MPI_Allreduce(&local_t_dot_s, &t_dot_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_t_dot_t, &t_dot_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (fabs(t_dot_t) < tol) break;

        // Вычисление omega
        omega = t_dot_s / (t_dot_t + 1e-16);

        // Обновление x и r
        updateXFullLocal << <gridSize, blockSize, 0, stream >> > (d_x, d_p, d_s, alpha, omega, local_N);
        updateRLocal << <gridSize, blockSize, 0, stream >> > (d_r, d_s, d_t, omega, local_N);

        // Проверка нормы r
        double local_norm_r_sq = 0.0;
        cublasDnrm2(handle, local_N, d_r, 1, &local_norm_r_sq);
        local_norm_r_sq = local_norm_r_sq * local_norm_r_sq;

        double global_norm_r_sq = 0.0;
        MPI_Allreduce(&local_norm_r_sq, &global_norm_r_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double norm_r = sqrt(global_norm_r_sq);

        if (norm_r / global_normb < tol) break;

        rho_old = rho;
        iter++;
    }



    // Сбор результатов
    CUDA_CHECK(cudaMemcpyAsync(local_x, d_x, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (rank == 0) {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        std::cout << "MPI+CUDA GPU time: " << elapsed_ms / 1000.0 << " s" << std::endl;
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_x, counts_b[rank], MPI_DOUBLE, x, counts_b, displs_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (iterCount)
        *iterCount = iter;

    // Освобождение ресурсов GPU
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_global_p));

    cublasDestroy(handle);
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Освобождение ресурсов CPU
    delete[] local_A;
    delete[] local_b;
    delete[] local_x;
    delete[] counts;
    delete[] displs;
    delete[] counts_b;
    delete[] displs_b;
    delete[] h_p_buffer;
    delete[] h_global_p;
}