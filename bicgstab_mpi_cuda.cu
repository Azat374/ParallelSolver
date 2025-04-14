#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include "spmv_kernel.h"   // должен содержать объявление SpMVKernelCSR
#include "utils.h"         // содержит convertDenseToCSR (помимо генерации матриц и векторов)
#include "device_launch_parameters.h"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if(err != cudaSuccess) {                                              \
            std::cerr << "CUDA ERROR in " << __FILE__ << ":" << __LINE__        \
                      << " — " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

const double eps = 1e-12;

// --- Таймер на GPU --- 
void gpuTimerStart(cudaEvent_t* start) {
    cudaEventCreate(start);
    cudaEventRecord(*start, 0);
}

float gpuTimerStop(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed;
}

// --- Локальные кернелы ---
// Обновление вектора: p = r + beta*(p - omega*v)
__global__ void updatePLocal(double* p, const double* r, const double* v, double beta, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        p[i] = r[i] + beta * (p[i] - omega * v[i]);
}

// Вычисление s: s = r - alpha*v
__global__ void computeSLocal(double* s, const double* r, const double* v, double alpha, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        s[i] = r[i] - alpha * v[i];
}

// Обновление x: x = x + alpha*p + omega*s
__global__ void updateXFullLocal(double* x, const double* p, const double* s, double alpha, double omega, int local_N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < local_N)
        x[i] += alpha * p[i] + omega * s[i];
}

// Обновление r: r = s - omega*t
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

    // Разделение данных: каждая MPI-процесс получает block строк
    int base = N / size;
    int rem = N % size;
    int local_N = base + (rank < rem ? 1 : 0);

    if (rank == 0)
        std::cout << "DEBUG: Total N = " << N << ", size = " << size << std::endl;
    std::cout << "Rank " << rank << ": local_N = " << local_N << std::endl;

    // Массивы для распределения плотной матрицы и вектора b
    int* counts = new int[size];
    int* displs = new int[size];
    int* counts_b = new int[size];
    int* displs_b = new int[size];
    for (int r = 0; r < size; r++) {
        counts[r] = (base + (r < rem ? 1 : 0)) * N; // каждая строка имеет N элементов
        displs[r] = (r * base + (r < rem ? r : rem)) * N;
        counts_b[r] = base + (r < rem ? 1 : 0);         // для вектора b – число строк
        displs_b[r] = r * base + (r < rem ? r : rem);
    }

    // Выделяем память для локальной плотной матрицы и локального вектора b
    double* local_A_dense = new double[counts[rank]];
    double* local_b = new double[counts_b[rank]];
    double* local_x = new double[counts_b[rank]];
    memset(local_x, 0, counts_b[rank] * sizeof(double));

    // Распределяем данные по MPI-процессам
    MPI_Scatterv(A, counts, displs, MPI_DOUBLE, local_A_dense, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts_b, displs_b, MPI_DOUBLE, local_b, counts_b[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Преобразуем локальную плотную матрицу в CSR-формат.
    // Функция convertDenseToCSR принимает: (dense_matrix, number_of_rows, &values, &nnz, &rowPtr, &colIdx).
    double* h_values = nullptr;
    int* h_rowPtr = nullptr;
    int* h_colIdx = nullptr;
    int local_nnz = 0;
    convertDenseToCSR(local_A_dense, local_N, &h_values, &local_nnz, &h_rowPtr, &h_colIdx);
    // Теперь local_A_dense более не нужен
    delete[] local_A_dense;

    // Выбор GPU: каждый процесс использует устройство по модулю своего ранга
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    CUDA_CHECK(cudaSetDevice(rank % devCount));
    if (rank == 0) std::cout << "Available GPUs: " << devCount << std::endl;

    // Замер времени на GPU (на процесс 0)
    cudaEvent_t startEvent, stopEvent;
    float elapsed_ms = 0.0f;
    if (rank == 0) {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);
    }

    // Создаём CUDA-поток и cuBLAS-хэндл
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetStream(cublasHandle, stream);

    // Выделение памяти для CSR-матрицы на GPU
    double* d_values; int* d_rowPtr; int* d_colIdx;
    CUDA_CHECK(cudaMalloc(&d_values, local_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (local_N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, local_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr, (local_N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    delete[] h_values; delete[] h_rowPtr; delete[] h_colIdx;

    // Выделение памяти для векторов на GPU (размер локального блока для локальных векторов и полный размер для коллективного вектора)
    double* d_x, * d_b, * d_r, * d_rhat, * d_p, * d_v, * d_s, * d_t;
    double* d_global_p;  // полный вектор размером N, используемый в коллективных операциях
    CUDA_CHECK(cudaMalloc(&d_x, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhat, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_s, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t, counts_b[rank] * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_global_p, N * sizeof(double)));

    // Копирование локального вектора b на GPU и инициализация d_x = 0
    CUDA_CHECK(cudaMemcpyAsync(d_b, local_b, counts_b[rank] * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_x, 0, counts_b[rank] * sizeof(double), stream));
    delete[] local_b; // дальше использовать на CPU не требуется

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Определяем размеры для CUDA кернелов: блоки и сетку по локальному числу строк
    int blockSize = 256;
    int gridSize = (local_N + blockSize - 1) / blockSize;

    // Инициализация: r = b и p = r
    CUDA_CHECK(cudaMemcpyAsync(d_r, d_b, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_p, d_r, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    // Устанавливаем начальные параметры метода BiCGStab
    double alpha = 1.0, omega = 1.0, rho_old = 1.0;
    int iter = 0;

    // Вычисляем норму локального вектора b с помощью cuBLAS
    double local_normb = 0.0;
    cublasDnrm2(cublasHandle, local_N, d_b, 1, &local_normb);
    local_normb = local_normb * local_normb;  // квадрат нормы
    double global_normb = 0.0;
    MPI_Allreduce(&local_normb, &global_normb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_normb = sqrt(global_normb);
    if (global_normb < 1e-10)
        global_normb = 1.0;

    // Выделяем CPU-буферы для сбора данных коллективных операций
    double* h_p_buffer = new double[counts_b[rank]]; // для копирования локального p с GPU
    double* h_global_p = new double[N];              // полный вектор p
    double* local_x_cpu = new double[counts_b[rank]];  // для локального решения

    // Главный итерационный цикл метода BiCGStab
    while (iter < maxIter) {
        // Вычисляем rho = (r_hat, r)
        double local_rho = 0.0;
        cublasDdot(cublasHandle, local_N, d_rhat, 1, d_r, 1, &local_rho);
        double rho = 0.0;
        MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rho) < tol)
            break;

        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);

        // Обновляем p = r + beta*(p - omega*v)
        updatePLocal << <gridSize, blockSize, 0, stream >> > (d_p, d_r, d_v, beta, omega, local_N);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Собираем локальное p с GPU в CPU-буфер
        CUDA_CHECK(cudaMemcpyAsync(h_p_buffer, d_p, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Собираем полный вектор p от всех процессов
        MPI_Allgatherv(h_p_buffer, counts_b[rank], MPI_DOUBLE,
            h_global_p, counts_b, displs_b, MPI_DOUBLE, MPI_COMM_WORLD);
        // Копируем полный вектор p обратно на GPU
        CUDA_CHECK(cudaMemcpyAsync(d_global_p, h_global_p, N * sizeof(double), cudaMemcpyHostToDevice, stream));

        // Вычисляем v = A * p с помощью CSR-формата: для локальной части (local_N строк)
        SpMVKernelCSR << <gridSize, blockSize, 0, stream >> > (local_N, d_values, d_rowPtr, d_colIdx, d_global_p, d_v);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Вычисляем dot product (r_hat, v)
        double local_rhat_dot_v = 0.0;
        cublasDdot(cublasHandle, local_N, d_rhat, 1, d_v, 1, &local_rhat_dot_v);
        double rhat_dot_v = 0.0;
        MPI_Allreduce(&local_rhat_dot_v, &rhat_dot_v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (fabs(rhat_dot_v) < tol)
            break;

        alpha = rho / (rhat_dot_v + eps);

        // Вычисляем s = r - alpha * v
        computeSLocal << <gridSize, blockSize, 0, stream >> > (d_s, d_r, d_v, alpha, local_N);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Вычисляем норму вектора s (с использованием cuBLAS) и собираем её по MPI
        double local_norm_s = 0.0;
        cublasDnrm2(cublasHandle, local_N, d_s, 1, &local_norm_s);
        double global_norm_s = 0.0;
        MPI_Allreduce(&local_norm_s, &global_norm_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if ((global_norm_s / global_normb) < tol) {
            // Обновляем x = x + alpha * p
            double alpha_val = alpha;
            cublasDaxpy(cublasHandle, local_N, &alpha_val, d_p, 1, d_x, 1);
            iter++;
            break;
        }

        // Сбор вектора s для вычисления t = A*s
        CUDA_CHECK(cudaMemcpyAsync(h_p_buffer, d_s, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Allgatherv(h_p_buffer, counts_b[rank], MPI_DOUBLE,
            h_global_p, counts_b, displs_b, MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(d_global_p, h_global_p, N * sizeof(double), cudaMemcpyHostToDevice, stream));

        // Вычисляем t = A * s (используя CSR, для локальной части)
        SpMVKernelCSR << <gridSize, blockSize, 0, stream >> > (local_N, d_values, d_rowPtr, d_colIdx, d_global_p, d_t);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Вычисляем скалярные произведения: t_dot_s и квадрат нормы t (t_dot_t)
        double local_t_dot_s = 0.0, local_t_norm = 0.0;
        cublasDdot(cublasHandle, local_N, d_t, 1, d_s, 1, &local_t_dot_s);
        cublasDnrm2(cublasHandle, local_N, d_t, 1, &local_t_norm);
        double local_t_dot_t = local_t_norm * local_t_norm;
        double t_dot_s = 0.0, t_dot_t = 0.0;
        MPI_Allreduce(&local_t_dot_s, &t_dot_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_t_dot_t, &t_dot_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (fabs(t_dot_t) < tol)
            break;

        omega = t_dot_s / (t_dot_t + eps);

        // Обновляем x и r
        updateXFullLocal << <gridSize, blockSize, 0, stream >> > (d_x, d_p, d_s, alpha, omega, local_N);
        updateRLocal << <gridSize, blockSize, 0, stream >> > (d_r, d_s, d_t, omega, local_N);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        double local_norm_r = 0.0;
        cublasDnrm2(cublasHandle, local_N, d_r, 1, &local_norm_r);
        double norm_r = 0.0;
        MPI_Allreduce(&local_norm_r, &norm_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if ((norm_r / global_normb) < tol)
            break;

        rho_old = rho;
        iter++;
    }

    // Сбор локальных решений x
    CUDA_CHECK(cudaMemcpyAsync(local_x, d_x, counts_b[rank] * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_x, counts_b[rank], MPI_DOUBLE, x, counts_b, displs_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (iterCount)
        *iterCount = iter;

    // GPU timing (на процессе 0)
    if (rank == 0) {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
        std::cout << "MPI+CUDA GPU time: " << elapsed_ms / 1000.0 << " s" << std::endl;
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    // Освобождение ресурсов GPU
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_global_p));

    cublasDestroy(cublasHandle);
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Освобождение ресурсов CPU
    delete[] local_x;
    delete[] counts;
    delete[] displs;
    delete[] counts_b;
    delete[] displs_b;
    delete[] h_p_buffer;
    delete[] h_global_p;
}
