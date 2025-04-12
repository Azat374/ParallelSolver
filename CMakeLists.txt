cmake_minimum_required(VERSION 3.10)
project(ParallelSolver LANGUAGES CXX CUDA)

# Поиск необходимых пакетов
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Установка стандарта C++ и флага OpenMP для MSVC
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")

# Установка флагов для компиляции CUDA (укажите архитектуру вашей видеокарты, например, sm_75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

# Явное указание путей к библиотекам CUDA (при необходимости)
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")

# Добавление исполняемого файла и исходных файлов проекта
add_executable(ParallelSolver 
    main.cpp 
    bicgstab_cpu.cpp 
    bicgstab_gpu.cu 
    bicgstab_mpi.cpp 
    bicgstab_mpi_cuda.cu 
    bicgstab_mpi_openmp.cpp 
    utils.cpp 
    preconditioner_cpu.cpp 
    preconditioner_cuda.cu 
    preconditioner_mpi.cpp 
    spmv_kernel.cu
)

target_include_directories(ParallelSolver PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(ParallelSolver PRIVATE MPI::MPI_CXX ${CUDA_LIBRARIES} cublas)

