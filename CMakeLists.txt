cmake_minimum_required(VERSION 3.10)
project(ParallelSolver LANGUAGES CXX CUDA)

# Поиск необходимых пакетов
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Устанавливаем стандарт C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Поддержка OpenMP (для MSVC используем /openmp, для остальных -fopenmp)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Указываем пути к библиотекам CUDA
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)

# Добавляем исполняемый файл и исходные файлы проекта
add_executable(ParallelSolver 
    main.cpp 
    bicgstab.cu 
    bicgstab_cpu.cpp 
    utils.cpp 
    preconditioner.cu 
    preconditioner_cpu.cpp
)

# Включаем заголовочные файлы (из корневой директории проекта)
target_include_directories(ParallelSolver PRIVATE ${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

# Линкуем с библиотеками CUDA и MPI
target_link_libraries(ParallelSolver PRIVATE ${CUDA_LIBRARIES} MPI::MPI_CXX)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_60 -code=sm_60")
