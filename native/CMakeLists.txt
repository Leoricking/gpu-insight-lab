cmake_minimum_required(VERSION 3.18)
project(gpu_insight_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Optional: set CUDA architectures (auto-detect or set explicitly)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Source files
set(CUDA_SOURCES
    cuda/benchmark_main.cu
    cuda/device_info.cu
    cuda/vector_add.cu
    cuda/reduction.cu
    cuda/transpose.cu
    cuda/gemm_naive.cu
    cuda/gemm_tiled.cu
    cuda/memory_bandwidth.cu
    cuda/stream_pipeline.cu
    cuda/image_grayscale.cu
)

set(CPP_SOURCES
    common/json_writer.cpp
    common/statistics.cpp
    common/benchmark_result.cpp
)

# Build executable
add_executable(gpu_insight_benchmark
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

target_include_directories(gpu_insight_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

target_link_libraries(gpu_insight_benchmark PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

# Windows: link against Windows API for hostname
if(WIN32)
    target_link_libraries(gpu_insight_benchmark PRIVATE ws2_32)
endif()

# CUDA compile options
target_compile_options(gpu_insight_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --generate-line-info
    >
)

# Output directory
set_target_properties(gpu_insight_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "GPU Insight Lab native benchmark configured")
message(STATUS "Output: ${CMAKE_BINARY_DIR}/bin/gpu_insight_benchmark")
