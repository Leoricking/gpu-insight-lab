cmake_minimum_required(VERSION 3.18)
project(GPUInsightLab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Add native subproject
add_subdirectory(native)

message(STATUS "GPU Insight Lab root CMake configured")
