// GPU Insight Lab - CUDA Error Checking Macro
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr,                                                     \
                    "[GPU Insight Lab] CUDA error at %s:%d: %s (%d)\n",        \
                    __FILE__, __LINE__,                                         \
                    cudaGetErrorString(_err), static_cast<int>(_err));         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Non-fatal version that returns false on error
inline bool cuda_check_nofatal(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU Insight Lab] CUDA error at %s:%d: %s\n",
                file, line, cudaGetErrorString(err));
        return false;
    }
    return true;
}

#define CUDA_CHECK_NF(call) cuda_check_nofatal((call), __FILE__, __LINE__)
