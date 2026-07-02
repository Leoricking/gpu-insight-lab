// GPU Insight Lab - HIP Vector Add (NOT_VALIDATED)
// AMD HIP example. This code is provided as a portability reference.
// Status: NOT_VALIDATED - no AMD GPU was available during development.
// See native/hip/README.md for build instructions.
#ifdef __HIP__
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

#define HIP_CHECK(call) do {                                           \
    hipError_t _err = (call);                                          \
    if (_err != hipSuccess) {                                          \
        fprintf(stderr, "HIP error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, hipGetErrorString(_err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

__global__ void vector_add_hip(const float* a, const float* b, float* c, int n) {
    int stride = hipBlockDim_x * hipGridDim_x;
    for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N, 0.0f);

    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    hipLaunchKernelGGL(vector_add_hip, dim3(grid_size), dim3(block_size), 0, 0,
                       d_a, d_b, d_c, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_c.data(), d_c, N * sizeof(float), hipMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_c[i] - (h_a[i] + h_b[i]));
        if (diff > max_err) max_err = diff;
    }
    printf("HIP Vector Add: N=%d, max_error=%.6f, %s\n",
           N, max_err, max_err < 1e-5f ? "PASS" : "FAIL");

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return max_err < 1e-5f ? 0 : 1;
}
#else
#include <cstdio>
int main() {
    printf("HIP not available. Build with hipcc to enable AMD GPU support.\n");
    printf("Status: NOT_VALIDATED in GPU Insight Lab v0.1.0\n");
    return 1;
}
#endif
