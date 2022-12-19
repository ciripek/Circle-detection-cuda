#ifndef CIRCLE_DETECTION_CUDA_CUDAERROR_CUH
#define CIRCLE_DETECTION_CUDA_CUDAERROR_CUH

#include <cstdio>
#include <cstdlib>
#include <fmt/core.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fmt::print(stderr,"GPUassert: {} {} {}\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //CIRCLE_DETECTION_CUDA_CUDAERROR_CUH
