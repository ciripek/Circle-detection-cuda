#ifndef CIRCLE_DETECTION_CUDA_KERNEL_CUH
#define CIRCLE_DETECTION_CUDA_KERNEL_CUH

#include "Point.cuh"
#include "Circle.cuh"

#ifndef GLOBAL_ARRAY_SIZE
#define GLOBAL_ARRAY_SIZE 1024
#endif

extern __constant__ Point GLOBAL_POINTS[GLOBAL_ARRAY_SIZE];
extern __constant__ size_t GLOBAL_POINTS_SIZE;
extern __constant__ float ERROR;

__global__ void ransac_kernel(Circle* bestCircle);

#endif //CIRCLE_DETECTION_CUDA_KERNEL_CUH
