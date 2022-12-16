#include "kernel.cuh"

#include <cuda/std/array>
#include <curand_kernel.h>

__constant__ Point GLOBAL_POINTS[GLOBAL_ARRAY_SIZE];
__constant__ size_t GLOBAL_POINTS_SIZE;

__device__ static cuda::std::array<Point,3> getRandomNumber();

__global__ void ransac_kernel() {
    const cuda::std::array<Point,3>  randomPoints = getRandomNumber();
}

__device__ static cuda::std::array<Point,3> getRandomNumber() {
    curandState state;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(clock64(), id, 0, &state);
    int index1 = -1, index2 = -1, index3 = -1;

    index1 = curand(&state) % GLOBAL_POINTS_SIZE;

    do{
        index2 = curand(&state) % GLOBAL_POINTS_SIZE;
    } while (index1 == index2);

    do {
        index3 = curand(&state) % GLOBAL_POINTS_SIZE;
    } while (index3 == index1 || index3 == index2);

    return {
            GLOBAL_POINTS[index1],
            GLOBAL_POINTS[index2],
            GLOBAL_POINTS[index3]
    };
}