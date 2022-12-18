#include "kernel.cuh"

#include <cuda/std/array>
#include <curand_kernel.h>

#include "Circle.cuh"

__constant__ Point GLOBAL_POINTS[GLOBAL_ARRAY_SIZE];
__constant__ size_t GLOBAL_POINTS_SIZE;
__constant__ float ERROR;

__device__ static cuda::std::array<Point,3> getRandomNumber();
__device__ static void count(Circle& circle);

__global__ void ransac_kernel() {
    const cuda::std::array<Point,3>  randomPoints = getRandomNumber();
    Circle circle = Circle::CircleFromThreePoints(randomPoints);
    count(circle);
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

__device__ static void count(Circle& circle){
    int db = 0;
    for(size_t i = 0; i < GLOBAL_POINTS_SIZE; ++i){
        if (circle.is_point_supported(GLOBAL_POINTS[i], ERROR)) ++db;
    }
    circle.setSupportedPoints(db);
}