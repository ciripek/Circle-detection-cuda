#include "kernel.cuh"

#include <cuda/std/array>
#if defined(USE_SEMAPHORE)
#include <cuda/semaphore>
#endif
#include <curand_kernel.h>

#include "Circle.cuh"

__constant__ Point GLOBAL_POINTS[GLOBAL_ARRAY_SIZE];
__constant__ size_t GLOBAL_POINTS_SIZE;
__constant__ float ERROR;

__device__ static cuda::std::array<Point,3> getRandomNumber();
__device__ static void count(Circle& circle);
#if defined(USE_SEMAPHORE)
__device__  cuda::binary_semaphore<cuda::thread_scope_device> binarySemaphore{1};
__device__ static void max(Circle* bestCircle, const Circle& circle);
#endif

__global__ void ransac_kernel(Circle* bestCircle) {
    const cuda::std::array<Point,3>  randomPoints = getRandomNumber();
    Circle circle = Circle::CircleFromThreePoints(randomPoints);
    count(circle);
#if defined(USE_SEMAPHORE)
    max(bestCircle, circle);
#else
    unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
    bestCircle[id] = circle;
#endif
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

#if defined(USE_SEMAPHORE)
__device__ static void max(Circle* bestCircle, const Circle& circle){
    binarySemaphore.acquire();

    if (bestCircle->getSupportedPoints() < circle.getSupportedPoints()){
        *bestCircle = circle;
    }

    binarySemaphore.release();
}
#else
__global__ void max_search(Circle* circles){
    extern __shared__ Circle data[];
    unsigned global_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned local_id = threadIdx.x;

    data[local_id] = circles[global_id];
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            unsigned left = local_id;
            unsigned right = left + stride;

            if (data[right].getSupportedPoints() > data[left].getSupportedPoints()){
                data[left] = data[right];
            }
        }
        __syncthreads();
    }

    if (local_id == 0) {
        circles[blockIdx.x] = data[0];
    }

}
#endif
