#ifndef CIRCLE_DETECTION_CUDA_POINT_CUH
#define CIRCLE_DETECTION_CUDA_POINT_CUH


class Point {
public:
    __device__ __host__ explicit Point(float x  = 0.F, float y = 0.f);

    [[nodiscard]] __device__ __host__ float getX() const;

    __device__ __host__ void setX(float x);

    [[nodiscard]] __device__ __host__ float getY() const;

    __device__ __host__ void setY(float y);

private:
    float x, y;
};


#endif //CIRCLE_DETECTION_CUDA_POINT_CUH
