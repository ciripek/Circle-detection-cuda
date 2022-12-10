#ifndef CIRCLE_DETECTION_CUDA_POINT_CUH
#define CIRCLE_DETECTION_CUDA_POINT_CUH


#include <iostream>

class Point {
public:
    __device__ __host__ explicit Point(float x  = 0.F, float y = 0.f);

    [[nodiscard]] __device__ __host__ float getX() const;

    __device__ __host__ void setX(float x);

    [[nodiscard]] __device__ __host__ float getY() const;

    __device__ __host__ void setY(float y);

    friend __host__ std::ostream &operator<<(std::ostream &os, const Point &point);
    friend __host__ std::istream &operator>>(std::istream &is, Point& point);

private:
    float x, y;
};


#endif //CIRCLE_DETECTION_CUDA_POINT_CUH
