
#include "Circle.cuh"

#include <limits>
#include <cuda/std/cmath>


constexpr static int int_min = std::numeric_limits<int>::min();

__device__ __host__ Circle::Circle() : radius(0.0), supported_points(int_min) {}

__device__ __host__ Circle::Circle(const Point &center, float radius) : center(center), radius(radius),supported_points(int_min) {}

__device__ __host__ Circle Circle::CircleFromThreePoints(const Point &p1, const Point &p2, const Point &p3) {
    float x1 = p1.getX();
    float y1 = p1.getY();
    float x2 = p2.getX();
    float y2 = p2.getY();
    float x3 = p3.getX();
    float y3 = p3.getY();

    float a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2;

    float b = (x1 * x1 + y1 * y1) * (y3 - y2)
              + (x2 * x2 + y2 * y2) * (y1 - y3)
              + (x3 * x3 + y3 * y3) * (y2 - y1);

    float c = (x1 * x1 + y1 * y1) * (x2 - x3)
              + (x2 * x2 + y2 * y2) * (x3 - x1)
              + (x3 * x3 + y3 * y3) * (x1 - x2);

    float x = -b / (2 * a);
    float y = -c / (2 * a);

    return {Point(x,y), cuda::std::hypot(x - x1, y - y1)};
}

__device__ __host__ const Point &Circle::getCenter() const {
    return center;
}

__device__ __host__ void Circle::setCenter(const Point &center) {
    Circle::center = center;
}

__device__ __host__ float Circle::getRadius() const {
    return radius;
}

__device__ __host__ void Circle::setRadius(float radius) {
    Circle::radius = radius;
}

__device__ __host__ int Circle::getSupportedPoints() const {
    return supported_points;
}

__device__ __host__ void Circle::setSupportedPoints(int supportedPoints) {
    supported_points = supportedPoints;
}

__device__ __host__ Circle Circle::CircleFromThreePoints(const cuda::std::array<Point, 3> &arr) {
    return CircleFromThreePoints(arr[0], arr[1], arr[2]);
}

__device__ __host__ bool Circle::is_point_supported(const Point &point, float error) const {
    const float distance = cuda::std::abs(cuda::std::hypot(point.getX() - center.getX(), point.getY() - center.getY()) - radius);
    return distance < error;
}
