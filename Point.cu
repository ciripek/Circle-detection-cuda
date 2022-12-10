#include "Point.cuh"

__device__ __host__ Point::Point(float x, float y) : x(x), y(y) {}

__device__ __host__ float Point::getX() const {
    return x;
}

__device__ __host__ void Point::setX(float x) {
    Point::x = x;
}

__device__ __host__ float Point::getY() const {
    return y;
}

__device__ __host__ void Point::setY(float y) {
    Point::y = y;
}

__host__ std::ostream &operator<<(std::ostream &os, const Point &point) {
    os << "x: " << point.x << " y: " << point.y;
    return os;
}

__host__ std::istream &operator>>(std::istream &is, Point &point) {
    is >> point.x >> point.y;
    return is;
}
