#ifndef CIRCLE_DETECTION_CUDA_POINT_CUH
#define CIRCLE_DETECTION_CUDA_POINT_CUH


#include <iostream>
#include <fmt/format.h>


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


template <> struct fmt::formatter<Point> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Point& p, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "(x: {}, y: {})", p.getX(), p.getY());
    }
};

#endif //CIRCLE_DETECTION_CUDA_POINT_CUH
