#ifndef CIRCLE_DETECTION_CUDA_CIRCLE_CUH
#define CIRCLE_DETECTION_CUDA_CIRCLE_CUH

#include "Point.cuh"

#include <cuda/std/array>
#include <fmt/format.h>

class Circle {
public:
    Circle() = default;
    __device__ __host__ Circle(const Point &center, float radius);
    __device__ __host__ static Circle CircleFromThreePoints(const Point &p1, const Point &p2, const Point &p3);

    __device__ __host__ static Circle CircleFromThreePoints(const cuda::std::array<Point, 3> &arr);

    [[nodiscard]] __device__ __host__ const Point &getCenter() const;

    void __device__ __host__ setCenter(const Point &center);

    [[nodiscard]] __device__ __host__ float getRadius() const;

    __device__ __host__ void setRadius(float radius);

    [[nodiscard]] __device__ __host__ int getSupportedPoints() const;

    __device__ __host__ void setSupportedPoints(int supportedPoints);

    [[nodiscard]] __device__ __host__ bool is_point_supported(const Point &point, float error) const;

private:
    Point center;
    float radius{};
    int supported_points{};
};

template <> struct fmt::formatter<Circle> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Circle& circle, FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "(center: {}, radius: {}, supported points: {})", circle.getCenter(),
                              circle.getRadius(), circle.getSupportedPoints());
    }
};


#endif //CIRCLE_DETECTION_CUDA_CIRCLE_CUH
