#include "Ransac.cuh"

#include <fstream>
#include <iterator>
#include <fmt/ranges.h>
#include <limits>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cudaError.cuh"
#include "kernel.cuh"

constexpr static size_t point_size = sizeof(Point);

Ransac::Ransac(int iteration, float error) : iteration(iteration), error(error) {}

void Ransac::read_file(const char *filename, std::vector<Point> &dataPoints) {
    std::ifstream inputstream(filename);
    dataPoints.assign(std::istream_iterator<Point>(inputstream), std::istream_iterator<Point>());
}

void Ransac::write_to_image(const char *filename, const std::vector<Point> &points) {
    cv::Mat mat = cv::Mat::zeros(1920, 2560, CV_8UC3);

    const cv::Point2f  point {bestCircle.getCenter().getX(),bestCircle.getCenter().getY()};
    cv::circle(mat, point, bestCircle.getRadius(), cv::Scalar( 0, 0, 255 ));

    for (const Point &p: points) {
        cv::Vec3b pixel;
        pixel.val[0] = 255;
        pixel.val[1] = 0,
                pixel.val[1] = 0;
        mat.at<cv::Vec3b>(p.getY(),p.getX())=pixel;
    }

    cv::imwrite(filename,mat);
}


std::pair<int, int> Ransac::getDeviceInfo() {
    constexpr int int_min = std::numeric_limits<int>::min();
    std::pair<int, int> deviceInfo{int_min, int_min};
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fmt::print(stderr, "There are no available device(s) that support CUDA\n");
        exit(EXIT_FAILURE);
    }
#ifdef DEBUG
    else {
        fmt::print("Detected {} CUDA Capable device(s)\n", deviceCount);
    }
#endif
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (deviceProp.maxThreadsPerBlock > deviceInfo.second) {
            deviceInfo.first = dev;
            deviceInfo.second = deviceProp.maxThreadsPerBlock;
        }
#ifdef DEBUG
        fmt::print("Device {}: {} \n", dev, deviceProp.name);
        fmt::print("Compute capability: {}.{}\n", deviceProp.major, deviceProp.minor);
        fmt::print("MaxThreadsPerBlock: {} \n", deviceProp.maxThreadsPerBlock);
        fmt::print("MaxThreadDim ({},{},{})  \n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
                   deviceProp.maxThreadsDim[2]);
        fmt::print("MaxGridSize ({},{},{})  \n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
                   deviceProp.maxGridSize[2]);
#endif
    }

    CUDA_CHECK(cudaDeviceReset());
    return deviceInfo;
}

void Ransac::run(const char *filename) {
    std::vector<Point> points;
    read_file(filename, points);
#ifdef DEBUG
    fmt::print("Points = {}\n", points);
#endif
    const auto &[dev, maxThreadsPerBlock] = getDeviceInfo();
    cudaSetDevice(dev);

    size_t byte = points.size() > GLOBAL_ARRAY_SIZE ? GLOBAL_ARRAY_SIZE * point_size : points.size() * point_size;
    size_t numberofelements = points.size();
    CUDA_CHECK(cudaMemcpyToSymbol(GLOBAL_POINTS, points.data(), byte));
    CUDA_CHECK(cudaMemcpyToSymbol(GLOBAL_POINTS_SIZE, &numberofelements, sizeof(numberofelements)));
    CUDA_CHECK(cudaMemcpyToSymbol(ERROR, &error, sizeof(error)));

#if defined(USE_SEMAPHORE)
    Circle* circle;
    cudaMallocManaged(&circle, sizeof(Circle));
    *circle = Circle{};
#else
    Circle* circle;
    CUDA_CHECK(cudaMalloc(&circle, iteration * maxThreadsPerBlock * sizeof(Circle)));
#endif

    ransac_kernel<<<iteration, maxThreadsPerBlock>>>(circle);
    CUDA_CHECK(cudaDeviceSynchronize());

#if defined(USE_SEMAPHORE)
    bestCircle = *circle;
#else
    int dbThreads = maxThreadsPerBlock;
    int dbBlock = iteration;

    while (dbBlock > 1){
        max_search<<<dbBlock, dbThreads, dbThreads * sizeof(Circle)>>>(circle);
        CUDA_CHECK(cudaDeviceSynchronize());
        dbBlock /= dbThreads;
    }
    max_search<<<1, dbThreads, dbThreads * sizeof(Circle)>>>(circle);
    CUDA_CHECK(cudaMemcpy(&bestCircle, circle, sizeof(Circle), cudaMemcpyDeviceToHost));
#endif

    cudaFree(circle);

    fmt::print("The best circle: {}", bestCircle);
    write_to_image("test.png", points);
}
