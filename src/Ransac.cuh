#ifndef CIRCLE_DETECTION_CUDA_RANSAC_CUH
#define CIRCLE_DETECTION_CUDA_RANSAC_CUH

#include <utility>
#include <vector>

#include "Point.cuh"
#include "Circle.cuh"

class Ransac {
public:
     Ransac() = default;
     Ransac(int iteration, float error);

     void run(const char* filename);
private:
    int iteration;
    float error;
    Circle bestCircle;

    void read_file(const char* filename,std::vector<Point> &dataPoints);
    void write_to_image(const char* filename, const std::vector<Point>& points);
    std::pair<int, int> getDeviceInfo();
};

#endif //CIRCLE_DETECTION_CUDA_RANSAC_CUH
