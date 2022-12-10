#ifndef CIRCLE_DETECTION_CUDA_RANSAC_CUH
#define CIRCLE_DETECTION_CUDA_RANSAC_CUH

#include <vector>

#include "Point.cuh"

class Ransac {
public:
     Ransac() = default;
     Ransac(int iteration, float error);
private:
    int iteration;
    float error;

    void read_file(const char* filename,std::vector<Point> &dataPoints);
};


#endif //CIRCLE_DETECTION_CUDA_RANSAC_CUH
