#ifndef CIRCLE_DETECTION_CUDA_RANSAC_CUH
#define CIRCLE_DETECTION_CUDA_RANSAC_CUH


class Ransac {
public:
    Ransac(int iteration, float error);
private:
    int iteration;
    float error;
};


#endif //CIRCLE_DETECTION_CUDA_RANSAC_CUH
