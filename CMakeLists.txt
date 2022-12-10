cmake_minimum_required(VERSION 3.24)
project(circle_detection_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(ransac STATIC Ransac.cu Ransac.cuh)
set_target_properties(ransac PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)



add_executable(circle_detection_cuda main.cu)
target_link_libraries(circle_detection_cuda ransac)

set_target_properties(circle_detection_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
