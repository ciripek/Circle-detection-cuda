cmake_minimum_required(VERSION 3.24)
project(circle_detection_cuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

find_package(fmt REQUIRED)

if (USE_SEMAPHORE)
    add_definitions(-DUSE_SEMAPHORE)
endif (USE_SEMAPHORE)

if (DEBUG)
    add_definitions(-DDEBUG=1)
endif (DEBUG)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(ransac STATIC src/Ransac.cu src/Ransac.cuh src/Point.cu src/Point.cuh src/Circle.cu src/Circle.cuh src/cudaError.cuh src/kernel.cu src/kernel.cuh)
set_target_properties(ransac PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ransac fmt::fmt ${OpenCV_LIBS})

target_compile_features(ransac PUBLIC cxx_std_17)



add_executable(circle_detection_cuda src/main.cu)
target_link_libraries(circle_detection_cuda ransac fmt::fmt)
target_compile_features(circle_detection_cuda PUBLIC cxx_std_17)

set_target_properties(circle_detection_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
