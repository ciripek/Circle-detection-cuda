#include "Ransac.cuh"

#include <fstream>
#include <iterator>

#ifdef DEBUG
#include <fmt/ranges.h>
#endif

Ransac::Ransac(int iteration, float error) : iteration(iteration), error(error) {}

void Ransac::read_file(const char *filename, std::vector<Point> &dataPoints) {
    std::ifstream inputstream(filename);
    dataPoints.assign(std::istream_iterator<Point>(inputstream), std::istream_iterator<Point>());
}

void Ransac::run(const char *filename) {
    std::vector<Point> points;
    read_file(filename, points);
#ifdef DEBUG
    fmt::print("{}", points);
#endif
}
