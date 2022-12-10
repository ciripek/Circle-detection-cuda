#include "Ransac.cuh"

#include <fmt/core.h>

int main(int argc , char** argv)
{
    if (argc != 4) {
        fmt::print(stderr, "Usage: cirle_detection file.text iteration error");
        return EXIT_FAILURE;
    }

    Ransac ransac(std::stoi(argv[2]), std::stof(argv[3]));
    ransac.run(argv[1]);
    return 0;
}