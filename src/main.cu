#include "Ransac.cuh"

#include <fmt/core.h>

int main(int argc , char** argv)
{

    if (argc == 2){
        Ransac ransac;
        ransac.run(argv[1]);
    } else if (argc == 4){
        Ransac ransac(std::stoi(argv[2]), std::stof(argv[3]));
        ransac.run(argv[1]);
    } else {
        fmt::print(stderr, "Usage: circle_detection file.text iteration error\n"
                           "Usage: circle_detection file.text");
        return EXIT_FAILURE;
    }
    return 0;
}